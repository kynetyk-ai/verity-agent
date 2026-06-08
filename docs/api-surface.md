# Verity API Surface

Reference for Verity's **in-process async Python API** — there is no HTTP layer. The "API" is the
control-plane service class plus the typed **port contracts** that cross the service boundaries to
the sandbox and verifier. Everything here is checkable against source; locations are cited as
`path:line`.

> Naming note: the orientation prompt for this task referred to a class `ControlPlaneApi`. The actual
> public class is **`ControlPlane`** (`src/verity/control_plane/api.py:125`). There is no
> `ControlPlaneApi`. This document uses the real name.

The control plane is the **only** service that mutates durable state (spec §3.4). It is async by
design: external calls are coroutines so today's in-process stubs and tomorrow's networked services
share one shape. The proven §7 commit path stays synchronous and runs in a worker thread while the
API awaits the advisory verifier (`api.py:1-22`, `api.py:230-260`).

The loop is **read → propose → gate → commit**.

---

## Endpoint index

`ControlPlane` (`src/verity/control_plane/api.py:125`):

| Method | Async? | One-line purpose | Mutates durable state? |
|---|---|---|---|
| `__init__` | sync | Construct the service over a store, assembler, policy, registries, clock | No (in-memory wiring) |
| `configure(config)` | **async** | Register a task: stamp schema version, resolve + provision sandbox/verifier | Yes (writes a `schema_versions` row) |
| `teardown(task_id)` | **async** | Tear down a task's sandbox and verifier | No (store untouched; service lifecycle only) |
| `serve_context(task_id, *, goal, scratch, feedback)` | **async** | Assemble §9 bounded context and hand it to the sandbox | No (pure read of the store) |
| `submit_proposal(task_id, envelope)` | **async** | Shape-check → harvest objects → propose → run commit path | Yes (object store, `proposed` row, decisions, status) |
| `run_cycle(task_id, *, goal, feedback)` | **async** | One full read→propose→gate→commit cycle, then regenerate sandbox | Yes (via `submit_proposal`) |
| `run(task_id, *, goal)` | **async** | Drive cycles until the orchestration policy stops | Yes (via `run_cycle`) |
| `accepted_artifacts(*, type=None)` | sync | Extract accepted artifacts | No (read) |
| `provenance(artifact_id)` | sync | Full lineage + decisions behind an artifact | No (read) |
| `rejected_log(*, type=None)` | sync | The rejected-artifact log | No (read) |
| `superseded_log(*, type=None)` | sync | The superseded-artifact log | No (read) |
| `revised_log(*, type=None)` | sync | The revised-artifact log | No (read) |
| `rationale_for(artifact_id)` | sync | The agent's how/why summary (never a gate input) | No (read of in-memory rationale channel) |

Private helpers (`_task`, `_run_commit`, `_link_revision_if_any`) are not part of the public surface
and are documented inline only where they explain an endpoint's downstream effect.

---

## Construction & lifecycle

### `ControlPlane.__init__`

`src/verity/control_plane/api.py:128-144`

```python
def __init__(
    self,
    store: SqliteStore,
    *,
    assembler: ContextAssembler | None = None,
    policy: OrchestrationPolicy | None = None,
    sandbox_providers: ProviderRegistry[SandboxPort] = SANDBOX_PROVIDERS,
    verifier_providers: ProviderRegistry[VerifierPort] = VERIFIER_PROVIDERS,
    clock: Clock = default_clock,
) -> None: ...
```

| Param | Type | Meaning |
|---|---|---|
| `store` | `SqliteStore` | The sole durable store (`store.py:254`). Implements both `Store` (reads/propose/objects) and `CommitSink` (privileged writes). |
| `assembler` | `ContextAssembler \| None` | §9 context assembler; defaults to `ContextAssembler()` with the bounded-context caps (`context.py:108`). |
| `policy` | `OrchestrationPolicy \| None` | The run-again/stop rule; defaults to `OrchestrationPolicy()` (`api.py:96`). |
| `sandbox_providers` | `ProviderRegistry[SandboxPort]` | Registry that resolves `config.sandbox_key` → a `SandboxPort`. Defaults to module-level `SANDBOX_PROVIDERS` (`ports.py:189`). |
| `verifier_providers` | `ProviderRegistry[VerifierPort]` | Registry that resolves `config.verifier_key` → a `VerifierPort`. Defaults to `VERIFIER_PROVIDERS` (`ports.py:190`). |
| `clock` | `Clock` | Injected ISO-8601 timestamp source for determinism (`store.py:120`). |

**Effect**: pure in-memory wiring; holds an empty `_tasks: dict[str, TaskState]`. No store writes.

**Ordering constraint**: providers for the keys named in any `TaskConfig` you later `configure`
must already be `register`ed on the supplied registries, or `configure` raises `ProviderError`.

```python
import asyncio
from verity.contracts import ProviderRegistry, SandboxPort, VerifierPort
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.store import SqliteStore

store = SqliteStore()  # in-memory; pass path="..."/object_dir=Path(...) to persist

sandbox_providers: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
verifier_providers: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
sandbox_providers.register("stub", lambda: MySandbox())   # zero-arg factory → SandboxPort
verifier_providers.register("stub", lambda: MyVerifier())  # zero-arg factory → VerifierPort

cp = ControlPlane(
    store,
    policy=OrchestrationPolicy(max_cycles=5, stop_on_accept=True),
    sandbox_providers=sandbox_providers,
    verifier_providers=verifier_providers,
)
```

### `OrchestrationPolicy`

`src/verity/control_plane/api.py:96-111`

```python
@dataclass(frozen=True, slots=True)
class OrchestrationPolicy:
    max_cycles: int = 20      # bounds the run
    refine_cap: int = 3       # max refines on one lineage before terminal reject (§12 seam)
    stop_on_accept: bool = False  # end the run as soon as an artifact is accepted
    def should_continue(self, *, cycles_run: int) -> bool: ...  # cycles_run < max_cycles
```

`refine_cap` bounds how many times a single lineage may be refined before it terminates in
`rejected`. It is enforced **per lineage at commit** (issue #17): `ControlPlane._count_lineage_refines`
walks the store's `revises` chain and passes `refine_exhausted` into `run_commit`, which converts a
capped refine into a reject. (`TaskState.refine_counts` was removed in favour of this store-derived
count.)

### `TaskState`

`src/verity/control_plane/api.py:114-122` — internal per-task record (not constructed by callers):
holds the `config`, the resolved `sandbox` and `verifier`, and the segregated `rationale` channel
(`dict[artifact_id, str]`).

---

## Configure-by-task

### `configure`

`src/verity/control_plane/api.py:148-160`

```python
async def configure(self, config: TaskConfig) -> None: ...
```

**Does**: registers a task. In order (`api.py:150-160`):
1. Reads `store.current_schema_version()`; computes the next version (`current.version + 1`, else `1`).
2. Writes a `schema_versions` row via `store.register_schema_version(config.schema.snapshot(...))`
   — a stamped snapshot of the domain's declared types + operation signatures (`registries.py:133`).
3. Resolves the sandbox via `sandbox_providers.create(config.sandbox_key)` and the verifier via
   `verifier_providers.create(config.verifier_key)`.
4. `await`s `sandbox.provision()` then `verifier.provision()` (the §3.4 lifecycle surface).
5. Stores a `TaskState` under `config.task_id` and logs `task_configured`.

**Downstream effect**: one durable `schema_versions` row; two services provisioned; an entry added
to the in-memory `_tasks` map. Re-configuring stamps a *new* schema version (the counter only
increments).

**Errors**: `ProviderError` if `sandbox_key`/`verifier_key` are unregistered (`ports.py:175-182`).

**Precondition**: the named providers must be registered on the registries passed to `__init__`.

```python
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import DefaultRetrievalPolicy
from verity.domains.fake import build_fake_domain

domain = build_fake_domain()  # schema + gated_types + shape_validator (Source/Note/Orphan)
config = TaskConfig(
    task_id="t1",
    instructions="write notes",
    domain_instructions="a Note has a text field",
    schema=domain.schema,
    gated_types=domain.gated_types,
    retrieval=DefaultRetrievalPolicy(),
    shape_validator=domain.shape_validator,
    sandbox_key="stub",
    verifier_key="stub",
)
await cp.configure(config)        # inside an async context
# or, top-level: asyncio.run(cp.configure(config))
```

### `teardown`

`src/verity/control_plane/api.py:162-165`

```python
async def teardown(self, task_id: str) -> None: ...
```

**Does**: `await`s `task.sandbox.teardown()` then `task.verifier.teardown()` for the named task.
Does **not** remove the task from `_tasks`, mutate the store, or release the schema version.

**Errors**: `UnknownTask` if `task_id` was never configured (`api.py:167-170`).

```python
await cp.teardown("t1")
```

---

## Cycle control

### `serve_context`

`src/verity/control_plane/api.py:174-191`

```python
async def serve_context(
    self, task_id: str, *, goal: str, scratch: str = "", feedback: str = ""
) -> ServedContext: ...
```

**Does**: assembles the §9 bounded context from the store and serves it to the sandbox.
1. `ContextAssembler.assemble(store, system_prompt=config.system_prompt(), goal=..., scratch=...,
   retrieval=config.retrieval)` — regenerates a stable prefix (composed 3-layer system prompt +
   ranked manifest) and a volatile tail (the task retrieval policy's selection + goal + scratch),
   capped by the assembler's bounded-context limits (`context.py:139-177`).
2. Wraps it in a `ServedContext(system_prompt=prefix, tail=volatile_tail, feedback=feedback)`.
3. `await`s `task.sandbox.serve_context(served)` and returns the `ServedContext`.

**Downstream effect**: read-only against the store; pushes context into the sandbox runtime. The
task's own `retrieval` policy (not the assembler default) drives the tail (`api.py:183`, verified by
`tests/test_api.py:215-243`).

**Returns**: the `ServedContext` that was served (so callers can inspect it).

**Errors**: `UnknownTask` for an unconfigured task.

```python
served = await cp.serve_context("t1", goal="find roots", scratch="", feedback="")
assert "Store manifest" in served.system_prompt
```

### `submit_proposal`

`src/verity/control_plane/api.py:195-228`

```python
async def submit_proposal(
    self, task_id: str, envelope: ProposalEnvelope
) -> IntakeResult: ...
```

This is the **intake "endpoint"** — the single path that turns a sandbox proposal into durable
provenance. Step by step (`api.py:199-228`):

1. **Shape check (§7.0, pre-gate).** Calls `config.shape_validator(envelope.artifact)`. If it
   returns a `ShapeError`, the method logs `intake_shape_error` and returns
   `IntakeResult(entered_protocol=False, shape_error=...)` — **nothing is recorded**: no object
   harvested, no `proposed` row, no decision, and the verifier is never dispatched
   (`tests/test_api.py:153-167`).
2. **Harvest before teardown.** `harvested = {name: store.put_object(data) for name, data in
   envelope.objects.items()}` — content-addresses each outbox object into the object store. This
   ordering (harvest before the sandbox regenerates) is the hard constraint.
3. **Object sidecar.** If anything was harvested, replaces the artifact with
   `replace(artifact, objects=tuple(harvested.items()))` — a control-plane-owned sidecar; the domain
   `payload` is stored verbatim and never reached into (`tests/test_api.py:178-196`).
4. **Rationale segregation.** If `envelope.metadata` is non-empty, stores it on
   `task.rationale[artifact.id]` — the provenance/context channel, **never** sent to a gate (§10).
5. **Propose.** `store.propose(artifact, envelope.operation)` writes a `proposed` artifact row + its
   producing `operations` row.
6. **Revision linkage.** `_link_revision_if_any(envelope)` — if the operation is `revises` and its
   first parent is currently `REVISED`, calls `store.link_revision(...)` to set `revised_by`
   (`api.py:262-273`).
7. **Commit.** `await self._run_commit(task, artifact.id, envelope.objects)` runs the synchronous
   §7 commit path in a worker thread, bridging the one opaque verifier handoff (see below). Logs
   `proposal_committed` and returns `IntakeResult(entered_protocol=True, commit=..., harvested=...)`.

**The verifier handoff** (`_run_commit`, `api.py:230-260`): a `dispatch` closure resolves the
declared store-inputs for the artifact's type (`config.gated_types.resolve(type)`), cuts the
rationale-free `store_slice` via `resolve_declared_slice` (`independence.py:65`), builds a
`VerifierRequest(proposal, store_slice, objects)`, and calls `verifier.dispatch(...)` across the
thread boundary with `asyncio.run_coroutine_threadsafe`. `run_commit` then records every decision,
validates the bundle is self-consistent and the transition is legal, and sets the status
(`commit.py:148-221`). One handoff per proposal — not one per gate (`tests/test_api.py:142`).

**Downstream durable effects**, by verdict bundle status:

| Bundle status | Store effect | `CommitResult.outcome` |
|---|---|---|
| `REJECTED` | `set_status → rejected` | `REJECTED` |
| `REVISED` | `set_status → revised`; `CommitResult.defects` carries the localized defect list | `REVISED` |
| `TENTATIVE` | provenance ensured, `set_status → tentative` | `TENTATIVE` |
| `ACCEPTED` | provenance ensured, `set_status → accepted`; if `bundle.supersedes` is set, `accept_superseding` flips the incumbent to `superseded` atomically | `ACCEPTED` |

In all non-shape-error cases a `decisions` row is written for every `GateDecision` in the bundle.

**Returns**: `IntakeResult` (`api.py:80-93`):

```python
@dataclass(frozen=True, slots=True)
class IntakeResult:
    entered_protocol: bool                    # False iff shape-error (nothing recorded)
    shape_error: ShapeError | None = None     # the correction, when malformed
    commit: CommitResult | None = None        # the §7 result, when it entered the protocol
    harvested: dict[str, ObjectRef] = {}      # outbox name → content-addressed ref
```

**Errors**: `UnknownTask`; and from the commit path (`commit.py`): `NoImplicitAccept` (type not
gated, §5.7), `ProposerIsGate` (verifier identity == `artifact.created_by`, §7.2),
`BundleInconsistent` (a reject/refine decision under a non-matching status), `CommitError` (unknown
incumbent to supersede, or a non-root artifact accepted with no provenance edge),
`IllegalTransition` (the bundle's status is not a legal edge from `proposed`),
`IndependenceViolation` (a declared store-input has no provider). `StoreError` if an id/op-id is
reused.

```python
from verity.contracts import Artifact, ArtifactStatus, Operation, OperationStatus, ProposalEnvelope
from verity.domains.fake import NOTE

envelope = ProposalEnvelope(
    artifact=Artifact(
        id="n1", type=NOTE, payload={"text": "hi"},
        status=ArtifactStatus.PROPOSED, created_by="agent", created_at="",
    ),
    operation=Operation(
        op_id="op-n1", op_name="author", parents=("src",), output_id="n1",
        status=OperationStatus.SUCCESS, created_at="",
    ),
    metadata="i derived this from the source",          # rationale → segregated channel
    objects={"submission.py": b"def feature(df): return df"},  # outbox → harvested
)

result = await cp.submit_proposal("t1", envelope)
assert result.entered_protocol is True
assert result.commit.outcome.value == "accepted"
ref = result.harvested["submission.py"]                 # ObjectRef(blob_ref="sha256:...", content_hash=...)
```

### `run_cycle`

`src/verity/control_plane/api.py:277-285`

```python
async def run_cycle(self, task_id: str, *, goal: str, feedback: str = "") -> IntakeResult: ...
```

**Does**: one complete read→propose→gate→commit cycle:
1. `await serve_context(task_id, goal=goal, feedback=feedback)` (read).
2. `envelope = await task.sandbox.collect_proposal()` (propose — pulls the proposal + outbox).
3. `result = await submit_proposal(task_id, envelope)` (gate + commit).
4. `await task.sandbox.regenerate()` — discards the ephemeral workspace (one ephemerality event,
   §3.5). Harvest already happened inside `submit_proposal`, so it is safe to regenerate now.

**Returns**: the `IntakeResult` from the embedded `submit_proposal`.

**Errors**: as `submit_proposal`, plus whatever the sandbox raises.

```python
result = await cp.run_cycle("t1", goal="make a good note", feedback="")
```

### `run`

`src/verity/control_plane/api.py:287-308`

```python
async def run(self, task_id: str, *, goal: str) -> list[IntakeResult]: ...
```

**Does**: drives `run_cycle` repeatedly while `policy.should_continue(cycles_run=...)` (i.e. while
`cycles < max_cycles`). After each cycle it appends the result, increments the counter, and — if
`policy.stop_on_accept` and the cycle's `commit.status is ACCEPTED` — breaks early. Otherwise it
computes `feedback` from the result (`_feedback_from`, `api.py:336-342`: a `shape-error: ...`
message, or `refine: <defects>`, or empty) and feeds it into the next cycle's served context.
Logs `run_complete`.

**Returns**: the list of per-cycle `IntakeResult`s.

```python
results = await cp.run("t1", goal="make a good note")
# with stop_on_accept=True and a sandbox queuing two proposals, len(results) == 1 on first accept
```

---

## Extraction (read-only)

All synchronous; all pure reads.

### `accepted_artifacts`

`src/verity/control_plane/api.py:312-313`

```python
def accepted_artifacts(self, *, type: str | None = None) -> list[Artifact]: ...
```
Returns `store.query_artifacts(type=type, status=ACCEPTED)` — the accepted artifacts, optionally
filtered by type, ordered by `(created_at, id)`.

```python
notes = cp.accepted_artifacts(type="Note")
```

### `provenance`

`src/verity/control_plane/api.py:315-317`

```python
def provenance(self, artifact_id: str) -> Provenance: ...
```
Returns the full transitive lineage + gathered decisions behind an artifact — the "why do we believe
X" answer (`store.get_provenance`, `store.py:518-546`). `Provenance` carries `artifact`,
`operations`, `ancestors`, and `decisions` (`store.py:103-114`).

**Errors**: `StoreError` if the artifact is unknown (`store.py:521`).

```python
prov = cp.provenance("n1")
assert "src" in {a.id for a in prov.ancestors}
assert any(d.verdict.value == "accept" for d in prov.decisions)
```

### `rejected_log` / `superseded_log` / `revised_log`

`src/verity/control_plane/api.py:319-326`

```python
def rejected_log(self,   *, type: str | None = None) -> list[Artifact]: ...
def superseded_log(self, *, type: str | None = None) -> list[Artifact]: ...
def revised_log(self,    *, type: str | None = None) -> list[Artifact]: ...
```
The terminal/lineage logs: `rejected`, `superseded`, and `revised` artifacts respectively (the first
two delegate to `store.rejected_log`/`store.superseded_log`; `revised_log` queries by the `REVISED`
status). Optional `type` filter.

```python
for art in cp.rejected_log(type="Note"):
    print(art.id, art.status.value)
```

### `rationale_for`

`src/verity/control_plane/api.py:328-333`

```python
def rationale_for(self, artifact_id: str) -> str | None: ...
```
Returns the agent's how/why summary captured at intake from `envelope.metadata`, by scanning every
task's segregated `rationale` channel. Returns `None` if no rationale was stored. This is **never** a
gate input (it lives off the `VerifierRequest` by construction).

```python
why = cp.rationale_for("n1")  # "i derived this from the source", or None
```

---

## Errors

`src/verity/control_plane/api.py:72-77`

```python
class ControlPlaneError(RuntimeError):  # a misuse of the control-plane API
class UnknownTask(ControlPlaneError):   # operation referenced an unconfigured task
```

Commit-path errors surface through `submit_proposal` (`commit.py:115-143`): `CommitError`,
`NoImplicitAccept`, `ProposerIsGate`, `BundleInconsistent`; lifecycle: `IllegalTransition`
(`lifecycle.py:48`); independence: `IndependenceViolation` (`independence.py:37`); provider
resolution: `ProviderError` (`ports.py:47`); store invariants: `StoreError` (`store.py:140`).

---

## Port contracts (the cross-service surface)

The control plane talks to the sandbox and verifier **only** through these ports
(`src/verity/contracts/ports.py`), and selects implementations by task config through a
`ProviderRegistry`. Every port call is `async` so a network hop is not a refactor. All ports also
expose the no-op-able lifecycle surface (`provision` / `teardown` / `health`).

### `ServiceLifecycle` (Protocol)

`ports.py:54-64`

```python
async def provision(self) -> None: ...
async def teardown(self) -> None: ...
async def health(self) -> bool: ...
```
The optional lifecycle the control plane may drive (start/stop/health-check). In-process
implementations satisfy these as no-ops.

### `VerifierPort` (Protocol)

`ports.py:86-101`

```python
class VerifierPort(Protocol):
    identity: str
    async def dispatch(self, request: VerifierRequest) -> VerdictBundle: ...
    async def provision(self) -> None: ...
    async def teardown(self) -> None: ...
    async def health(self) -> bool: ...
```
The advisory, **opaque** verifier. `dispatch` is the only path to a verdict — one proposal in, one
`VerdictBundle` out; the verifier owns which checks run and in what order. `identity` must differ
from the proposer's (`created_by`), enforced at commit (Builder/Breaker, §7.2). The sandbox has no
path to it.

### `VerifierRequest`

`ports.py:70-83`

```python
@dataclass(frozen=True, slots=True)
class VerifierRequest:
    proposal: Artifact
    store_slice: tuple[Artifact, ...] = ()
    objects: Mapping[str, bytes] = {}
```

| Field | Type | Meaning |
|---|---|---|
| `proposal` | `Artifact` | The artifact under test. |
| `store_slice` | `tuple[Artifact, ...]` | The declared, control-plane-cut slice (incumbents, rejected-log) for this type. A tuple of `Artifact` — which has **no rationale field**, so the proposer's reasoning cannot ride along (§10). |
| `objects` | `Mapping[str, bytes]` | Harvested attachments a check may need to *execute* (e.g. submitted code), keyed by outbox name. |

There is deliberately **no** rationale field and no gate name/pipeline on this type.

### `SandboxPort` (Protocol)

`ports.py:137-151`

```python
class SandboxPort(Protocol):
    async def serve_context(self, context: ServedContext) -> None: ...
    async def collect_proposal(self) -> ProposalEnvelope: ...
    async def regenerate(self) -> None: ...
    async def provision(self) -> None: ...
    async def teardown(self) -> None: ...
    async def health(self) -> bool: ...
```
The ephemeral agent-runtime + workspace. `serve_context` hands the assembled context in;
`collect_proposal` takes the proposal (with its outbox objects) out; `regenerate` rebuilds the
workspace and flushes chat (one ephemerality event). The sandbox never reaches the verifier or the
store.

### `ServedContext`

`ports.py:107-119`

```python
@dataclass(frozen=True, slots=True)
class ServedContext:
    system_prompt: str       # the stable prefix (composed system prompt + manifest)
    manifest: str = ""       # reserved; serve_context currently folds the manifest into system_prompt
    tail: str = ""           # the volatile tail (retrieved artifacts + goal + scratch)
    feedback: str = ""       # last cycle's correction: a shape-error msg or a refine defect list
```
Note: `serve_context` populates `system_prompt`, `tail`, and `feedback`; it leaves `manifest` empty
because the assembler already renders the manifest into the stable prefix (`api.py:185-189`,
`context.py:155`).

### `ProposalEnvelope`

`ports.py:122-134`

```python
@dataclass(frozen=True, slots=True)
class ProposalEnvelope:
    artifact: Artifact
    operation: Operation
    metadata: str = ""                  # agent's how/why → provenance channel, kept OFF the verifier
    objects: Mapping[str, bytes] = {}   # outbox contents, harvested before teardown
```

| Field | Type | Meaning |
|---|---|---|
| `artifact` | `Artifact` | The proposed artifact (status must be `PROPOSED`). |
| `operation` | `Operation` | The provenance edge that produced it (`parents → output_id`). |
| `metadata` | `str` | The agent's rationale — stored on the segregated channel, never sent to a gate. |
| `objects` | `Mapping[str, bytes]` | Outbox objects, content-addressed at intake. |

### `VerdictBundle`

`contracts/model.py:157-170`

```python
@dataclass(frozen=True, slots=True)
class VerdictBundle:
    status: ArtifactStatus                       # the terminal status the verifier recommends
    decisions: tuple[GateDecision, ...] = ()     # the per-check rulings
    supersedes: str | None = None                # an incumbent this proposal beats, applied atomically on accept
```
The opaque verifier's reply to one dispatch. The control plane records the decisions and sets the
status **after** validating self-consistency (`commit.py:224-234`) and that the transition is legal
(`lifecycle.py:62`), but never sequences or parses.

### `GateDecision`

`contracts/model.py:142-155`

```python
@dataclass(frozen=True, slots=True)
class GateDecision:
    gate: str
    kind: VerdictKind                    # accept | reject | refine
    rationale: str
    defects: tuple[str, ...] | None = None   # localizes a refine
    score: float | None = None
```
One recorded ruling inside a bundle. Self-contained (no artifact id) so it crosses the wire cleanly;
the control plane turns each into a durable `Decision` row at commit (`commit.py:184-196`).

### `GateVerdict`

`contracts/model.py:126-140` — one *internal* check's verdict (`kind`, `rationale`, `defects`,
`score`, plus `supersedes` for a selection check). A verifier primitive returns this; the verifier
composes per-check verdicts into the `GateDecision`s of a `VerdictBundle`. Not handed across the
control-plane boundary directly.

### `ProviderRegistry[PortT]`

`ports.py:157-185`

```python
class ProviderRegistry[PortT]:
    def __init__(self, kind: str) -> None: ...
    def register(self, key: str, factory: Callable[[], PortT]) -> None: ...   # raises ProviderError on dup
    def create(self, key: str) -> PortT: ...                                  # raises ProviderError if unknown
    def keys(self) -> tuple[str, ...]: ...
```
A config-keyed registry of zero-arg port factories. Module-level singletons `SANDBOX_PROVIDERS` and
`VERIFIER_PROVIDERS` are the defaults the control plane resolves against (`ports.py:189-190`).

### Boundary value model (referenced by the ports)

From `src/verity/contracts/model.py`:

- **`Artifact`** (`model.py:86-107`): `id, type, payload, status, created_by, created_at,
  superseded_by=None, revised_by=None, is_root=False, objects=()`. Frozen; a status change produces a
  new snapshot via the commit path. `objects` is the control-plane sidecar (name → `ObjectRef`).
- **`Operation`** (`model.py:110-124`): `op_id, op_name, parents, output_id, status, created_at` — a
  typed provenance edge.
- **`ObjectRef`** (`model.py:69-79`): `blob_ref` (e.g. `"sha256:<hash>"`), `content_hash`.
- **Enums** (`model.py:37-62`): `ArtifactStatus` (`proposed/tentative/accepted/rejected/superseded/
  revised`), `OperationStatus` (`success/failed/retried`), `VerdictKind` (`accept/reject/refine`).

---

## Lifecycle state machine

`src/verity/control_plane/lifecycle.py:32-45` — the transitions the commit path validates the
verifier's recommended status against before writing:

```
proposed  -> { tentative, accepted, rejected, revised }
tentative -> { accepted, rejected, revised, superseded }
accepted  -> { superseded }
rejected | superseded | revised -> {}   (terminal, retained)
```

`accepted` is **not** terminal — it may still be superseded. An illegal edge raises
`IllegalTransition`, even on the verifier's say-so.

---

## End-to-end: one full cycle

A complete configure → serve → intake → gate → commit → extract, mirroring `tests/test_api.py`. This
uses the in-tree fake domain and minimal in-test port doubles.

```python
import asyncio
from verity.contracts import (
    Artifact, ArtifactStatus, Operation, OperationStatus,
    ProposalEnvelope, ProviderRegistry, SandboxPort, ServedContext,
    VerdictBundle, GateDecision, VerdictKind, VerifierPort, VerifierRequest,
)
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import DefaultRetrievalPolicy
from verity.control_plane.store import SqliteStore
from verity.domains.fake import NOTE, SOURCE, build_fake_domain


class StubVerifier:                       # a VerifierPort
    identity = "fake-verifier"            # MUST differ from the proposer's created_by
    async def dispatch(self, request: VerifierRequest) -> VerdictBundle:
        return VerdictBundle(
            status=ArtifactStatus.ACCEPTED,
            decisions=(
                GateDecision(gate="well-formed", kind=VerdictKind.ACCEPT, rationale="ok"),
                GateDecision(gate="worth-keeping", kind=VerdictKind.ACCEPT, rationale="ok"),
            ),
        )
    async def provision(self): ...
    async def teardown(self): ...
    async def health(self): return True


class StubSandbox:                        # a SandboxPort
    def __init__(self, proposals): self.queue = list(proposals); self.regenerated = 0
    async def serve_context(self, context: ServedContext): ...
    async def collect_proposal(self) -> ProposalEnvelope: return self.queue.pop(0)
    async def regenerate(self): self.regenerated += 1
    async def provision(self): ...
    async def teardown(self): ...
    async def health(self): return True


async def main() -> None:
    # 0. store seeded with a root Source the Note's `author` op consumes
    store = SqliteStore()
    store.propose(
        Artifact("src", SOURCE, {"raw": 1}, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation("op-src", "load", (), "src", OperationStatus.SUCCESS, "t0"),
    )

    note = ProposalEnvelope(
        artifact=Artifact("n1", NOTE, {"text": "hi"}, ArtifactStatus.PROPOSED, "agent", ""),
        operation=Operation("op-n1", "author", ("src",), "n1", OperationStatus.SUCCESS, ""),
        metadata="derived from src",
    )

    sandbox_providers: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    verifier_providers: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sandbox_providers.register("stub", lambda: StubSandbox([note]))
    verifier_providers.register("stub", lambda: StubVerifier())

    cp = ControlPlane(
        store,
        policy=OrchestrationPolicy(max_cycles=5, stop_on_accept=True),
        sandbox_providers=sandbox_providers,
        verifier_providers=verifier_providers,
    )

    domain = build_fake_domain()
    config = TaskConfig(
        task_id="t1", instructions="write notes",
        domain_instructions="a Note has a text field",
        schema=domain.schema, gated_types=domain.gated_types,
        retrieval=DefaultRetrievalPolicy(), shape_validator=domain.shape_validator,
        sandbox_key="stub", verifier_key="stub",
    )

    # 1. configure -> stamps schema v1, provisions services
    await cp.configure(config)

    # 2. serve context (optional when using run/run_cycle, which call it)
    served = await cp.serve_context("t1", goal="make a note")

    # 3-5. intake -> gate -> commit (a single cycle); harvest happens inside, then regenerate
    results = await cp.run("t1", goal="make a note")
    assert results[0].commit.outcome.value == "accepted"

    # 6. extract
    accepted = cp.accepted_artifacts(type=NOTE)        # [Artifact(id="n1", status=accepted, ...)]
    prov = cp.provenance("n1")                          # ancestors include "src"; decisions include accepts
    why = cp.rationale_for("n1")                        # "derived from src" (never a gate input)
    assert accepted[0].status is ArtifactStatus.ACCEPTED


asyncio.run(main())
```

---

## Notable / ambiguous findings

- **Class name (#16 — closed, no change needed).** The public class is `ControlPlane`; no
  `ControlPlaneApi` symbol exists anywhere in the package, and the repo is consistent on
  `ControlPlane`. The `...Api` name only ever appeared in the audit's orientation prompt, not the code.
- **`refine_cap` enforced (#17 — resolved).** The cap now binds per *lineage* at commit:
  `_count_lineage_refines` (over the store's `revises` chain) → `run_commit(refine_exhausted=…)`
  converts a capped refine into a terminal `rejected`. `TaskState.refine_counts` was removed in
  favour of the store-derived count.
- **`ServedContext.manifest` removed (#18 — resolved).** The manifest is delivered folded into
  `system_prompt` (the stable prefix); the dead field was removed and the docstring updated.
- **`teardown` does not deregister the task.** It tears down the services but leaves the `_tasks`
  entry and the stamped schema version in place; a subsequent call on that `task_id` would still find
  the (now torn-down) services.
- **Independence/rationale segregation is structural, not conventional.** `store_slice` is a tuple of
  `Artifact` (no rationale field) and `resolve_declared_slice` asserts the resolved slice never
  exceeds the declared allowlist (`independence.py:84-91`) — so "no proposer rationale, no undeclared
  state reaches a gate" holds at the type level.
- **Thread-crossing commit.** The sync §7 commit path runs in a worker thread (`asyncio.to_thread`)
  while the async verifier is awaited via `run_coroutine_threadsafe`; the `SqliteStore` connection is
  opened `check_same_thread=False` precisely because the control plane is the sole serialized mutator
  (`store.py:274-278`).
