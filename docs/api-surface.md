# Verity API Surface

Reference for Verity's **in-process async Python API** — there is no HTTP layer on the control plane.
The "API" is the control-plane service class plus the typed **port contracts** that cross the service
boundaries to the sandbox and verifier. Everything here is checkable against source; locations are
cited as `path:line`.

The public class is **`ControlPlane`** (`src/verity/control_plane/api.py:169`). It is the **only**
service that mutates durable state (spec §3.4). It is async by design: external calls are coroutines
so today's in-process services and tomorrow's networked ones share one shape. The proven §7 commit
path stays synchronous and runs in a worker thread while the API awaits the advisory verifier
(`api.py:1-22`, `api.py:409-453`).

The loop is **read → propose → gate → commit**.

> **Genericity.** A `ControlPlane` is constructed knowing **no task**. Its provider registries default
> to fresh, empty `ProviderRegistry` instances, and a task is applied *through the CP's API* —
> `register_sandbox` / `register_verifier` + `configure` — never baked into the class. This is enforced
> mechanically: an AST guard asserts `verity.control_plane` imports nothing from `verity.domains`
> (`tests/test_control_plane_genericity.py`), and one generic CP runs both the FE and `code` tasks. The
> `verity.composition` layer is the one place allowed to know both a domain and the control plane; see
> the **worked example** at the end.

---

## Endpoint index

`ControlPlane` (`src/verity/control_plane/api.py:169`):

| Method | Async? | One-line purpose | Mutates durable state? |
|---|---|---|---|
| `__init__` | sync | Construct the service over a store, assembler, policy, (empty) registries, clocks | No (in-memory wiring) |
| `store` (property) | sync | The durable provenance store — exposed so a task builder can seed its root inputs | No (accessor) |
| `register_sandbox(key, factory)` | sync | Register a task's sandbox provider under `key` | No (registry wiring) |
| `register_verifier(key, factory)` | sync | Register a task's verifier provider under `key` | No (registry wiring) |
| `configure(config)` | **async** | Register a task: stamp schema + contract versions, provision + bind sandbox/verifier | Yes (schema/contract rows) |
| `teardown(task_id)` | **async** | Tear down a task's sandbox and verifier | No (service lifecycle only) |
| `serve_context(task_id, *, goal, scratch, feedback)` | **async** | Assemble §9 bounded context and hand it to the sandbox | No (pure read) |
| `submit_proposal(task_id, envelope)` | **async** | Shape-check → harvest objects → propose → run commit path | Yes (object store, rows, status) |
| `run_cycle(task_id, *, goal, feedback)` | **async** | One read→propose→gate→commit cycle, then regenerate sandbox | Yes (via `submit_proposal`) |
| `run(task_id, *, goal)` | **async** | Drive cycles until the orchestration policy stops; emit a `RunRecord` | Yes (via `run_cycle`) |
| `run_report(task_id)` | sync | Store-derived, score-agnostic JSON projection of the run so far | No (read) |
| `run_records()` | sync | The run-results read side (operational metadata for finished runs) | No (read) |
| `accepted_artifacts(*, type=None)` | sync | Extract accepted artifacts | No (read) |
| `provenance(artifact_id)` | sync | Full lineage + decisions behind an artifact | No (read) |
| `rejected_log(*, type=None)` | sync | The rejected-artifact log | No (read) |
| `superseded_log(*, type=None)` | sync | The superseded-artifact log | No (read) |
| `revised_log(*, type=None)` | sync | The revised-artifact log | No (read) |
| `rationale_for(artifact_id)` | sync | The agent's how/why summary (never a gate input) | No (read of in-memory channel) |

Private helpers (`_task`, `_run_commit`, `_harvest_children`, `_count_lineage_refines`,
`_link_revision_if_any`, `_record_cycle`, `_record_run`) are not part of the public surface and are
documented inline only where they explain an endpoint's downstream effect.

---

## Construction & lifecycle

### `ControlPlane.__init__`

`src/verity/control_plane/api.py:172-208`

```python
def __init__(
    self,
    store: SqliteStore,
    *,
    assembler: ContextAssembler | None = None,
    policy: OrchestrationPolicy | None = None,
    sandbox_providers: ProviderRegistry[SandboxPort] | None = None,
    verifier_providers: ProviderRegistry[VerifierPort] | None = None,
    clock: Clock = default_clock,
    timer: Callable[[], float] = time.monotonic,
    dispatch_timeout_s: float | None = None,
    run_id_source: Callable[[], str] | None = None,
    run_records: RunRecordStore | None = None,
) -> None: ...
```

| Param | Type | Meaning |
|---|---|---|
| `store` | `SqliteStore` | The sole durable store (`store.py`). Implements both `Store` (reads/propose/objects) and `CommitSink` (privileged writes). In-memory by default; pass `path=...`/`object_dir=Path(...)` to persist. |
| `assembler` | `ContextAssembler \| None` | §9 context assembler; defaults to `ContextAssembler()` with the bounded-context caps. |
| `policy` | `OrchestrationPolicy \| None` | The run-again/stop rule; defaults to `OrchestrationPolicy()`. |
| `sandbox_providers` | `ProviderRegistry[SandboxPort] \| None` | Registry resolving `config.sandbox_key` → a `SandboxPort`. **Defaults to a fresh empty `ProviderRegistry("sandbox")`** — a task registers its provider via `register_sandbox`. |
| `verifier_providers` | `ProviderRegistry[VerifierPort] \| None` | Registry resolving `config.verifier_key` → a `VerifierPort`. **Defaults to a fresh empty `ProviderRegistry("verifier")`.** |
| `clock` | `Clock` | Injected ISO-8601 timestamp source for determinism. |
| `timer` | `Callable[[], float]` | Monotonic seconds source, injected so `RunReport` timings are testable (default `time.monotonic`). |
| `dispatch_timeout_s` | `float \| None` | Backstop on the one opaque verifier handoff: a hang past this becomes a recoverable `GateUnavailable`, not a blocked cycle. `None` = no backstop. |
| `run_id_source` | `Callable[[], str] \| None` | Source of a run's id, minted per `run()` (task ≠ run). Defaults to a fresh `uuid4().hex`; tests inject a deterministic source. |
| `run_records` | `RunRecordStore \| None` | Where the per-run `RunRecord` is emitted at run end. Defaults to `InMemoryRunRecordStore()`; the standing job server swaps the adapter (7.4.h / ADR 0003). |

**Effect**: pure in-memory wiring; holds an empty `_tasks: dict[str, TaskState]`. No store writes.

**Key change from earlier builds.** The registries are **empty by default** — there are no
module-level `SANDBOX_PROVIDERS`/`VERIFIER_PROVIDERS` defaults on the control plane any more. Register
each task's providers through `register_sandbox`/`register_verifier` (or hand pre-populated registries
to `__init__`) **before** you `configure` a task that names those keys, or `configure` raises
`ProviderError`.

```python
import asyncio
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.store import SqliteStore

store = SqliteStore()  # in-memory; pass path="..."/object_dir=Path(...) to persist

cp = ControlPlane(store, policy=OrchestrationPolicy(max_cycles=5, stop_on_accept=True))
cp.register_sandbox("stub", lambda: MySandbox())   # zero-arg factory → SandboxPort
cp.register_verifier("stub", lambda: MyVerifier())  # zero-arg factory → VerifierPort
```

### `store`, `register_sandbox`, `register_verifier`

`src/verity/control_plane/api.py:212-225`

```python
@property
def store(self) -> SqliteStore: ...
def register_sandbox(self, key: str, factory: Callable[[], SandboxPort]) -> None: ...
def register_verifier(self, key: str, factory: Callable[[], VerifierPort]) -> None: ...
```

The **task-registration surface** that keeps the control plane task-agnostic. `store` exposes the
durable provenance store so a task builder can seed its **root inputs** (e.g. a dataset version) at
setup; the loop's own mutations still go only through the commit path. `register_sandbox`/
`register_verifier` add a task's providers under the keys its `TaskConfig` names (`sandbox_key` /
`verifier_key`). A task is *applied to* a generic control plane through these, never compiled into it.

### `OrchestrationPolicy`

`src/verity/control_plane/api.py:134-155`

```python
@dataclass(frozen=True, slots=True)
class OrchestrationPolicy:
    max_cycles: int = 20                       # bounds the run
    refine_cap: int = 3                        # max refines on one lineage before terminal reject
    stop_on_accept: bool = False               # end the run as soon as an artifact is accepted
    max_consecutive_sandbox_failures: int = 3  # abort after this many failed cycles in a row
    def should_continue(self, *, cycles_run: int) -> bool: ...  # cycles_run < max_cycles
```

- `refine_cap` binds **per lineage at commit**: `_count_lineage_refines` walks the store's `revises`
  chain and passes `refine_exhausted` into `run_commit`, which converts a capped refine into a
  terminal `rejected` — so an un-satisfiable artifact cannot loop forever.
- `max_consecutive_sandbox_failures` is the **degrade-don't-crash breaker** (ROADMAP 5.1): a cycle
  where the sandbox could not propose **or** the gate was unavailable is a *recorded, fed-back* failed
  cycle, not a fatal error; but this many in a row aborts the run with `OrchestrationError`, so a
  wholly-broken sandbox or verifier cannot spin to `max_cycles`. The counter resets on any cycle that
  proposes and is evaluated.

### `TaskState`

`src/verity/control_plane/api.py:157-167` — internal per-task record (not constructed by callers):
holds the `config`, the resolved `sandbox` and `verifier`, the segregated `rationale` channel
(`dict[artifact_id, str]`), the per-cycle `history` (the `RunReport`'s input), and the mutable
`run_context` (the `RunContext` stamped onto the work each cycle provisions).

---

## Configure-by-task

### `configure`

`src/verity/control_plane/api.py:229-260`

```python
async def configure(self, config: TaskConfig) -> None: ...
```

**Does**: registers a task. In order (`api.py:231-260`):
1. Computes the next schema version from `store.current_schema_version()` (`current.version + 1`,
   else `1`) and writes a `schema_versions` row via `store.register_schema_version(config.schema.snapshot(...))`.
2. Stamps the **workspace-contract / orientation version** too (#12, §3.4): a `ContractVersion`
   (with an `orientation_digest`) recorded like the schema, not left ambient. Idempotent across tasks.
3. Resolves the sandbox via `sandbox_providers.create(config.sandbox_key)` and the verifier via
   `verifier_providers.create(config.verifier_key)`.
4. `await`s `sandbox.provision()` then `verifier.provision()` (the §3.4 lifecycle surface).
5. Builds a `TaskState` with a fresh `RunContext(tenant_id=config.tenant_id, run_id=...)` and **binds
   it** to any service that implements `SupportsRunContext` (`service.bind_run_context(...)`) — so a
   backend-backed sandbox/verifier labels every worker it launches; an in-process stub ignores the
   bind. Bound once: the cell is mutable, so advancing `cycle` / re-minting `run_id` shows through
   without re-binding.
6. Stores the `TaskState` under `config.task_id` and logs `task_configured`.

**Downstream effect**: one `schema_versions` row + one (idempotent) `contract_versions` row; two
services provisioned and bound; an entry in the in-memory `_tasks` map. Re-configuring stamps a *new*
schema version (the counter only increments).

**Errors**: `ProviderError` if `sandbox_key`/`verifier_key` are unregistered.

**Precondition**: the named providers must be registered (via `register_sandbox`/`register_verifier`
or the registries passed to `__init__`) before this call.

### `teardown`

`src/verity/control_plane/api.py:262-265`

```python
async def teardown(self, task_id: str) -> None: ...
```

**Does**: `await`s `task.sandbox.teardown()` then `task.verifier.teardown()` for the named task. Does
**not** remove the task from `_tasks`, mutate the store, or release the schema version.

**Errors**: `UnknownTask` if `task_id` was never configured.

---

## Cycle control

### `serve_context`

`src/verity/control_plane/api.py:274-294`

```python
async def serve_context(
    self, task_id: str, *, goal: str, scratch: str = "", feedback: str = ""
) -> ServedContext: ...
```

**Does**: assembles the §9 bounded context from the store and serves it to the sandbox.
1. `ContextAssembler.assemble(store, system_prompt=config.system_prompt(), goal=..., scratch=...,
   retrieval=config.retrieval)` — regenerates a stable prefix (composed 3-layer system prompt + ranked
   manifest) and a volatile tail (the task retrieval policy's selection + goal + scratch), capped by
   the assembler's bounded-context limits.
2. Wraps it in a `ServedContext`, including `workspace_objects =
   config.object_provisioning.materialize(store)` — durable refs the agent builds on, selected by
   status/recency only (§9, no verifier knowledge), re-materialized every cycle into a writable role.
3. `await`s `task.sandbox.serve_context(served)` and returns it.

**Downstream effect**: read-only against the store; pushes context into the sandbox runtime. The
task's own `retrieval` policy (not the assembler default) drives the tail.

**Errors**: `UnknownTask` for an unconfigured task.

### `submit_proposal`

`src/verity/control_plane/api.py:322-366`

```python
async def submit_proposal(self, task_id: str, envelope: ProposalEnvelope) -> IntakeResult: ...
```

The **intake "endpoint"** — the single path that turns a sandbox proposal into durable provenance.
Step by step:

1. **Shape check (§7.0, pre-gate).** `_check_shape` runs three kernel-first layers (`api.py:298-320`):
   (a) the payload must be a **JSON object**; (b) any **required payload keys** the operation declares
   must be present; (c) the domain's own `config.shape_validator`. If any returns a `ShapeError`, the
   method logs `intake_shape_error` and returns `IntakeResult(entered_protocol=False, shape_error=...)`
   — **nothing is recorded**: no object harvested, no `proposed` row, no decision, verifier never
   dispatched.
2. **Harvest before teardown.** `harvested = {name: store.put_object(data) for name, data in
   envelope.objects.items()}` — content-addresses each outbox object into the object store. This
   ordering (harvest before the sandbox regenerates) is the hard constraint.
3. **Object sidecar.** If anything was harvested, replaces the artifact with
   `replace(artifact, objects=tuple(harvested.items()))` — a control-plane-owned sidecar; the domain
   `payload` is stored verbatim and never reached into (§4.1, ADR 0001).
4. **Rationale segregation.** If `envelope.metadata` is non-empty, stores it on
   `task.rationale[artifact.id]` — the provenance/context channel, **never** sent to a gate (§10).
5. **Propose.** `store.propose(artifact, envelope.operation)` writes a `proposed` artifact row + its
   producing `operations` row; `_link_revision_if_any` sets `revised_by` when this is a `revises` of a
   currently-`REVISED` parent.
6. **Commit.** `await self._run_commit(...)` runs the synchronous §7 commit path in a worker thread,
   bridging the one opaque verifier handoff (see below). On acceptance, if the task declares a
   `harvester`, `_harvest_children` mints the domain's child artifacts (e.g. one `Feature` per declared
   feature) and gates each through its own declared gate. Logs `proposal_committed` and returns
   `IntakeResult(entered_protocol=True, commit=..., harvested=...)`.

**The verifier handoff** (`_run_commit`, `api.py:409-453`): a `dispatch` closure resolves the declared
store-inputs for the artifact's type (`config.gated_types.resolve(type)`), cuts the rationale-free
`store_slice` via `resolve_declared_slice`, builds a `VerifierRequest(proposal, store_slice, objects)`,
and calls `verifier.dispatch(...)` across the thread boundary with `asyncio.run_coroutine_threadsafe`.
If `dispatch_timeout_s` elapses, the future is cancelled and a `GateUnavailable` is raised (a
recoverable failed cycle, not an aborting timeout). `run_commit` records every decision, validates the
bundle is self-consistent and the transition is legal, and sets the status. **One handoff per
proposal — not one per gate.**

**Downstream durable effects**, by verdict bundle status:

| Bundle status | Store effect | `CommitResult.outcome` |
|---|---|---|
| `REJECTED` | `set_status → rejected` | `REJECTED` |
| `REVISED` | `set_status → revised`; `CommitResult.defects` carries the localized defect list | `REVISED` |
| `TENTATIVE` | provenance ensured, `set_status → tentative` | `TENTATIVE` |
| `ACCEPTED` | provenance ensured, `set_status → accepted`; if `bundle.supersedes` is set, the incumbent flips to `superseded` atomically | `ACCEPTED` |

In all non-shape-error cases a `decisions` row is written for every `GateDecision` in the bundle.

**Errors**: `UnknownTask`; and from the commit path: `NoImplicitAccept` (type not gated, §5.7),
`ProposerIsGate` (verifier identity == `artifact.created_by`, §7.2), `BundleInconsistent`,
`CommitError`, `IllegalTransition`, `IndependenceViolation`, `StoreError`. A `GateUnavailable` is
raised when the verifier could not render a verdict — caught by `run_cycle` as a recoverable failed
cycle (it does **not** propagate out of `run`).

### `IntakeResult`

`src/verity/control_plane/api.py:113-132`

```python
@dataclass(frozen=True, slots=True)
class IntakeResult:
    entered_protocol: bool                    # False on malformed OR sandbox-fail OR gate-unavailable
    shape_error: ShapeError | None = None     # the correction, when malformed
    commit: CommitResult | None = None        # the §7 result, when it entered the protocol
    harvested: dict[str, ObjectRef] = {}      # outbox name → content-addressed ref
    sandbox_error: str | None = None          # the reason, when the sandbox produced no usable proposal
    gate_error: str | None = None             # the reason, when the gate could not render a verdict
```

`entered_protocol` is `False` exactly when the proposal was malformed, the sandbox produced no usable
proposal this cycle, **or** the gate was unavailable. For shape/sandbox failures **nothing is
recorded**; for a gate failure the proposal *was* recorded (`proposed`) but no verdict could be
rendered.

### `run_cycle`

`src/verity/control_plane/api.py:491-541`

```python
async def run_cycle(self, task_id: str, *, goal: str, feedback: str = "") -> IntakeResult: ...
```

**Does**: one complete read→propose→gate→commit cycle:
1. Advances `task.run_context.cycle` so this cycle's workers are labelled.
2. `await serve_context(...)` (read).
3. `envelope = await task.sandbox.collect_proposal()` — a `SandboxError` here is caught: the broken
   workspace is regenerated, the failure is recorded as a failed cycle (`sandbox_error`), and the run
   continues (a single bad cycle cannot destroy a multi-round session, §3.4 / #21).
4. `result = await submit_proposal(...)` (gate + commit) — a `GateUnavailable` is caught symmetrically
   as a `gate_error` failed cycle.
5. `await task.sandbox.regenerate()` — discards the ephemeral workspace (one ephemerality event, §3.5).
6. `_record_cycle` appends this cycle's facts (timings + agent telemetry) to the task `history`.

**Returns**: the `IntakeResult` (carrying `commit`, or a `sandbox_error`/`gate_error`).

### `run`

`src/verity/control_plane/api.py:559-605`

```python
async def run(self, task_id: str, *, goal: str) -> list[IntakeResult]: ...
```

**Does**: drives `run_cycle` while `policy.should_continue(cycles_run=...)`. A run is one execution of
a task (task ≠ run): it **mints a fresh `run_id`** and resets `cycle = 0` up front. After each cycle it
feeds the cycle's correction back into the next (`_feedback_from`, `api.py:672-697` — a shape-error, a
refine defect list, a sandbox-failure reason, a gate-unavailable note, **or** a rejection reason). It
counts consecutive failed cycles and aborts with `OrchestrationError` at
`max_consecutive_sandbox_failures`; with `stop_on_accept` it breaks on the first accept. At the end it
emits a `RunRecord` (via `_record_run`) and logs `run_complete`.

**Returns**: the list of per-cycle `IntakeResult`s.

**Errors**: `OrchestrationError` on too many consecutive failed cycles (a `RunRecord` with
`status="aborted"` is still recorded first).

---

## Run metadata & reporting

### `run_report`

`src/verity/control_plane/api.py:628-646`

```python
def run_report(self, task_id: str) -> RunReport: ...
```

A generic, machine-readable projection of the task's run so far (ROADMAP 5.3a), built from the
recorded per-cycle facts + the store's lifecycle queries. It assumes nothing about the verifier or
domain (decisions are projected verbatim, `score` nullable). `RunReport`
(`control_plane/run_report.py:190`) carries `report_schema_version`, `task_id`, `tenant_id`,
`generated_at`, `policy`, `cycles: list[CycleReport]`, a `summary: RunSummary` (with `cycles_run`,
per-outcome counts, and the `accepted` ids), and run-total `agent_telemetry` (tokens / steps /
tool-calls, or `None`). `RunReport.to_dict()` is the JSON form. The control plane **emits** it; parsing
it is the receiving service's job (the `verity.eval` benchmark and `verity.telemetry` sink both consume
it).

### `run_records`

`src/verity/control_plane/api.py:622-624`

```python
def run_records(self) -> RunRecordStore: ...
```

The run-results read side — operational metadata for finished runs (7.4.h). Each `run()` persists a
`RunRecord` (`control_plane/run_record.py:24`) keyed by `(tenant_id, run_id)`:

```python
@dataclass(frozen=True, slots=True)
class RunRecord:
    tenant_id: str
    task_id: str                        # which task (configuration) this run executed
    run_id: str                         # the run's own identity, distinct from the task
    status: str                         # "complete" / "aborted" / …
    report: RunReport
    accepted_artifact_ids: tuple[str, ...] = ()   # pointers INTO the store, not a copy
```

`RunRecordStore` is a `Protocol` (`put` / `get(tenant_id, run_id)` / `list(tenant_id, *, task_id=None)`)
with an `InMemoryRunRecordStore` default; the standing job server swaps the adapter. `RunRecord` points
into the provenance store (the system of record) — it never copies artifacts.

```python
results = await cp.run("fe", goal=FE_GOAL)
record = cp.run_records().list("default", task_id="fe")[-1]
assert record.status == "complete"
assert record.accepted_artifact_ids            # ids you can resolve via cp.provenance(...)
```

---

## Extraction (read-only)

All synchronous; all pure reads.

### `accepted_artifacts`

`src/verity/control_plane/api.py:648-649`

```python
def accepted_artifacts(self, *, type: str | None = None) -> list[Artifact]: ...
```
Returns `store.query_artifacts(type=type, status=ACCEPTED)` — the accepted artifacts, optionally
filtered by type, ordered by `(created_at, id)`.

### `provenance`

`src/verity/control_plane/api.py:651-653`

```python
def provenance(self, artifact_id: str) -> Provenance: ...
```
Returns the full transitive lineage + gathered decisions behind an artifact — the "why do we believe
X" answer. `Provenance` carries `artifact`, `operations`, `ancestors`, and `decisions`.

**Errors**: `StoreError` if the artifact is unknown.

### `rejected_log` / `superseded_log` / `revised_log`

`src/verity/control_plane/api.py:655-662`

```python
def rejected_log(self,   *, type: str | None = None) -> list[Artifact]: ...
def superseded_log(self, *, type: str | None = None) -> list[Artifact]: ...
def revised_log(self,    *, type: str | None = None) -> list[Artifact]: ...
```
The terminal/lineage logs: `rejected`, `superseded`, and `revised` artifacts respectively. Optional
`type` filter.

### `rationale_for`

`src/verity/control_plane/api.py:664-669`

```python
def rationale_for(self, artifact_id: str) -> str | None: ...
```
Returns the agent's how/why summary captured at intake from `envelope.metadata`, by scanning every
task's segregated `rationale` channel. Returns `None` if none was stored. This is **never** a gate
input (it lives off the `VerifierRequest` by construction).

---

## Errors

`src/verity/control_plane/api.py:101-110`

```python
class ControlPlaneError(RuntimeError):  # a misuse of the control-plane API
class UnknownTask(ControlPlaneError):   # operation referenced an unconfigured task
class OrchestrationError(ControlPlaneError):  # run aborted on too many consecutive failed cycles
```

Commit-path errors surface through `submit_proposal` (`commit.py`): `CommitError`, `NoImplicitAccept`,
`ProposerIsGate`, `BundleInconsistent`; lifecycle: `IllegalTransition`; independence:
`IndependenceViolation`; provider resolution: `ProviderError` (`contracts/ports.py:47`); store
invariants: `StoreError`; transient gate infrastructure: `GateUnavailable` (`contracts/errors.py`,
caught inside the loop).

---

## Port contracts (the cross-service surface)

The control plane talks to the sandbox and verifier **only** through these ports
(`src/verity/contracts/ports.py`), and selects implementations by task config through a
`ProviderRegistry`. Every port call is `async` so a network hop is not a refactor. All ports also
expose the no-op-able lifecycle surface (`provision` / `teardown` / `health`).

### `ServiceLifecycle` (Protocol)

`contracts/ports.py:55-64`

```python
async def provision(self) -> None: ...
async def teardown(self) -> None: ...
async def health(self) -> bool: ...
```
The optional lifecycle the control plane may drive. In-process implementations satisfy these as no-ops.

### `VerifierPort` (Protocol)

`contracts/ports.py:87-101`

```python
class VerifierPort(Protocol):
    identity: str
    async def dispatch(self, request: VerifierRequest) -> VerdictBundle: ...
    async def provision(self) -> None: ...
    async def teardown(self) -> None: ...
    async def health(self) -> bool: ...
```
The advisory, **opaque** verifier. `dispatch` is the only path to a verdict — one proposal in, one
`VerdictBundle` out; the verifier owns which checks run and in what order. `identity` must differ from
the proposer's (`created_by`), enforced at commit (Builder/Breaker, §7.2). The sandbox has no path to
it.

### `VerifierRequest`

`contracts/ports.py:70-83`

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
| `store_slice` | `tuple[Artifact, ...]` | The declared, control-plane-cut slice (incumbents, rejected-log) for this type. `Artifact` has **no rationale field**, so the proposer's reasoning cannot ride along (§10). |
| `objects` | `Mapping[str, bytes]` | Harvested attachments a check may need to *execute* (e.g. submitted code), keyed by outbox name. |

There is deliberately **no** rationale field and no gate name/pipeline on this type.

### `SandboxPort` (Protocol)

`contracts/ports.py:150-163`

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

`contracts/ports.py:107-128`

```python
@dataclass(frozen=True, slots=True)
class ServedContext:
    system_prompt: str       # the stable prefix (composed system prompt + manifest already folded in)
    tail: str = ""           # the volatile tail (retrieved artifacts + goal + scratch)
    feedback: str = ""       # last cycle's correction: a shape-error msg or a refine defect list
    workspace_objects: Mapping[str, Mapping[str, bytes]] = {}  # durable objects → writable role this cycle
```
`serve_context` populates `system_prompt`, `tail`, `feedback`, and `workspace_objects`. The manifest is
folded into `system_prompt` (the cache-friendly stable prefix), not a separate field.

### `ProposalEnvelope`

`contracts/ports.py:131-148`

```python
@dataclass(frozen=True, slots=True)
class ProposalEnvelope:
    artifact: Artifact
    operation: Operation
    metadata: str = ""                              # agent's how/why → provenance channel, OFF the verifier
    objects: Mapping[str, bytes] = {}               # outbox contents, harvested before teardown
    agent_telemetry: Mapping[str, object] | None = None  # tokens / steps / model (5.3b), or None
```

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
status **after** validating self-consistency and that the transition is legal, but never sequences or
parses.

### `GateDecision`

`contracts/model.py:143-155`

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
the control plane turns each into a durable `Decision` row at commit.

### `GateVerdict`

`contracts/model.py:127-140` — one *internal* check's verdict (`kind`, `rationale`, `defects`, `score`,
plus `supersedes` for a selection check). A verifier primitive returns this; the verifier composes
per-check verdicts into the `GateDecision`s of a `VerdictBundle`. Not handed across the control-plane
boundary directly.

### `RunContext` / `SupportsRunContext`

`contracts/run_context.py:26-49`

```python
@dataclass(slots=True)
class RunContext:                       # mutable on purpose
    tenant_id: str = "default"
    run_id: str = ""
    cycle: int = 0

@runtime_checkable
class SupportsRunContext(Protocol):
    def bind_run_context(self, ctx: RunContext) -> None: ...
```
The neutral `(tenant_id, run_id, cycle)` identity of the work a cycle provisions (ROADMAP 7.4.h, ADR
0003). The control plane binds one mutable cell per task at `configure`, then advances `cycle` per
cycle and sets `run_id` per `run()` in place; a bound service (a backend-backed sandbox/verifier) reads
the current values lazily and stamps them onto every worker's `{harness, tenant, run, cycle, role,
config}` labels — the seam the `verity-reaper` and run-control machinery key off. An in-process service
that doesn't implement the protocol simply isn't bound.

### `ProviderRegistry[PortT]`

`contracts/ports.py:169-196`

```python
class ProviderRegistry[PortT]:
    def __init__(self, kind: str) -> None: ...
    def register(self, key: str, factory: Callable[[], PortT]) -> None: ...   # raises ProviderError on dup
    def create(self, key: str) -> PortT: ...                                  # raises ProviderError if unknown
    def keys(self) -> tuple[str, ...]: ...
```
A config-keyed registry of zero-arg port factories. Module-level singletons `SANDBOX_PROVIDERS` /
`VERIFIER_PROVIDERS` still exist in `contracts.ports` (`ports.py:201-202`) for callers that want a
shared registry, but the **control plane no longer defaults to them** — its registries are empty unless
you register through `register_sandbox`/`register_verifier` or pass populated ones to `__init__`.

### Boundary value model (referenced by the ports)

From `src/verity/contracts/model.py`:

- **`Artifact`** (`model.py:87`): `id, type, payload, status, created_by, created_at,
  superseded_by=None, revised_by=None, is_root=False, objects=()`. Frozen; a status change produces a
  new snapshot via the commit path. `objects` is the control-plane sidecar (name → `ObjectRef`).
- **`Operation`** (`model.py:111`): `op_id, op_name, parents, output_id, status, created_at` — a typed
  provenance edge.
- **`ObjectRef`** (`model.py:70`): `blob_ref` (e.g. `"sha256:<hash>"`), `content_hash`.
- **Enums** (`model.py:37-62`): `ArtifactStatus` (`proposed/tentative/accepted/rejected/superseded/
  revised`), `OperationStatus` (`success/failed/retried`), `VerdictKind` (`accept/reject/refine`).

---

## Lifecycle state machine

`src/verity/control_plane/lifecycle.py` — the transitions the commit path validates the verifier's
recommended status against before writing:

```
proposed  -> { tentative, accepted, rejected, revised }
tentative -> { accepted, rejected, revised, superseded }
accepted  -> { superseded }
rejected | superseded | revised -> {}   (terminal, retained)
```

`accepted` is **not** terminal — it may still be superseded. An illegal edge raises
`IllegalTransition`, even on the verifier's say-so.

---

## Worked example — the feature-engineering task

The §12 feature-engineering domain is the MVP validation target and the canonical end-to-end use of
the API. It shows the **genericity contract** in action: a `ControlPlane` is built knowing no task, and
the FE task is *applied to it* through `configure_fe_task` — the one composition-layer function allowed
to know both FE and the control plane (`src/verity/composition/fe.py`).

### A. Programmatic — apply FE to a generic control plane, then run it

`configure_fe_task` (`composition/fe.py:76`) registers the FE sandbox + verifier providers via the
CP's API, seeds the dataset root into `cp.store`, and calls `cp.configure(...)`. The sandbox runs as a
**worker** (`BackendSandboxDriver`); the FE verifier's gate logic stays **trusted and in-process** and
executes the untrusted submission as a `BackendCodeRunner` worker — so the answer key
(`reserved_labels`) never enters any worker. The worker substrate (Docker now) is chosen by the
`WorkerBackend` you pass; `ProvisioningConfig` carries the substrate shape, kept **off** `TaskConfig`.

```python
import asyncio
from pathlib import Path

from verity.composition import FE_GOAL, FE_TASK_ID, ProvisioningConfig, configure_fe_task
from verity.composition.dataset import stratified_split, subsample
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.store import SqliteStore
from verity.domains.feature_engineering import SUBMISSION
from verity.provisioning import DockerBackend
from verity.sandbox.model_spec import ModelSpec


async def main() -> None:
    # 1. Split a labelled CSV into the agent's training data and a reserved hold-out it never sees.
    raw = subsample(Path("feature-engineering-test/train.csv").read_bytes(), per_class=300)
    split = stratified_split(raw, target="class", id_column="id", reserved_fraction=0.5)

    # 2. A GENERIC control plane — it knows no task at construction.
    cp = ControlPlane(SqliteStore(), policy=OrchestrationPolicy(max_cycles=4))

    # 3. Apply FE through the CP's API (register providers + seed dataset + configure). The
    #    DockerBackend launches sandbox + code-runner worker containers per cycle.
    await configure_fe_task(
        cp,
        backend=DockerBackend(),
        split=split,
        provisioning=ProvisioningConfig(
            model_spec=ModelSpec.from_provider_string("anthropic:claude-sonnet-4-6"),
            sandbox_image="verity-sandbox:latest",
        ),
    )

    # 4. Drive read → propose → gate → commit until the policy stops.
    await cp.run(FE_TASK_ID, goal=FE_GOAL)

    # 5. Extract: the accepted submissions, the audit report, and the run record.
    accepted = cp.accepted_artifacts(type=SUBMISSION)
    report = cp.run_report(FE_TASK_ID)
    print("accepted:", [a.id for a in accepted])
    print("cycles:", report.summary.cycles_run, "outcomes:", report.summary.outcomes)


asyncio.run(main())
```

This is the exact shape the containerized entrypoint uses (`composition/fe_run.py`) and that
`tests/test_fe_composition.py` asserts against a `FakeBackend` (offline — config-driven flow + real
gate logic, scripted execution). A frontier model reliably reaches an accepted `Submission`; a small
local model may stall at the `features-defined` refine gate (a known prompt/feedback-tuning gap, see
ROADMAP Phase 6 — *system* correctness is the deliverable, model score is separate).

### B. Containerized — one command, the generic CP in a container

The same code runs **fully containerized**: a task-agnostic control plane runs *in* a container
(`Dockerfile.controlplane`) and launches the sandbox + code-runner **sibling** worker containers on the
host daemon via a mounted `/var/run/docker.sock` (controlled socket, ADR 0003 §f — not
docker-in-docker). The entrypoint is `python -m verity.composition.fe_run` (`composition/fe_run.py`),
configured by environment:

| Env var | Default | Meaning |
|---|---|---|
| `VERITY_FE_DATASET` | `/data/train.csv` | Path to the labelled CSV inside the container. |
| `VERITY_MODEL` | `anthropic:claude-sonnet-4-6` | Provider string for the sandbox model. |
| `VERITY_LOCAL_BASE_URL` | *(unset)* | If set, `VERITY_MODEL` is a local model served at this OpenAI-compatible URL (e.g. `http://host.docker.internal:11434/v1`). |
| `VERITY_MAX_CYCLES` | `4` | Refine cycles. |
| `VERITY_PER_CLASS` | `300` | Rows per class to subsample for speed. |
| `VERITY_RESERVED_FRACTION` | `0.5` | Held-out share of the split. |
| `VERITY_SANDBOX_IMAGE` | `verity-sandbox:latest` | The agent worker image. |
| `VERITY_WORKER_STAGING` | *(unset)* | Shared host↔container staging dir, at an **identical path** on both, so worker bind-mounts resolve on the host daemon (the sibling-mount fix). |

Run it with one command (builds the sandbox + control-plane images, creates the staging dir, runs one
FE task through `infra/compose.fe.yml`):

```sh
just fe-containerized
```

The compose file (`infra/compose.fe.yml`) mounts the docker socket, bind-mounts
`/tmp/verity-staging` at a matching path, mounts the dataset read-only at `/data/train.csv`, and passes
the `VERITY_*` env through. For a local model instead of Anthropic, set `VERITY_MODEL` +
`VERITY_LOCAL_BASE_URL` (see `docs/local-models.md`). The run prints the `RunReport` JSON and the
accepted submission ids, and `verity-reaper` (the `verity.provisioning.reaper:main` console script)
clears any orphaned workers by label.

### C. The standing daemon — one container, many tasks, no rebuild (Phase 8, ADR 0004)

The same image also runs a **long-lived control-plane daemon** (`verity serve`) you configure at
runtime — define tasks, ingest data, run, and pull results without rebuilding (ADR 0004). The daemon
keeps a `DockerBackend`, the `TaskCatalog`, and a **generic `ControlPlane` + `SqliteStore` per task
instance** as process state, and serves an HTTP surface over a **Unix-domain socket** inside the
container. The `verity` console script is a thin **client** over that socket; `just cp <verb>` is
`docker exec verity-cp verity <verb>`.

Lifecycle: `just cp-serve` (build images + `docker compose -f infra/compose.daemon.yml up -d`),
`just cp <verb>`, `just cp-down`. Verbs: `catalog` (task types + published contracts), `ingest`
(exchange file → a data handle), `create` (a `TaskRequest` → a `task_id`), `run` (background → a
`run_id`), `status` / `results`, `export` (durable artifacts → the exchange out-dir).

`infra/compose.daemon.yml` adds two CP-only volumes on top of the socket + staging mounts:

| Env var | Default (container) | Meaning |
|---|---|---|
| `VERITY_SOCKET` | `/run/verity.sock` | The daemon's IPC socket; the v1 trust boundary (local / `docker exec`-only, no auth). |
| `VERITY_EXCHANGE` | `/exchange` | The client↔CP file channel (`in/` for ingest, `out/` for export). **CP-only — never a worker.** Host dir via `VERITY_EXCHANGE_HOST` (default `/tmp/verity-exchange`). |
| `VERITY_STORE_ROOT` | `/var/lib/verity` | Persistent per-task stores + task definitions (durable across restart). **CP-only.** Host dir via `VERITY_STORE_HOST` (default `/tmp/verity-store`). |

`VERITY_WORKER_STAGING`, `VERITY_SANDBOX_IMAGE`, `VERITY_MODEL`, `VERITY_LOCAL_BASE_URL`, and the
model key (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, forwarded to the agent worker via sandbox
env-passthrough) carry over from section B. The exchange + store volumes reach the control-plane
container alone — workers receive bytes **only** through the CP provisioning path, asserted by
`tests/test_daemon_volume_isolation.py` (a `WorkerSpec` has no host-bind-mount field; neither host
path nor the answer key reaches a worker). The end-to-end live proof — two task types and a restart,
no rebuild — is `tests/test_daemon_live.py`.

> **Entrypoint note.** The image's default `ENTRYPOINT` stays the one-shot `fe_run` (so
> `compose.fe.yml` is unchanged); `compose.daemon.yml` overrides it to `verity serve`. A minor,
> deliberate deviation from ADR 0004 sprint 3's literal "ENTRYPOINT → the daemon."

### D. External HTTP/REST — the network control API (Phase 8.4, ADR 0004 (e))

The *same* `build_app` the daemon serves over a Unix socket also binds to a **network TCP port**, so a
remote client (no `docker exec`, no shared volume) can drive the control plane:

```sh
VERITY_API_TOKEN=<secret> verity serve --http 0.0.0.0:8080   # (or `python -m verity.service` + $VERITY_HTTP)
```

**Auth — bearer token (resolves ADR 0004 open Q3).** When bound to a port, **every route except
`GET /health` requires `Authorization: Bearer $VERITY_API_TOKEN`** (401 otherwise); the daemon
**refuses to start a network binding without a token**. The Unix-socket binding stays auth-free (the
local trust boundary). The authenticated principal is stashed on the request (the identity hook);
per-principal authz/tenancy is deferred (#58). mTLS / OAuth are out of scope for v1.

**Byte data plane (pulled forward from #3, bounded).** Because a network client has no shared
exchange volume:

| Route | Purpose |
|---|---|
| `POST /objects` (raw body, non-JSON) | Upload object **bytes** → a content-addressed `{handle}` (vs the JSON `{"name": …}` form that reads the server-side exchange). Capped (413 over the limit), in-memory, no streaming. |
| `GET /artifacts/{run_id}/{object_path}` | Download a single durable artifact object's **bytes** (`<artifact_id>/<name>` or a bare `<name>`). |

All other routes are the §(d) verbs (`/catalog`, `/tasks`, `/tasks/{id}/runs`, `/runs/{id}`,
`/runs/{id}/results`, `/runs/{id}/export`) — **endpoint-parity with the CLI**. The `verity` CLI is the
same client against a TCP daemon: `verity --url http://host:8080 --token <secret> catalog`
(or `$VERITY_URL` / `$VERITY_API_TOKEN`). Removing the shared exchange volume entirely and
large-file streaming remain the deferred #3 data-plane work.

---

## Notable findings

- **The control plane is provably task-agnostic.** Empty registries by default + the
  `register_sandbox`/`register_verifier`/`configure` application path + the AST import guard
  (`tests/test_control_plane_genericity.py`) mean FE-specific code lives only in
  `verity.domains.feature_engineering` + `verity.composition.fe`; `verity.control_plane` has zero
  domain references. The earlier `build_fe_control_plane` (which built a `ControlPlane` inside an
  FE-specific function) was deleted for this reason.
- **`refine_cap` enforced per lineage at commit.** `_count_lineage_refines` over the store's `revises`
  chain → `run_commit(refine_exhausted=…)` converts a capped refine into a terminal `rejected`.
- **Degrade-don't-crash is symmetric.** A sandbox that can't propose and a verifier that can't render a
  verdict are both *recorded, fed-back* failed cycles (`sandbox_error` / `gate_error`); only
  `max_consecutive_sandbox_failures` in a row aborts the run.
- **`teardown` does not deregister the task.** It tears down the services but leaves the `_tasks` entry
  and the stamped schema version in place.
- **Independence/rationale segregation is structural.** `store_slice` is a tuple of `Artifact` (no
  rationale field) and `resolve_declared_slice` asserts the resolved slice never exceeds the declared
  allowlist — so "no proposer rationale, no undeclared state reaches a gate" holds at the type level.
- **Thread-crossing commit.** The sync §7 commit path runs in a worker thread (`asyncio.to_thread`)
  while the async verifier is awaited via `run_coroutine_threadsafe`; the `SqliteStore` connection is
  opened `check_same_thread=False` precisely because the control plane is the sole serialized mutator.
