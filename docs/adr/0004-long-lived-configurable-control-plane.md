# ADR 0004 — The long-lived, configurable control-plane service (task catalog, control surface, file I/O)

- **Status:** Accepted — implemented (Phase 8, the long-lived `verity serve` daemon, v1 complete); drafted on `docs/adr-0004-configurable-control-plane`
- **Date:** 2026-06-10 (revised same day after review — resolved the CLI↔daemon transport, replaced the
  verifier compatibility matrix with registry self-description, deferred crash recovery to #57)
- **Affects spec:** §3.3–§3.4 (the control-plane service + its API surface), §3.9 (deployment &
  portability), §16 (standing-service / multi-tenancy seams). **Activates issue #3** (the standing
  control-plane data plane). Builds directly on **[ADR 0003](0003-control-plane-and-ephemeral-worker-provisioning.md)**.
- **Supersedes:** nothing. This is **purely additive** — a control/data plane *in front of* the
  existing kernel and the ADR 0003 worker model. No kernel refactor.

> **Editorial note (2026-09-19).** The decision stands and is implemented. Two things this record
> describes as *present* no longer exist: the one-shot batch entrypoint
> `composition/fe_run.py` and its `infra/compose.fe.yml` wiring were removed with the basic-`fe`
> task (#68) — the image's default `ENTRYPOINT` is `verity serve` and the daemon is the only path —
> and the per-image Dockerfiles were consolidated into one multi-stage `Dockerfile` whose
> `controlplane` target builds the daemon. Read those references as the state of the world the ADR
> argued from, not as a map of the tree.

## Context

After ROADMAP 7.5 the control plane runs *in a container* and launches ephemeral sandbox + code-runner
worker containers (ADR 0003). But that container is still a **one-shot batch job**: the entrypoint
`python -m verity.composition.fe_run` (`src/verity/composition/fe_run.py`) reads env vars at startup,
applies a single hard-coded task (`configure_fe_task`) to a fresh `ControlPlane`, runs it once, prints
the `RunReport` to stdout, and exits. Input data arrives as a read-only bind-mount baked into
`infra/compose.fe.yml`; output is stdout. To run a *different* task, or feed *different* files, you
edit the compose/env and restart — and anything genuinely new touches the image.

**The target.** A **long-lived, configurable control-plane container** that a client — concretely,
Claude Code — treats as a standing service it delegates work to. The mental model:

> Claude Code has a task to delegate to Verity, reachable through the long-lived control plane. He
> **(1)** defines a task — selecting a **sandbox configuration** and a **verifier approach** from those
> registered — **(2)** provides any data/files the task needs, **(3)** triggers a run, and **(4)** once
> it completes, pulls back the run results and durable artifacts (submitted code, a built database) —
> all **without rebuilding any Docker image**.

This ADR is developed **inside-out first** (what Verity already gives us, so we extend rather than
refactor), then **outside-in** (the smallest additive surface that delivers the target).

### Inside-out: what we already have to build on

The kernel is already the long-lived, configurable thing — we are wrapping it in a request loop, not
rebuilding it. The leverage points:

- **A generic, API-configured control plane.** `ControlPlane` is task-agnostic; a task is *applied*
  through its API — `register_sandbox(key, factory)` / `register_verifier(key, factory)` +
  `configure(TaskConfig)` + the `store` property for seeding root inputs (`control_plane/api.py:212-260`).
  An AST guard keeps `verity.control_plane` importing nothing from `verity.domains`. **This is exactly
  the seam a control surface drives.**
- **The composition layer is already a task catalog in embryo.** `configure_fe_task` /
  `configure_code_task` (`composition/fe.py:76`, `composition/code.py:39`) take a generic `cp` + a
  `WorkerBackend` + a `ProvisioningConfig` and register + configure a named task — returning its
  `task_id`. `composition` is the one layer allowed to know both a domain and the control plane.
- **The sandbox/verifier "configurations" are already declarative.** A sandbox config is a
  `ModelSpec` (`provider`/`model`/`base_url`/`api_key_env`/`extra`, `sandbox/model_spec.py:47`) +
  `ProvisioningConfig` (image, memory, runtime, timeouts; `composition/fe.py:56`), with `local_spec` /
  `openai_spec` convenience builders. A verifier approach is the per-task builder (e.g.
  `build_feature_engineering_verifier`). **Selecting these by name + knobs is the registry the target
  describes — and none of it requires an image rebuild** (prebuilt sandbox/code images are
  *parameterized*, ADR 0003).
- **A content-addressed object store + a complete extraction API.** `store.put_object(bytes) ->
  ObjectRef` / `store.get_object(content_hash) -> bytes` (`store.py:162-194`); `SqliteStore(path=...,
  object_dir=...)` persists the DB and the blobs to disk when pointed at a path/dir (`store.py:282`).
  Extraction is already there: `accepted_artifacts`, `provenance`, `run_report`, `run_records`,
  `rejected_log` (`api.py:622-669`). **This is the egress foundation — "pull durable artifacts out" is
  reading the object store keyed by accepted artifacts.**
- **Worker isolation is structural and already exactly what we need.** A sandbox worker **never** sees
  a host volume: `BackendSandboxDriver` reads role bytes into a per-cycle `readonly_inputs` dict
  (`sandbox/backend_driver.py:129-168`); `DockerBackend` materializes those bytes into a **fresh
  per-worker `tempfile.mkdtemp` staging dir**, mounts staging paths `-v`, and deletes it on exit
  (`provisioning/docker.py:79-169`). Data reaches a worker **only** through the control plane's
  provisioning path (`static_contents` → `readonly_inputs` → ephemeral staging). The answer key
  (`reserved_labels`) lives in the in-process verifier, never in a worker (`composition/fe.py:117`,
  `domains/feature_engineering.py`). **A client↔CP file-exchange channel is therefore isolatable from
  workers by construction — provided we route its files *through* the control plane, never straight at
  a worker.**
- **The networked seams are placed but unwired.** The `transport/` package is a generic
  service-to-service **port-RPC** (CP→worker: `RemoteVerifier`/`VerifierServer`, a one-route FastAPI
  `build_asgi_app`, `transport/http.py:99`) — *not* a public control API, but the proven pattern for
  hosting a FastAPI surface. The `JobQueue` port + `InProcessJobQueue` default
  (`contracts/jobqueue.py`) and the `RunRecord` store keyed by `(tenant_id, run_id)`
  (`control_plane/run_record.py`) are in the tree but **not yet wired into `ControlPlane`** (and
  `RunRecord` is written only at run *end*). The `service` extra (`fastapi`/`uvicorn`/`httpx`) and the
  `verity-reaper` console script (`pyproject.toml`) are the precedents for an HTTP layer and a CLI
  entrypoint.

The gap is small and well-shaped: **there is no client-facing control surface, no persistent task
catalog the surface selects from, and no file ingress/egress** — but every primitive those need
already exists. Nothing in the target requires touching the commit path, the store split, independence,
the lifecycle, or the sandbox-isolation boundary.

## The core idea

**Wrap, don't rebuild.** Add a thin **control surface** over a small, transport-agnostic
**`ControlService` core** that lives inside a **long-lived daemon** owning one `DockerBackend`, a
**task catalog** of the pre-built building blocks, and a **generic `ControlPlane` + store per task
instance** it multiplexes. The **daemon is the executor**; the client surface (CLI now, network HTTP
later) is a **thin client that reaches the daemon over a local IPC channel** — it never instantiates
its own control plane. The surface turns client
requests into the *same* calls `fe_run.py` makes today — `configure_<task>(cp, ...)` → `cp.run(...)` →
extraction — but on a standing process, parameterized per request, with files moving in and out
through the control plane (never through a worker).

The "no rebuild" boundary is **catalog-select-and-parameterize**: new task *instances*, sandbox
flavors, verifier approaches, and data, all at runtime; a genuinely new *domain* (new gate code) still
needs an image build, with a clean seam for runtime code registration left for later.

> The daemon shape below ((a)–(i)) is the proposed v1; the transport decision in (a)/(d) is the one
> the rest hangs on, and Sprint 1–2 validate it before the surface detail is locked.

## Decision

### Vocabulary

- **Task type** — a *catalog entry*: a named, pre-built domain + its configure-builder (`"fe"`,
  `"code"`). Image-resident. Selectable by name, and **self-describing** (it publishes its contract —
  see (c)).
- **Task** — a *durable instance* of a task type: the declarative `TaskRequest` that created it
  (persisted), identified by `task_id`, runnable later. Created at runtime, no rebuild. **Owns its own
  provenance store** (a `ControlPlane` + `SqliteStore` the daemon multiplexes — see (a)).
- **Sandbox configuration / verifier approach** — the declarative selections per task: a `ModelSpec` +
  `ProvisioningConfig` (sandbox) and a verifier-builder selection + knobs (verifier). Each carries
  **registry metadata** describing what it is and what it expects.
- **Run** — one execution of a task (`run_id`); **Exchange** — the client↔CP file channel.

The terms reuse the glossary (`docs/glossary.md`): tenant ⊃ task ⊃ run.

### (a) The long-lived daemon — the executor, reached over local IPC

`verity serve` keeps alive, as the process's state, a **`DockerBackend`**, the **`TaskCatalog`**, an
in-memory **`JobQueue`**, a daemon-level **run-records view**, and a global **run lock** — and it
**multiplexes a generic `ControlPlane` + its own `SqliteStore(path=…, object_dir=…)` per task
instance** (created/rehydrated from the persisted `TaskRequest`, (b)). It binds a **local IPC
endpoint** — a Unix-domain socket (e.g. `/run/verity.sock`) inside the container — and serves the
`ControlService` core over it.

**Why a store per task, not one shared store.** A single shared store across task instances would let
**same-typed instances bleed**: task B's `INCUMBENTS` slice (and the context manifest) would surface
task A's accepted artifacts *of the same type*, contaminating B's selection-gate scoring — a
correctness bug. A store per task scopes `INCUMBENTS`/manifest to the task **by construction**, at **no
kernel cost**: it is exactly the one-CP-per-run shape `composition/fe_run.py` already uses today. The
daemon is the multiplexer; the kernel is untouched. (Cross-*tenant* isolation — namespaced stores,
concurrent writers, GC — is the harder, deferred tier; see Deferred and issue #58.)

**Why an IPC endpoint and not bare `docker exec`.** A `docker exec verity-cp verity …` spawns a
*separate process*; if that process built its own `ControlService` it would get its own `ControlPlane`
and `DockerBackend` — sharing the filesystem but **not** the daemon's memory (the configured tasks, the
in-memory queue, the run lock). So the exec'd `verity` binary is a **client**: it connects to the
daemon's socket and issues a request; it holds no control plane. This is the load-bearing correction
from review — and it means a **minimal internal transport is part of v1**, not deferred to the external
HTTP layer ((e)). It keeps the CLI-first ergonomics (you still type `docker exec … verity …`) while
making the daemon story in this section real.

**Execution model.** `run` does not block on the (minutes-long, LLM-driven) loop: the daemon starts
the run as a **background `asyncio` task** and returns a `run_id`. A **run lock** serializes run
*bodies* (one at a time — the sole-mutator guarantee is trivially preserved); the serve loop stays free
to service `status`/`results` reads **concurrently** with an in-flight run (they are store reads + a
peek at in-memory run state). Serial execution is a v1 choice, not a constraint of the design;
concurrency is the deferred engine.

`fe_run.py` is **kept** as the batch/CI one-shot path and as a worked example — the daemon is an
*additional* entrypoint, not a replacement of the library API.

### (b) The task catalog — durable, declarative task definitions

A small **`TaskCatalog`**: a registry mapping a **task-type name** → an async **builder** with a
uniform signature, generalized from today's composition builders so it takes a **declarative
`TaskRequest`** rather than a pre-split Python object:

```python
# sketch — lives in verity.composition (allowed to know domains), not in control_plane
async def build(cp: ControlPlane, *, backend: WorkerBackend, request: TaskRequest) -> str: ...
```

`TaskRequest` carries: the **sandbox configuration** (`ModelSpec` fields + `ProvisioningConfig` knobs),
the **verifier approach** (selector + knobs), the **orchestration policy** (`max_cycles`, `refine_cap`,
`stop_on_accept`), and **input data references** (object hashes / exchange filenames the builder will
split + seed). It is **JSON-serializable** — which is the whole point of the catalog model: the
callables (`shape_validator`, `harvester`, gates) come from the *named domain*, not the wire.

**Tasks are durable runnable entities.** The `TaskRequest` is persisted (in the store). `task create`
records it and returns a `task_id`; `run <task_id>` configures the task onto the live CP from the
persisted request if it is not already resident (and re-configures it after a restart). This closes the
"queryable ≠ runnable" gap from review: you can create a task and run it later, across restarts,
because the *definition* persists — distinct from in-flight *run* recovery, which is deferred (#57).

**The builder is the riskiest new logic — treat it as such, not as a refactor.** Today the split and
the answer-key (`reserved_labels`) derivation happen in the *caller* (`fe_run` via
`stratified_split`/`subsample`). Moving them **inside the builder, against client-supplied data**, is
where a bad split or an answer-key leak would occur — a correctness/security surface. It gets its own
focused treatment (Sprint 1): split-correctness tests, and a **positive byte-provenance assertion**
that client-supplied data can never route the answer key (or any wrong-role bytes) into a worker (see
(f) and Guardrails).

The CP stays generic: the catalog and all domain knowledge live in `verity.composition`; the daemon
holds a `TaskCatalog` and drives `cp.register_sandbox/register_verifier/configure` *through the
builder*, never importing a domain itself.

### (c) Registry self-description — published contracts, not a compatibility matrix

A sandbox configuration and a verifier approach each carry **registry metadata** describing *what they
are*, so the client (a human or Claude) can judge fit. We **do not** maintain a task-type × verifier
allowlist or a precondition lock — that would re-encode verifier semantics into the catalog and fight
the verifier's deliberate **opacity** (ADR 0001), and the verifier is, per the spec, the least
definable component. We **trust the user** to select an appropriate pairing, and we make a poor pairing
**fail gracefully and informatively** rather than fail safe (below).

**The verifier's published contract has two halves** (this is the more important metadata — it tells
the user how to design the task and what approval means):

1. **Output-shape contract — *what the agent must produce*.** Sourced from the domain's **schema**
   (artifact types, `OperationSignature` inputs/outputs, `required_payload_keys`, `shape_validator`) +
   `domain_instructions`. This is **definable**, so it is the formal half.
2. **Approval semantics — *what the verifier will accept*.** A human-readable summary of the gate
   pipeline: which checks run, cheap-vs-hard staging, what "runs-clean → tentative" vs "selection →
   accepted on score-beats-incumbent" means, the reliability-ladder rung. This is the **fuzzy** half —
   prose the user reads, not a machine contract.

**Single source of truth for the shape contract.** The schema in the registry is what **renders into
the composed system prompt** that instructs the agent *and* what the registry **publishes** to the
user — one source, so the agent can't be told one shape and graded on another. (We do **not** police
the user's free-text task instructions; those can still be inaccurate — that is the user's
responsibility, not something the system can prevent.)

**Sandbox metadata** is lighter and informational: model identity, whether the sandbox can execute
code/shell, context-window size, and tool-calling caveats for weak local models (cross-ref
`docs/local-models.md`). It informs selection but is not a "contract" in the verifier's sense.

**Enforcement posture — trust + graceful failure.** An ill-suited verifier is **not** rejected at
`create`. At run time it surfaces through the loop's existing **degrade-don't-crash** machinery: a
shape mismatch becomes a gate refine/reject with a clear rationale fed back, and a verifier that cannot
operate becomes a recorded `GateUnavailable` failed cycle with its reason — never a cryptic crash or a
hang. **v1 requirement (tested):** a generally-incompatible task/verifier pairing produces an
informative, gracefully-recorded failed run, and `status`/`results` report *why*.

Metadata is exposed via `verity catalog` (and the future HTTP), so the user can read each verifier's
contract before choosing.

### (d) The control surface — CLI-first (`verity`), a thin client over the daemon

A **`verity` console script** (new `[project.scripts]` entry, precedent: `verity-reaper`) is the
**primary** surface, invoked via `docker exec verity-cp verity …`. It is a **thin client over the
daemon's local IPC socket** (a): it serializes the request, sends it, prints the reply — it holds no
`ControlPlane`. Verbs map to the mental model:

| Verb | Does |
|---|---|
| `verity catalog [--type fe]` | List task types, sandbox flavors, and verifier approaches **with their published contracts** ((c)). |
| `verity ingest <file>` | Copy a file from the **exchange** into the CP object store; returns an immutable object handle. |
| `verity task create --type fe --model … --max-cycles … --data <handle>` | Persist a `TaskRequest`; returns `task_id`. |
| `verity run <task_id> [--goal …]` | Start a run (background); returns `run_id` immediately. |
| `verity status <run_id>` | The `JobStatus` + a `RunReport` summary (cycles, outcomes). |
| `verity results <run_id>` | The full `RunReport` JSON + accepted-artifact ids. |
| `verity artifact get <id>` / `verity export <run_id>` | Write durable artifact bytes (code, DB) to the **exchange** out-dir. |

### (e) The control surface — external HTTP/REST (deferred; same core)

The same `ControlService` core is hosted over **FastAPI/uvicorn** (the `service` extra; mirror
`verifier/__main__.py`) as a **network-exposed** second adapter, REST endpoints in 1:1 correspondence
with the CLI verbs (`POST /tasks`, `POST /tasks/{id}/runs`, `GET /runs/{id}`, `GET
/runs/{id}/results`, `POST /objects`, `GET /artifacts/{id}`). **This is distinct from the `transport/`
port-RPC** (CP→worker) — it is a new *client→CP* control plane. It is **deferred past v1**: it needs
the network exposure + auth story (below), whereas the **internal** IPC transport (a) is required for
the CLI to function and ships in v1. Building both adapters over one core is why the core is
transport-agnostic from day one.

### (f) File I/O — ingress + egress through the control plane (v1: shared exchange volume)

**v1 (delivered): a shared "exchange" bind-mount, client↔CP only.** Compose mounts a host directory at
`/exchange` in the CP container (`/exchange/in`, `/exchange/out`). `ingest` reads a file from
`/exchange/in` and `put_object`s it into the store (an **immutable, content-addressed root input** —
resolving Open Question 2 toward copy-into-store, for clean provenance); `export`/`results` writes
durable artifact bytes (`get_object` keyed by `accepted_artifacts`/`provenance`) to `/exchange/out`.
Egress covers the target's "code or databases out": an accepted submission's code and any harvested
durable object are already content-addressed.

**The load-bearing isolation rule (non-negotiable):** the exchange volume is mounted **only into the CP
container, never into any worker**. Workers receive data **exclusively** through the CP provisioning
path — `static_contents` → `BackendSandboxDriver.readonly_inputs` → `DockerBackend` per-worker
ephemeral staging. A worker can therefore never see another task's files, durable artifacts
mid-export, or a verifier answer key. **This is enforced by two tests, not one:** (i) the exchange dir
is absent from every `WorkerSpec`/`docker run` argv (catches the *mount* mistake); and — the real
guarantee — (ii) a **byte-provenance** assertion that a worker's `readonly_inputs` for a cycle contain
**only** the intended role's bytes, and that `reserved_labels`/answer-key bytes never appear in any
worker input (catches the *builder logic* mistake that argv cannot).

**Operational notes (local footguns to handle):** namespace `/exchange/out` outputs by `run_id` to
avoid cross-run collisions; define `/exchange/in` cleanup after `ingest` (the bytes now live in the
store); and handle host-uid vs container-root **ownership** on the bind mount (the daemon often runs as
root — `chown`/`chmod` exports or run as a matched uid). We have hit Docker mount/ownership walls on
this project before; treat them as known.

**Seamed extension (deferred): over-the-wire file API.** `POST /objects` (upload bytes) + `GET
/artifacts/{id}` (download bytes) backed by the same object store, so a remote client needs no shared
volume — the proper realization of issue #3's networked data plane. The shared volume is the v1
local-testing fast-path and will be extended this way; the `ControlService` core is written so this is
a second ingress/egress adapter, not a reshape.

### (g) Run lifecycle & identity — wire the existing seams, serial execution

A `run` mints a `Job(job_id, tenant_id, run_id)` through the `JobQueue` port (`InProcessJobQueue`
default), the daemon executes the run body as the background task (a), and a `RunRecord` keyed by
`(tenant_id, run_id)` is recorded at run end (already minted per `run()` in 7.4.h). The client polls
`status`/`results` by `run_id` (submit → id → poll → fetch). **Concurrency and multi-tenancy stay a
deferred seam** (the multi-tenancy engine); v1 is serial, single default tenant.

### (h) Persistence — durable definitions and records (in-flight recovery deferred)

Each task's `SqliteStore(path=…, object_dir=…)` lives under a **mounted volume**, keyed by `task_id`
(a per-task DB + `object_dir`), so the provenance DB, the content-addressed blobs, the persisted
`TaskRequest`s ((b)), and the daemon-level `RunRecord`s survive restarts (already supported,
`store.py:282`). A restarted daemon re-derives its catalog from the installed domains and
re-configures any persisted task — re-opening *its* store — on demand. **In-flight *run*
recovery is explicitly out of scope (#57):** a run executing at a crash/restart is lost; the daemon's
boot reconcile marks orphaned non-terminal runs terminal so a client poll resolves, and `verity-reaper`
clears orphaned workers by label. A durable run-status engine (candidate: **Temporal**) is hardening
work.

### (i) Seam for future runtime code registration

`TaskCatalog` exposes a `register(type_name, builder)` method (the same shape the daemon uses for
built-in entries). A future plugin loader can discover builders from an installed entry-point group
(`verity.task_types`) or a mounted plugin package and call `register` at boot — no daemon change. This
ADR **places** the seam and **does not build** the loader; new *domain code* still requires an image
build in v1.

### Guardrails — what must NOT change

- The **kernel**: the commit path, the `Store`/`CommitSink` split, independence, the lifecycle table,
  context assembly. The control surface only *calls* the existing async API.
- The **sandbox-isolation boundary** ((f)): workers get bytes only via CP provisioning; no shared
  volume reaches a worker; the **byte-provenance** invariant (only the intended role's bytes; never the
  answer key) is the real guarantee, tested directly.
- **Per-task store isolation** ((a)): each task instance owns its own store, so no same-typed
  `INCUMBENTS`/manifest bleed across tasks. (Cross-tenant isolation is the deferred tier, #58.)
- The **genericity invariant**: `verity.control_plane` imports no domain; the catalog lives in
  `verity.composition`. The AST guard stays green.
- The **verifier's opacity** (ADR 0001): no task-type/verifier locking in the CP or catalog; fit is the
  user's judgment, surfaced via published metadata and enforced only by graceful run-time failure.
- **`fe_run.py` and the library API** remain valid (the daemon is additive).

## Deferred / seamed (explicitly not in this ADR's build)

- **Crash recovery for in-flight runs (#57).** Durable run-status + resumable execution; v1 sweeps
  orphaned runs terminal at boot and reaps workers. Candidate at hardening: **Temporal**.
- **The concurrency & multi-tenancy engine (#58)** — v1 delivers **per-task** store isolation ((a));
  the *residual, harder* tier is deferred: **cross-tenant** isolation (namespaced stores, no
  cross-tenant reads), the single-writer→Postgres store-engine upgrade, object GC (#9), and parallel
  live runs / fairness (the real `JobQueue` adapter + worker model, ADR 0003 c/h; ROADMAP
  "Multi-tenancy & run-control", #27). v1 is serial, single-tenant.
- **The over-the-wire file API** ((f) seam) — the networked data plane that removes the shared volume
  (issue #3 proper).
- **Runtime registration of new *domain code*** ((i) seam) — a plugin path/entry-point loader so a
  brand-new domain can be added without an image rebuild. v1 requires a build for new code.
- **Auth / authz** on the external HTTP surface ((e)). v1 ships only the internal IPC + CLI.

## Trust model (v1, stated plainly)

v1 exposes control **only** over a local IPC socket reached by `docker exec`, plus a shared exchange
volume — and authz is deferred. So the v1 trust boundary is: **whoever can exec into the container has
full control.** That is acceptable for local, single-operator use. The control plane is **not
network-exposed** until the external HTTP API ((e)) lands *with* an auth story.

## Relationship to existing work

- **Issue #3 (standing control-plane data plane)** — this ADR is its activation. The CLI + serve loop
  deliver the control half now; (f)'s over-the-wire API is the data half, seamed.
- **ADR 0003** — unchanged and depended-upon: the daemon launches workers exactly as 7.5 does.
- **The parked `feat/containerized-services-gvisor` branch** — its CP-as-service + transport seed is
  reusable reference for (e)'s HTTP layer, though its topology was superseded by ADR 0003. Mine the
  FastAPI/serve scaffolding; don't rebase it.
- **`docs/api-surface.md`** — the control-surface verbs are a thin façade over the API documented
  there; that reference's "Worked example" is the batch analogue of the daemon flow.

## Sprints (additive, each a green commit; suite green at every step)

1. **`ControlService` core + catalog + per-task durable stores.** A transport-agnostic async facade
   over a **per-task `ControlPlane`+store multiplexer**; `TaskCatalog` + JSON `TaskRequest`; refactor
   `configure_fe_task` / `configure_code_task` into catalog entries consuming a `TaskRequest` (old
   signatures preserved); persist `TaskRequest`s (durable, rehydratable tasks); wire `JobQueue` +
   daemon-level `RunRecord` over per-task persistent stores. **Carve-out:** the split + answer-key
   builder, with split-correctness tests + the **byte-provenance** isolation test (and a test that two
   same-typed task instances do **not** see each other's `INCUMBENTS`). Offline on `FakeBackend`.
2. **Daemon + internal IPC + CLI.** `verity serve` (background executor + run lock + concurrent status)
   over a Unix-domain socket; the `verity` console-script **client** over that socket
   (`catalog`/`ingest`/`task create`/`run`/`status`/`results`/`export`); registry metadata rendered by
   `verity catalog`. The graceful-incompatible-pairing test.
3. **Container wiring + live proof.** `Dockerfile.controlplane` ENTRYPOINT → the daemon;
   `infra/compose.fe.yml` mounts the **exchange** + **persistent store** volumes (+ a test that neither
   reaches a worker). Live: one running container, **two different tasks / two data sets, no image
   rebuild between them**, artifacts exported to the exchange; restart-then-`run` a pre-restart task.
4. **External HTTP/REST.** FastAPI over the same core (`service` extra), endpoint-parity with the CLI,
   an in-process/loopback test — paired with the auth story when it lands.

*(Then, as separate future work: the over-the-wire file API; the concurrency engine; the plugin loader;
crash recovery #57.)*

## Open questions (for review)

1. **Ingest immutability** — confirmed assumption: `ingest` copies bytes into the object store
   immediately (immutable, content-addressed root input), rather than registering a path read later.
2. **Multiple task *types* live at once** — **confirmed**: a single daemon exposes all installed task
   types. Multiple *instances* (even of the same type) are safe because each owns its own store ((a)).
3. **External HTTP auth** ((e)) — when it lands, what auth model (mTLS / token / reverse-proxy)? Out of
   scope for v1; flagged so the core's request context carries an identity hook from the start.
   **Resolved (ROADMAP 8.4): a shared bearer token** (`VERITY_API_TOKEN`) on every route but
   `/health`; the daemon refuses a network binding without it; the identity hook is in place.
   mTLS / OAuth / per-principal authz remain deferred (pair with multi-tenancy, #58).
