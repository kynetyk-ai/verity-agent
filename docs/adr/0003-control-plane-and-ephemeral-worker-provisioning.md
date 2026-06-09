# ADR 0003 — The long-lived control plane and backend-agnostic ephemeral-worker provisioning

- **Status:** Proposed — architecture only (no implementation); drafted on `integration-architecture-planning`
- **Date:** 2026-06-09 (revised same day to fold in external review — see "Decisions recorded after review")
- **Affects spec:** §3.3–§3.6 (the physical services), §3.9 (deployment & portability), §16 (the
  "*N* sandboxes against one control plane", multi-tenancy, and four-service seams). Realizes the
  Phase-7 seams already in the tree: `contracts/jobqueue.py` (`JobQueue`), `control_plane/run_record.py`
  (`RunRecord` keyed by `(tenant_id, run_id)`), and `tenant_id` on `TaskConfig`.
- **Supersedes the topology (not the contracts) of** the un-merged `feat/containerized-services-gvisor`
  branch (S1–S4): its transport seam and CP-as-service are reused; its *single standing sandbox
  service* is replaced by *launched, run-scoped workers* (see "Relationship to existing work").

## Context

We are moving from "three services in a Compose file" to the deployment shape the spec always
designed for: **a long-lived control plane that schedules many concurrent, ephemeral, per-cycle
(run-scoped) sandbox and verifier instances**, deployable to our own tenancy. Four requirements drive
this, and none may be foreclosed by what we build first:

1. **The control plane is long-lived and may manage many jobs at once,** across multiple tasks.
2. **A task (configuration) may use any sandbox or verifier in the registry** — swapping them is
   config, not code (already true via `ProviderRegistry` + `sandbox_key`/`verifier_key`).
3. **A job's configuration may, in future, request concurrent runs** — multiple sandboxes for the
   *same* task/job alongside sandboxes for other tasks/jobs. Not built now; nothing may take it off
   the table. (Spec §16: "*N* sandboxes against one control plane … diversity or race … designed-for,
   not built now.")
4. **Structured regeneration of the sandbox environment is an invariant** — every cycle the workspace
   is rebuilt clean and the agent's chat history is flushed, state re-derived from the authoritative
   store (§3.5). This holds **regardless of which registry configuration or deployment backend** is in
   play. It is the one property that survives everything.

Two cross-cutting constraints:

- **Containerize for our tenancy, likely Kubernetes *later* but local-first *now*.** The goal of this
  ADR is to keep k8s on the table without paying for it yet — local Docker today, k8s when we deploy.
- **No docker-in-docker.** But — we are a small team running **trusted** code. **A socket is
  acceptable if it is auditable and controlled by trusted code.** That relaxation reopens the cleanest
  pattern: the control plane (a trusted component) provisions sibling containers via a controlled
  runtime API, rather than peers nesting daemons or sharing a raw host socket.

## The core idea: two orthogonal axes

The decisive move is to **separate two things the current code conflates**:

- **Axis A — the *configuration* of a worker (logical, per-task).** *What* a sandbox/verifier is:
  which model, which framework, which gate pipeline. This is the `ProviderRegistry`, selected by
  `TaskConfig.sandbox_key`/`verifier_key`. It answers requirement (2).
- **Axis B — the *provisioning backend* (physical, per-deployment).** *Where and how* an instance of
  that worker is launched, observed, and destroyed: a local Docker container now, a Kubernetes Job
  later. This is **new**, and it answers requirements (1), (3), and the local-now/k8s-later constraint.

Today these are fused: a registered "container" sandbox config bakes a `docker run` into the driver.
That fusion is exactly what makes concurrency and k8s-portability hard. Splitting them means a task
picks a *configuration* and the *deployment* picks a *backend*, independently. The **structured
regeneration invariant (4) lives at their intersection** and is guaranteed by neither alone but by the
control-plane loop over both.

This mirrors the one design choice shared by every mature system in this space — SWE-ReX's
`Deployment` vs `Runtime` split, Kubernetes' CRI (`RunPodSandbox` vs `CreateContainer`), GitLab
Runner's `Executor`, Nomad's task-driver plugins: **"where it runs" is separated from "what you do
inside it."**

## Decision

### Vocabulary: run vs cycle (and a worker's lifetime)

These are **distinct** in the codebase (`docs/glossary.md`) and this ADR uses them precisely. A
**run** is one *execution* of a task (`ControlPlane.run`); a **cycle** is one
`read → propose → gate → commit` *iteration within* a run (`run_cycle`) — i.e. one proposal attempt. A
run **contains** cycles (`OrchestrationPolicy.max_cycles` bounds it; `cycles_run` counts them). The
consequence for provisioning: a worker is **scoped to a run in *identity*** (its labels carry the run)
but, under the default per-cycle disposal (d.4), its **lifetime is one *cycle*** — destroyed and
relaunched each proposal. Where this ADR says "per-run worker" it means the *alternative* in (d.4):
one worker spanning a whole run, flushing internally each cycle.

### (a) A backend-agnostic `WorkerBackend` port — the provisioning seam

Introduce one new port the control plane depends on for *provisioning*, distinct from the existing
`SandboxPort`/`VerifierPort` which remain the contracts for *talking to a running worker*. The
backend is deliberately **dumb**: it knows how to create/observe/kill one labelled unit of work and
nothing about gates, provenance, or policy (those stay in the control plane, per Principle 9).

```python
class WorkerBackend(Protocol):
    # provision / lifecycle  (the "Deployment" layer)
    def launch(self, spec: WorkerSpec) -> WorkerHandle: ...   # create + start; spec carries the
                                                              # labels {tenant, job, run, cycle, role},
                                                              # image, config-key, env, mounts, limits,
                                                              # and the OCI runtime (default plain;
                                                              # see "Isolation posture")
    def status(self, h: WorkerHandle) -> WorkerStatus: ...    # phase + health
    def wait(self, h, timeout=None) -> ExitResult: ...
    def logs(self, h, *, follow=False) -> Iterable[bytes]: ...# worker stdout/stderr (see "(k) Observability")
    def stop(self, h, *, grace: float) -> None: ...           # SIGTERM -> SIGKILL
    def destroy(self, h) -> None: ...                         # remove + reclaim (idempotent)

    # fleet management  (label-keyed; powers reconciliation + GC)
    def list(self, selector: Labels) -> list[WorkerHandle]: ...
    def reap(self, selector: Labels) -> int: ...              # delete all matching (orphan sweep)

    # optional
    def recover(self, h) -> WorkerHandle | None: ...          # re-adopt a live worker after a CP restart
```

The control plane *talks to* a launched worker over the existing `SandboxPort`/`VerifierPort` via the
transport (the `RemoteSandbox`/`RemoteVerifier` + HTTP seam already built in S1–S2), addressed at the
instance the backend just launched. So there are two layers, cleanly: **`WorkerBackend.launch` makes
an instance exist; `SandboxPort.serve_context`/`collect_proposal` drives it.** This is the
SWE-ReX/CRI deployment-vs-runtime split, applied to our ports.

> Minimal method set is load-bearing and consistent across the prior art: `launch`, `wait`, `status`,
> **`logs`**, `destroy`, plus `list`/`reap` for fleet GC and an optional `recover`. We fold "drive"
> (`exec`) into the existing ports rather than re-inventing it. We **deliberately omit `capabilities()`**
> (a backend-negotiation method): the moment the control plane branches on it, backend differences leak
> upward and erode the "CP is backend-blind" property the whole design rests on. Push any backend
> difference *down* into the backend; add negotiation only when a concrete need forces it (YAGNI).

### (b) Labels are the universal key

Every worker is stamped, at launch, with a composite identity as labels:
`{harness: verity, tenant: <t>, job: <j>, run: <r>, cycle: <n>, role: sandbox|verifier, config: <key>}`.
The `cycle` ordinal matters under per-cycle disposal (d.4): consecutive workers of the *same* run are
distinct units, so GC and addressing never alias cycle *N*'s worker with cycle *N+1*'s. **All
selection, concurrency accounting, and garbage collection become label queries** — and those map
*identically* onto Docker label filters and Kubernetes label selectors. This is how Testcontainers
(`sessionId`), GitLab Runner, and k8s all do it, and it is what makes the Docker→k8s swap mechanical.

### (c) The control plane is a level-triggered reconciliation loop

The long-lived control plane does **not** "spawn and forget." It runs the controller pattern: read
desired state, read observed state, drive the diff, repeat.

```
each tick / on a job event:
  # one live worker per ACTIVE RUN, for that run's CURRENT CYCLE (run != cycle; see Vocabulary)
  desired = {(run, run.current_cycle) for run in active_runs}   # from the JobQueue + the store
  actual  = backend.list({harness: verity})
  for (run, cycle) in desired - actual:  backend.launch(spec_for(run, cycle))  # (re)generate clean
  for w in actual - desired:             backend.destroy(w)     # reap (a finished cycle, or no owning run)
  enforce per-tenant concurrency slots before launching         # slots count concurrent RUNS — (h)
  periodically: backend.reap(stale-selector)                    # belt-and-suspenders GC
```

Per-cycle disposal makes the launch/destroy churn *within* a run: when run *R* finishes cycle *N*
(worker destroyed after harvest+commit), the control plane advances *R* to cycle *N+1* and the next
tick launches a fresh worker for it. Slots bound concurrent **runs**; regeneration happens per
**cycle**.

Because the loop is **level-triggered and idempotent**, "throw the worker away and rebuild it from the
store" is a *routine* operation, not an exceptional one — which is precisely why the §3.5 regeneration
invariant is safe to perform every cycle, and why a missed event or a control-plane restart self-heals.
This keeps the control plane "deliberately unintelligent" (Principle 9): it reconciles; it does not
reason.

### (d) The structured-regeneration contract, made precise (requirement 4)

Regeneration is one event with a fixed order and a fixed source of truth, **independent of config
(Axis A) and backend (Axis B):**

1. **Source of truth is the store, never the worker.** Durable memory lives only in the typed-provenance
   store (§2, Principle 2; §3.5). A worker holds nothing across cycles.
2. **Ordering is fixed and load-bearing:** `propose → control plane harvests the outbox (content-addresses
   objects) → commit protocol → regenerate the worker`. Harvest-before-teardown (§3.4) is a hard
   constraint: the ephemeral workspace is about to be destroyed.
3. **What a "cycle" is — and why disposal is affordable.** A cycle is **one proposal attempt** in
   `read → propose → gate → commit`, **not** an agent step. The agent takes many model steps / tool
   calls *inside* one cycle, inside one worker; `regenerate()` fires only at the cycle boundary (after
   a proposal is resolved). So a cycle is **minutes-scale** (the agent does substantial work — for the
   FE domain it writes and runs a training script), which means a few seconds of container start-up
   per cycle **amortizes cleanly**. This is the number the cold-start question turns on, stated
   explicitly: per-cycle disposal is affordable *because* a cycle is a proposal, not a step.
4. **Regeneration = a clean rebuild, two parts flushed as one event (§3.5):** the writable workspace
   (`scratch/`, `outbox/`) is discarded and re-provisioned to the invariant contract; the agent
   process / chat history is flushed. **Disposal granularity is a decision, defaulting to per-cycle
   worker disposal:** destroy the worker and launch a fresh one each cycle, so process death gives a
   total flush with no reliance on enumerating in-memory state (the lesson from the dropped
   `SubprocessCodeRunner` work: the flush must be *structural*, not best-effort). The alternative — a
   **per-run** worker that flushes *internally* each cycle (workspace `rmtree` + a fresh per-cycle
   subprocess, the S1 model) — trades a weaker structural guarantee for one container per run instead
   of one per cycle; it is the right choice *only if* cycles ever become cheap/frequent enough that
   start-up dominates, which the per-proposal definition above makes unlikely. Default per-cycle; keep
   per-run as a `WorkerSpec`-level option, not a hidden behaviour.
5. **Re-hydration is explicit:** the fresh worker is rebuilt from the store — the read-only roles
   (`data/`, `context/`, `spec/`), the regenerated context (§9), and any provisioned objects are
   re-materialized onto the clean workspace. The agent re-reads current state; it never drifts on
   stale in-context memory.
6. **The invariant is asserted, not assumed:** a no-cross-cycle-bleed property test (a marker written
   in cycle *N* is absent in cycle *N+1*) is the acceptance check, against every backend.

This is the industry's "cattle, not pets" applied to the spec's ephemerality: the worker is immutable
and disposable; the store is the system of record; each worker is re-hydrated from it.

### (e) Garbage collection is belt-and-suspenders

A long-lived, many-worker system must never leak workers, even if the control plane crashes. Two
mechanisms, by backend:

- **Docker (now):** every worker carries its run/session labels; a **label-based reaper** (the
  Testcontainers/Ryuk pattern — a sidecar with a connection death-switch) sweeps orphans. Plus the
  reconciliation loop's `reap` of any worker whose owning run is terminal.
- **Kubernetes (later):** `ownerReferences` cascading deletion + `ttlSecondsAfterFinished` on Jobs
  (the docs recommend always setting it). Lingering Pods are a documented cause of cluster degradation,
  so GC is not optional.

### (f) The trust boundary is the trusted-code assumption; rootless caps blast radius; the proxy is hygiene

Be honest about what is load-bearing here, in this order:

1. **The trusted-code assumption *is* the boundary.** We run our own code on our own infrastructure;
   the control plane is the trusted, privileged orchestrator by design (Principle 9 — sole mutator of
   durable state, now also sole mutator of the *runtime*). Nothing below is a containment boundary
   against a *compromised* control plane — they reduce surface and blast radius. The day the code is
   no longer trusted (untrusted/multi-tenant), the boundary must change — and that is exactly when a
   **provisioning service with a validating API** is warranted (see "(l) Provisioning: in-process vs a
   separate service"), because only it can semantically refuse a dangerous launch.
2. **Rootless Docker is the default posture, not an option.** The Docker socket is root-equivalent;
   rootless runs the daemon as an unprivileged user, so even a misused `create` cannot reach host
   root. This is the real blast-radius cap, so it is the baseline — *not* an afterthought.
3. **`docker-socket-proxy` is hygiene (surface reduction), not containment.** It whitelists API
   *sections* (`CONTAINERS`, `EXEC`, the `POST`s the provisioner needs) and 403s the rest (`IMAGES`,
   `SWARM`, …). But it gates by endpoint+method and **does not inspect request bodies** — once
   `CONTAINERS`+`POST` are allowed (which you need to launch anything), a compromised control plane
   could still `POST /containers/create` with `--privileged` or `-v /:/host` and the proxy would pass
   it. So the proxy reduces the reachable API; it is **not** a barrier against a compromised CP. (Real
   request validation needs the provisioning service of (l).)
4. **One audited choke point.** Exactly one component — the provisioner inside the control plane —
   holds the (rootless, proxied) credential; every `launch`/`destroy` is structured-logged with its
   `(tenant, run)` labels. Auditability comes from the single choke point, which is the Principle-9
   boundary extended to the runtime.

**Kubernetes (later)** is the native analogue and a genuinely stronger boundary: a **namespaced
`Role`** (not a `ClusterRole`) granting only `create`/`delete`/`get`/`list` on `jobs`/`pods` in one
namespace, bound to the control plane's ServiceAccount; the API server audits the calls, and
`PodSecurity`/admission can reject privileged specs (the body-inspection the socket-proxy can't do).

So the credential is backend config; what is *not* negotiable is being explicit that **trusted code is
the boundary today** — rootless + proxy + audit are defense-in-depth on top of it, not a substitute.

### (g) Sandbox and verifier are symmetric ephemeral workers

Both are launched by the `WorkerBackend` (per cycle by default, scoped to the run — see Vocabulary),
distinguished only by their `role` label, and driven over their respective ports
(`SandboxPort` / `VerifierPort`). This honours the decision that verifiers get the same ephemeral
treatment as sandboxes. It is consistent with §3.6 ("the verifier is **queue-fronted and async**"): a
freshly-launched verifier worker is one valid shape, and a shared queue-fronted verifier *pool* is
simply a backend that returns a handle to a pooled instance instead of launching a fresh one — the
abstraction accommodates both without the control plane knowing which.

### (h) Concurrency and multi-tenancy bounds live in the orchestrator

Per requirement (3) and §16, the design must allow *N* concurrent workers across runs/jobs/tenants.
The **bound and the fairness policy live in the long-lived control plane**, not the backend (which
stays dumb):

- **Bound:** per-tenant concurrency **slots/semaphores** (the Temporal "task-slot + poller" model) —
  a tenant may hold at most *k* live workers; the scheduler only launches when a slot is free, giving
  implicit backpressure.
- **Fairness:** per-tenant queues with per-queue caps, so one tenant cannot starve others (avoid a
  global FIFO).
- **Isolation:** Docker — label scoping (+ optionally a per-tenant network); Kubernetes — a
  **namespace per tenant** with `ResourceQuota`/`LimitRange`. The `tenant_id` already threaded through
  `TaskConfig`/`RunRecord` is the key.

None of this is built now; the seams (`JobQueue`, `RunRecord`, `tenant_id`, the slot accounting point)
are placed so it slots in without reshaping the control plane.

### (i) Isolation posture — a knob, default plain (don't over-engineer the threat model)

Isolation strength is a **`WorkerSpec.runtime` knob, not a baked-in requirement**, and it **defaults
to a plain container (`runc`)**. For the current posture — **trusted code, single-user, local, not
internet-facing** — a plain container (namespaces + cgroups + dropped caps + read-only root + no
network), itself inside Docker's hardware-isolated Linux VM on a developer Mac, is a sufficient
boundary; the realistic failure mode is buggy/dumb generated code, not a motivated escape.

**gVisor (`runsc`) and microVMs (Firecracker/Kata) are the documented *upgrade*, gated on the threat
model actually changing** — i.e. when Verity runs *untrusted* or *multi-tenant* code (a hosted product
running strangers' submissions). At that point it is one field (`WorkerSpec.runtime=runsc`, or a k8s
`RuntimeClass`), no redesign. Firecracker needs Linux/KVM and is not a Mac-local option.

Concretely, defaulting to plain **removes** complexity rather than adding it: gVisor cannot reach
Docker's embedded DNS (google/gvisor#115), which is the *only* reason the earlier compose sketch
needed static IPs + explicit upstream DNS. A plain-container default uses normal Docker service-name
DNS, so that workaround simply disappears. Enabling gVisor "now" for trusted code was premature; the
knob is the part worth keeping.

### (j) Commit under concurrency — the independence property is the whole ballgame

Concurrency admission (slots/fairness) is the easy half; the *commit protocol under N concurrent runs*
is where the dragons live. The deciding question is **independence**:

- **Across different tasks/tenants → genuinely independent.** Each run writes only its own
  `(tenant, run)` provenance; there is no shared mutable state, so this is embarrassingly parallel.
  This is the common case and it is trivial.
- **Same-task "race" mode (§16) → shared state.** When *N* runs compete on one task (best-wins), the
  gate is *supersede-on-beat against the **incumbents** slice* (the FE optimizer): two runs comparing
  themselves to "the current best" and superseding it is a **read-modify-write race** on shared state.

The answer Verity already has: **parallel proposal, serialized commit.** The commit path is the single
privileged mutator and is **atomic (`transaction()`)** — so proposals fan out concurrently, but the
supersede/incumbent decision happens *inside the commit transaction* (re-read incumbents under the
lock at commit time, not from the gate's earlier view). The control plane already serializes here; the
decision is to **keep same-task-race correctness in the commit transaction**, never in the gate's
read. Record explicitly: independent runs need nothing new; same-task race needs the incumbent
comparison to be transactional. (This is also what pushes the Postgres store sooner — see risks.)

### (k) Observability across the fleet

A long-lived, many-worker system is undebuggable without per-`(tenant, run)` log + telemetry
aggregation — it is not optional. Decision:

- **Workers already emit structured logs** (`structlog` JSON to stdout) and the agent already ships
  `__telemetry__.json` per cycle (harvested into the `RunReport`). Stamp both with the
  `{tenant, run, cycle, role}` labels at the source.
- **`logs()` on `WorkerBackend`** is for *pull* (debugging a specific worker); **a telemetry/log sink**
  is for *push* aggregation across the fleet. The control plane (or the worker) ships structured
  events to a sink — stdout-collected by the platform on k8s; an explicit sink (a log shipper / the
  `RunRecord` store) on the Docker backend, which does **not** get fleet logging for free. Name the
  sink as a deployment choice; do not leave it implicit.

### (l) Provisioning: in-process vs a separate service

Should sandbox/verifier provisioning be pulled out of the control plane into its own service that the
CP merely instructs ("spin up / tear down")? **For v1, no — keep provisioning in-process behind the
`WorkerBackend` port.** Why this forecloses nothing and why it is the right call now:

- **The port is the seam, not the service.** Extracting provisioning into a standalone service later
  is a `RemoteWorkerBackend` implementation that RPCs to it — the *same* refactor already done for the
  sandbox and verifier (`RemoteSandbox`/`RemoteVerifier`). The service is one *deployment* of the
  port, registered like any other; keeping it in-process now costs no future option.
- **It fits the spec.** Provisioning is *mechanical*, not intelligent ("spin up / tear down"), so it
  belongs in the control plane's "mechanical orchestration" role (Principle 9) — not a microservices
  violation.
- **On Kubernetes the question largely dissolves.** k8s *is* the provisioner; the CP becomes a
  controller calling the API server with a scoped ServiceAccount. A bespoke provisioner service in
  front of the k8s API would be a redundant layer. So "separate provisioner" is mostly a *Docker-backend*
  question (where the socket lives), not a universal principle.

**When it *is* warranted** (record as a designed-for-not-built trigger): (1) **untrusted/multi-tenant
code** — a provisioner service with a *validating* API (`launch(config_key, run_id, labels)`, no
"create arbitrary container" verb) is the real containment boundary the socket-proxy cannot be (f.3),
and it co-arrives with gVisor (i) under the same threat-model shift; (2) **multiple/sharded control
planes** sharing one scheduler; (3) a **k8s** deployment (which resolves it natively). Until one of
those, in-process is the bounded, accepted cost of the trusted-code assumption.

## Docker-now ↔ Kubernetes-later mapping

The abstraction is portable because both runtimes expose the same primitives — *create a labelled
unit, drive it, select by label, delete by label*:

| Concern | `DockerBackend` (now) | `K8sBackend` (later) |
|---|---|---|
| worker unit | a `container` (`/containers/create` + `/start`) | a `Job` → `Pod` |
| run-to-completion + retry | control-plane policy | `Job.spec.backoffLimit` / `completions` |
| identity / selection | container **labels** | Pod/Job **labels** + `namespace` |
| find my workers | `GET /containers/json?filters=label` | list with `labelSelector` |
| inject config / re-hydrate | env + bind/volume mounts | env + `ConfigMap`/`Secret` + `volumeMounts` |
| drive the worker | `SandboxPort`/`VerifierPort` over HTTP | same ports over HTTP |
| isolation runtime | `WorkerSpec.runtime` — **default plain**; `runsc` opt-in | default `RuntimeClass`; `gvisor` opt-in |
| status / health | `inspect` State / healthcheck | Pod `phase` / readiness probe |
| stop / destroy | `/stop` then `/rm` | delete Pod/Job (cascade) |
| GC backstop | label reaper (Ryuk-style) | `ttlSecondsAfterFinished` + `ownerReferences` |
| tenant isolation | label + per-tenant network | namespace + `ResourceQuota` |
| trust boundary | trusted code + **rootless** daemon (proxy = hygiene) | namespaced RBAC `Role` + admission |

Only the `DockerBackend` is built first; `K8sBackend` is a second implementation of the *same* port —
designed-for, not built.

## Relationship to existing work

- **Reused as-is:** the async ports (`SandboxPort`/`VerifierPort`) + `ProviderRegistry` (Axis A); the
  transport seam (`RemoteSandbox`/`RemoteVerifier`, `*Server`, `HttpTransport`, the envelope/error
  map) from S1–S2 — the control plane drives launched workers over exactly this; `JobQueue`,
  `RunRecord`, `tenant_id` (the long-lived/multi-tenant scaffolding); the `runtime` knob (S3) → becomes
  a `WorkerSpec` field, **defaulting to plain** (see "Isolation posture"); `Store` behind a volume /
  external backend (§3.9, §4.2).
- **Changes from the S1–S4 sketch:** S1 stood up a **single standing sandbox service** (one container,
  the agent as a per-cycle subprocess *inside* it). That is the wrong cardinality for requirement (3):
  the sandbox must be **instantiable per run**, launched by the `WorkerBackend`, not a singleton. The
  per-cycle-subprocess idea for ephemerality is retained *inside* a worker as one valid flush
  mechanism, but worker-instance disposal (d.3) is the backend-level regeneration. The compose
  static-IP/DNS workaround (an artifact of the singleton topology *and* of running gVisor by default)
  is dropped: the backend addresses workers by handle, and the default plain runtime restores normal
  Docker DNS anyway.
- **Informs, doesn't block:** the dropped `SubprocessCodeRunner` question (FE untrusted-code
  execution) becomes a *worker* like any other — the verifier's code execution is just another
  launched worker the backend isolates, which resolves the isolation regression that
  killed it: the isolation is the worker boundary (gVisor container / k8s Pod), provisioned by the
  trusted backend, not an in-process subprocess.

## Extensibility — the two axes, restated

| To vary… | Change… | Cost |
|---|---|---|
| the model / framework / gates of a worker | register a config in `ProviderRegistry` (Axis A) | one `register(...)`; task picks it by key |
| the substrate (Docker → k8s) | a new `WorkerBackend` implementation (Axis B) | one backend class; no control-plane change |
| the isolation runtime (plain → gVisor → microVM) | a `WorkerSpec.runtime` field | a config value |
| from 1 run to N concurrent runs | the scheduler launches N labelled workers; bound by per-tenant slots | no contract change |
| from single-user to multi-tenant | `tenant_id` label + namespace/quota in the backend | the seam is already placed |

The control plane depends only on `WorkerBackend` + the ports + `ProviderRegistry`; everything else is
a registration or a backend swap.

## Consequences

- **Local-now, k8s-later is real, not aspirational:** the same control-plane loop and the same worker
  contracts run on a Docker host today and a cluster later by swapping one port implementation.
- **Concurrency and multi-tenancy are unblocked structurally** without being built — the cardinality
  (ephemeral per-cycle workers, run-scoped, labelled, slot-bounded) is correct from day one.
- **The regeneration invariant is strengthened**, not weakened: it becomes a structural property of
  worker disposal + reconciliation, asserted by a property test against every backend.
- **No anti-pattern, but no false comfort either:** privilege is concentrated in one logged choke
  point on a **rootless** daemon, not smeared across peers or nested daemons — *and* the ADR states
  plainly that the trusted-code assumption is the actual boundary today (rootless + proxy + audit are
  defense-in-depth, not containment against a compromised CP).
- **Cost:** a new provisioning layer (the `WorkerBackend`, the reconciliation loop, the GC reaper,
  rootless + the socket-proxy) that did not exist; the control plane grows a scheduler responsibility
  (still mechanical — reconcile, don't reason). The existing in-process and single-container paths
  remain valid as the trivial backend for dev/tests.

## Decisions recorded after review

External review surfaced six items; the load-bearing ones are now decided in the body above. Summary:

- **Worker addressing is on the critical path (not deferred) — address *by handle*.** You cannot drive
  a worker you cannot reach, and `WorkerHandle` + `launch`'s return type *are* the addressing decision.
  Decision: the backend returns the address in the `WorkerHandle` (Docker: a published port read back
  from `inspect`, or a shared user-defined network + the worker's address; k8s: the Pod/Service
  address) and the control plane drives the worker **by handle**, never by Docker service-name DNS.
  This is the portable, k8s-compatible key and is also what gVisor would *require* (google/gvisor#115).
  (Note: the plain-runtime default merely removes the *forced* static-IP workaround — it does **not**
  mean we lean on service-name DNS; addressing is by handle regardless. This reconciles an earlier
  inconsistency.)
- **CP restart = reap-and-regenerate is *correct*, not just a style choice; `recover` is a deferrable
  cost optimization.** The sharp edge is harvest-before-teardown across a crash. But two existing
  properties resolve it: harvest-before-teardown means an un-harvested proposal **never entered durable
  state**, and the commit is an **atomic `transaction()`** — so every crash window resolves to exactly
  one clean state (durably committed → just reap the worker; or as-if-never-proposed → re-propose).
  The worst case is **re-running an agent cycle (wasted compute/$), never data loss or corruption.** So
  reap is correct; `recover` only saves the re-run and is deferrable. (For a system *without* atomic
  commit + harvest-ordering, `recover` could be a correctness requirement — Verity's commit discipline
  is what downgrades it.)
- **Verifier shape — default a freshly-launched verifier worker per cycle** (symmetric with the
  sandbox, (g)); a shared queue-fronted pool (§3.6) is a backend variant the abstraction already
  permits, adopted only if launching a verifier worker per cycle proves too costly.
- **Scheduler placement — in-process now** (resolved by (l)); the `WorkerBackend` port keeps extraction
  to a `RemoteWorkerBackend` registration later.

## Remaining open questions

- **State store under concurrency.** Same-task race + many concurrent runs writing provenance pushes
  toward the Postgres backend (§4, and (j)) sooner than single-user did; the `Store` interface already
  permits it, but the migration timing is unsettled.
- **Telemetry sink choice on the Docker backend** ((k)) — which concrete log/telemetry sink, decided at
  deployment time.

## Explicitly deferred (designed-for, not built)

The `K8sBackend`; the multi-tenancy *engine* (namespaces, quotas, auth — §16); **actually running**
concurrent/parallel runs against one task (§16 "*N* sandboxes"); the four-service split (workspace as
its own worker — §3.3). Each is a backend, a config, or a scheduler-policy change over this
architecture, not a rework.

## References

**Spec:** §3.3 (physical services + the runtime/workspace seam), §3.4 (the unintelligent control
plane, async + task-keyed), §3.5 (the regeneration event; harvest-before-teardown), §3.6
(queue-fronted advisory verifier), §3.9 (containerizable, no host assumptions, store behind a volume),
§16 (*N* sandboxes; multi-tenancy; four-service split — all "designed-for, not built").

**Prior art (primary):** SWE-ReX deployment/runtime split
(<https://swe-rex.com/latest/architecture/>); Kubernetes CRI
(<https://kubernetes.io/docs/concepts/containers/cri/>); GitLab Runner executor interface
(<https://docs.gitlab.com/runner/development/internal/engineering/executor_interface/>); Nomad task
drivers (<https://github.com/hashicorp/nomad/blob/main/plugins/drivers/driver.go>); Kubernetes Jobs,
TTL-after-finished, and garbage collection
(<https://kubernetes.io/docs/concepts/workloads/controllers/job/>,
<https://kubernetes.io/docs/concepts/workloads/controllers/ttlafterfinished/>,
<https://kubernetes.io/docs/concepts/architecture/garbage-collection/>); Kubernetes RBAC good
practices (<https://kubernetes.io/docs/concepts/security/rbac-good-practices/>); Testcontainers
Ryuk/reaper (<https://golang.testcontainers.org/features/garbage_collector/>); rootless Docker
(<https://docs.docker.com/engine/security/rootless/>); docker-socket-proxy
(<https://github.com/Tecnativa/docker-socket-proxy>); Temporal worker slots/pollers
(<https://docs.temporal.io/develop/worker-performance>).
