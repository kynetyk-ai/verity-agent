# ADR 0003 — The long-lived control plane and backend-agnostic ephemeral-worker provisioning

- **Status:** Proposed — architecture only (no implementation); drafted on `integration-architecture-planning`
- **Date:** 2026-06-09
- **Affects spec:** §3.3–§3.6 (the physical services), §3.9 (deployment & portability), §16 (the
  "*N* sandboxes against one control plane", multi-tenancy, and four-service seams). Realizes the
  Phase-7 seams already in the tree: `contracts/jobqueue.py` (`JobQueue`), `control_plane/run_record.py`
  (`RunRecord` keyed by `(tenant_id, run_id)`), and `tenant_id` on `TaskConfig`.
- **Supersedes the topology (not the contracts) of** the un-merged `feat/containerized-services-gvisor`
  branch (S1–S4): its transport seam and CP-as-service are reused; its *single standing sandbox
  service* is replaced by *per-run launched workers* (see "Relationship to existing work").

## Context

We are moving from "three services in a Compose file" to the deployment shape the spec always
designed for: **a long-lived control plane that schedules many concurrent, ephemeral, per-run
sandbox and verifier instances**, deployable to our own tenancy. Four requirements drive this, and
none may be foreclosed by what we build first:

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

### (a) A backend-agnostic `WorkerBackend` port — the provisioning seam

Introduce one new port the control plane depends on for *provisioning*, distinct from the existing
`SandboxPort`/`VerifierPort` which remain the contracts for *talking to a running worker*. The
backend is deliberately **dumb**: it knows how to create/observe/kill one labelled unit of work and
nothing about gates, provenance, or policy (those stay in the control plane, per Principle 9).

```python
class WorkerBackend(Protocol):
    # provision / lifecycle  (the "Deployment" layer)
    def launch(self, spec: WorkerSpec) -> WorkerHandle: ...   # create + start; spec carries the
                                                              # labels {tenant, job, run, role},
                                                              # image, config-key, env, mounts, limits,
                                                              # and the OCI runtime (default plain;
                                                              # see "Isolation posture")
    def status(self, h: WorkerHandle) -> WorkerStatus: ...    # phase + health
    def wait(self, h, timeout=None) -> ExitResult: ...
    def stop(self, h, *, grace: float) -> None: ...           # SIGTERM -> SIGKILL
    def destroy(self, h) -> None: ...                         # remove + reclaim (idempotent)

    # fleet management  (label-keyed; powers reconciliation + GC)
    def list(self, selector: Labels) -> list[WorkerHandle]: ...
    def reap(self, selector: Labels) -> int: ...              # delete all matching (orphan sweep)

    # optional
    def recover(self, h) -> WorkerHandle | None: ...          # re-adopt a live worker after a CP restart
    def capabilities(self) -> Capabilities: ...               # exec? file IO? per-tenant namespaces?
```

The control plane *talks to* a launched worker over the existing `SandboxPort`/`VerifierPort` via the
transport (the `RemoteSandbox`/`RemoteVerifier` + HTTP seam already built in S1–S2), addressed at the
instance the backend just launched. So there are two layers, cleanly: **`WorkerBackend.launch` makes
an instance exist; `SandboxPort.serve_context`/`collect_proposal` drives it.** This is the
SWE-ReX/CRI deployment-vs-runtime split, applied to our ports.

> Minimal method set is load-bearing and consistent across the prior art: `launch`, `exec`/drive,
> `wait`, `status`, `logs`, `destroy`, plus `list`/`reap` for fleet GC and an optional `recover`. We
> fold "drive" into the existing ports rather than re-inventing `exec`.

### (b) Labels are the universal key

Every worker is stamped, at launch, with a composite identity as labels:
`{harness: verity, tenant: <t>, job: <j>, run: <r>, role: sandbox|verifier, config: <key>}`. **All
selection, concurrency accounting, and garbage collection become label queries** — and those map
*identically* onto Docker label filters and Kubernetes label selectors. This is how Testcontainers
(`sessionId`), GitLab Runner, and k8s all do it, and it is what makes the Docker→k8s swap mechanical.

### (c) The control plane is a level-triggered reconciliation loop

The long-lived control plane does **not** "spawn and forget." It runs the controller pattern: read
desired state, read observed state, drive the diff, repeat.

```
each tick / on a job event:
  desired = runs that should have a live worker     # from the JobQueue + the provenance store
  actual  = backend.list({harness: verity})
  for run in desired - actual:  backend.launch(spec_for(run))   # (re)generate a clean worker
  for w   in actual  - desired: backend.destroy(w)              # reap workers with no owning run
  enforce per-tenant concurrency slots before launching         # (h)
  periodically: backend.reap(stale-selector)                    # belt-and-suspenders GC
```

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
3. **Regeneration = a clean rebuild, two parts flushed as one event (§3.5):** the writable workspace
   (`scratch/`, `outbox/`) is discarded and re-provisioned to the invariant contract; the agent
   process / chat history is flushed. We realize the flush by **worker-instance disposal** — destroy
   the worker (or its per-cycle process) and launch a fresh one — so process death gives a total flush
   with no reliance on enumerating in-memory state. (This is the lesson from the dropped
   `SubprocessCodeRunner` work: the flush must be structural, not best-effort.)
4. **Re-hydration is explicit:** the fresh worker is rebuilt from the store — the read-only roles
   (`data/`, `context/`, `spec/`), the regenerated context (§9), and any provisioned objects are
   re-materialized onto the clean workspace. The agent re-reads current state; it never drifts on
   stale in-context memory.
5. **The invariant is asserted, not assumed:** a no-cross-cycle-bleed property test (a marker written
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

### (f) The trusted provisioner holds one narrowed, audited credential

The relaxation ("a socket is OK if auditable and trusted") is realized as: **exactly one component —
the control plane's provisioner — talks to the runtime API, through a least-privileged, audited
credential, and nothing else does.** That is the same Principle-9 boundary ("only the control plane
mutates durable state") extended to "only the control plane mutates the *runtime*."

- **Docker (now):** the provisioner talks to the Docker Engine API **through a narrowing proxy**
  (`Tecnativa/docker-socket-proxy`) that whitelists exactly the API sections it needs
  (`containers`, `exec`, the specific `POST`s) and 403s everything else; the raw socket is never
  mounted into the control-plane image. Optionally rootless Docker to cap blast radius. Every
  `launch`/`destroy` is structured-logged with its `(tenant, run)` labels — one audited choke point.
- **Kubernetes (later):** the native analogue — a **namespaced `Role`** (not a `ClusterRole`) granting
  only `create`/`delete`/`get`/`list` on `jobs`/`pods` in one namespace, bound to the control plane's
  ServiceAccount; the API server audits the calls.

"Trusted provisioner holds a narrowed credential to the runtime API" is thus the *same pattern* on
both backends — the credential is backend config, not a hole in the abstraction.

### (g) Sandbox and verifier are symmetric ephemeral workers

Both are launched per-run via the `WorkerBackend`, distinguished only by their `role` label, and
driven over their respective ports (`SandboxPort` / `VerifierPort`). This honours the decision that
verifiers get the same per-run treatment as sandboxes. It is consistent with §3.6 ("the verifier is
**queue-fronted and async**"): a per-run verifier instance is one valid shape, and a shared
queue-fronted verifier *pool* is simply a backend that returns a handle to a pooled instance instead
of launching a fresh one — the abstraction accommodates both without the control plane knowing which.

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
| trust boundary | socket-proxy whitelist | namespaced RBAC `Role` |

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
  per-run worker the backend launches and isolates, which resolves the isolation regression that
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
  (per-run workers, labelled, slot-bounded) is correct from day one.
- **The regeneration invariant is strengthened**, not weakened: it becomes a structural property of
  worker disposal + reconciliation, asserted by a property test against every backend.
- **A trusted, audited socket replaces the anti-pattern:** privilege is concentrated in one
  least-privileged, logged choke point, not smeared across peers or nested daemons.
- **Cost:** a new provisioning layer (the `WorkerBackend`, the reconciliation loop, the GC reaper, the
  socket-proxy) that did not exist; the control plane grows a scheduler responsibility (still
  mechanical — reconcile, don't reason). The existing in-process and single-container paths remain
  valid as the trivial backend for dev/tests.

## Risks & open questions

- **Worker addressing & discovery.** The control plane must reach each launched worker over HTTP. On
  Docker this is per-worker host/port or an internal network + the backend returning the address in
  the handle; on k8s a Service/Pod IP. Addressing **by handle** (not service-name DNS) is the portable
  choice and is also what the optional gVisor runtime would *require* (it cannot use Docker's embedded
  DNS — google/gvisor#115); under the default plain runtime, normal Docker DNS is available but the
  handle is still the cleaner key. Needs a concrete addressing decision per backend.
- **CP restart semantics.** Reap-and-regenerate (cattle) vs re-adopt live workers (Nomad `RecoverTask`).
  The spec's "regenerated each cycle" leans reap-and-regenerate; `recover` is the fallback for a
  mid-flight cycle. Decide explicitly.
- **Verifier shape.** Per-run instance vs shared queue-fronted pool — both expressible; pick the
  default and confirm it against the cost of standing up a verifier worker per run.
- **Where the scheduler lives.** A loop inside the control-plane process vs a separable scheduler
  component. Inside is simpler now; the seam should not preclude extracting it.
- **State store under concurrency.** Many concurrent runs writing provenance pushes toward the Postgres
  backend (§4) sooner than single-user did; the `Store` interface already permits it.

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
Ryuk/reaper (<https://golang.testcontainers.org/features/garbage_collector/>); docker-socket-proxy
(<https://github.com/Tecnativa/docker-socket-proxy>); Temporal worker slots/pollers
(<https://docs.temporal.io/develop/worker-performance>).
