# Glossary — tenant, task, run, job

Verity's run-control vocabulary, with the maintainability rule that keeps the words from blurring:
**task ≠ run ≠ job.** This is the canonical reference; if code or a docstring drifts from it, fix the
drift.

## The four nouns

| Term | What it is | Identity | Kind |
|---|---|---|---|
| **Tenant** | The owner an operation runs under. One user/org today (`"default"`). | `tenant_id` | scope |
| **Task** | A **configuration** — *what to work on and how*. | `task_id` | spec (a noun) |
| **Run** | One **execution** of a task — the read→propose→gate→commit loop until the policy stops. | `run_id` | event (a verb) |
| **Job** | A run **enqueued for a worker** — the scheduling envelope around a run. | `job_id` | scheduling unit |

### Task — the specification
`TaskConfig` (`control_plane/config.py`), identified by `task_id`: instructions, the domain schema,
`sandbox_key` / `verifier_key`, the `OrchestrationPolicy`, `tenant_id`. `ControlPlane.configure(config)`
registers it; `TaskState` holds its live resolved services + history. The whole control-plane API is
keyed on `task_id` (spec §3.4, "keyed by a task identity"). A task is durable and re-runnable.

### Run — an execution of a task
`ControlPlane.run(task_id)` drives one run; `run_cycle` is one iteration within it; `RunReport`
projects what a run did (its outcomes, latencies, telemetry); `RunRecord` is the operational record of
a run. A run **belongs to a task** — `RunReport.task_id` and `RunRecord.task_id` name *which
configuration* it executed — but has its **own** identity, `run_id`.

### Job — a run, scheduled
`Job(job_id, tenant_id, run_id, payload)` + the `JobQueue` (`contracts/jobqueue.py`). The operational
layer *around* execution: a job is "this run, enqueued to be claimed and executed by a worker." A job
exists only in the multi-tenant run-control track; the in-process default just records job state.

## Relationships

```
tenant ─1:many→ task ─1:many→ run ─1:1→ job
                          run ─1:1→ RunReport
                          run ─1:1→ RunRecord
```

- A **task** is configured once and can be **run many times** (re-runs, retries) — so `task_id` is
  *not* a run identity.
- A **run** is identified by `run_id`, scoped within a `tenant_id`. Run-results are keyed
  `(tenant_id, run_id)`; `task_id` is a queryable attribute ("which runs belong to task X" —
  `RunRecordStore.list(tenant_id, task_id=...)`).
- A **job** schedules exactly one run; `JobStatus` (queued → claimed → done/failed) is the
  **scheduling** lifecycle — orthogonal to the **artifact** lifecycle (proposed → accepted / rejected
  / …). "Job done" is *not* "proposal accepted."

## The rule that prevents the blur

**`run_id` is never silently the `task_id`.** `RunRecord.from_report(report, *, run_id, status)`
*requires* an explicit `run_id` — the caller mints a run's identity rather than reusing the task's.
The trap this avoids: if `run_id` defaulted to `task_id`, a second run of the same task would collide
on `(tenant_id, run_id)` and a run-record store would silently overwrite the first.

## Current state (run-identity live; the multi-tenant *engine* still deferred)

**Run-identity now lives inside the loop** (ROADMAP 7.4.h). `ControlPlane.run()` mints a fresh `run_id`
per run (`run_id_source`, default `uuid4().hex`), binds a mutable `RunContext(tenant_id, run_id, cycle)`
that advances `cycle` per cycle, and emits a `RunRecord` keyed by `(tenant_id, run_id)` at run end —
so a backend-backed sandbox/verifier stamps `{harness, tenant, run, cycle, role, config}` onto every
worker it launches, and a second run of the same task gets its own id (no `(tenant_id, run_id)`
collision). `run_report()` is still keyed on `task_id` (the in-process engine runs one task's run at a
time), but task : run is now genuinely 1 : many in the records.

What remains deferred is the multi-tenant **engine**, not the identity: the real `JobQueue` + worker
model, per-tenant store isolation, and the standing job server (issue #27). Until that lands, `Job` is
a seam (the in-process default records job state synchronously) and there is one default tenant.
`RunRecord.from_report` still *requires* an explicit `run_id` — the control plane supplies its minted
`run_context.run_id`, never the `task_id`.
