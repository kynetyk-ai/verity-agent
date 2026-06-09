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

## Current state (seams-first, engine-later)

The control-plane engine is still **1 : 1 task ↔ run** today: `run()` / `run_report()` are keyed on
`task_id` and there is no run-identity inside the loop. The distinct `run_id` lives in the
**multi-tenancy seams** (`RunRecord`, `Job`). When the run-control *engine* lands (a standing job
server — issue #27), `ControlPlane.run()` mints a real `run_id` and keys run history by it, making
task : run genuinely 1 : many in the engine, not just the seams. Until then, callers that persist a
`RunRecord` supply the `run_id`.
