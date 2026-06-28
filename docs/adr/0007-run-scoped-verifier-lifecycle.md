# ADR 0007 — Run-scoped verifier provisioning lifecycle

- **Status:** Accepted — implemented in PR #105 (issue #96).
- **Date:** 2026-06-27
- **Affects spec:** §3.4 (the lifecycle surface the control plane drives), §3.6 (the opaque verifier),
  §3.9 (deployment / resource hygiene). Builds on
  **[ADR 0004](0004-long-lived-configurable-control-plane.md)** (the standing daemon — *resident*
  tasks) and the §9.1 verifier-as-sibling extraction (#73), which together created the leak this
  records the fix for.
- **Supersedes:** nothing. Refines the verifier half of the lifecycle surface ADR 0003 §g already
  scopes to the run at the *worker-backend* level — this records the matching **control-plane API**
  shift.

## Context

Since §9.1 (#73) the trusted verifier runs as a **sibling container** launched by the
`WorkerBackend`. It was provisioned in `ControlPlane.configure()` and torn down only in
`ControlPlane.teardown()` — i.e. bound to the **task** lifecycle.

That was fine when a process ran one task and exited. Under the standing daemon (ADR 0004), tasks are
**resident**: a configured `ControlPlane` survives across runs and is never torn down for the
process's life, and — because each `run.sh`/`create` call makes a *new* task — idle verifier
containers accumulated **one per task ever run**, each holding its full `VERITY_VERIFIER_MEMORY`
(4 GB default). Observed live: a verifier left running ~9 h past its run, two 4 GB verifiers up at
once. This is a direct blocker for the multi-tenancy engine (#3), where verifier count must be bounded
by *active* work, not history.

The sandbox/code-runner workers never leaked — they are launched `--rm` and destroyed per cycle. Only
the verifier's teardown was deferred to task lifecycle.

## Decision

**The verifier's provision/teardown are bound to the RUN, not the task.** `ControlPlane.run()`
provisions the task's verifier at run-start and tears it down in a `finally` at run-end (so it is
reaped even on abort/exception); `configure()` no longer provisions the verifier (it still *creates*
and run-context-binds it). `teardown()` keeps an idempotent `verifier.teardown()` as a backstop.

Mechanically, the eager launch-at-configure is replaced by a thin **`_LazyLaunchVerifier`** adapter
(`composition/fe_kaggle.py`) that holds the launch factory + the per-task `VerifierSetup` and
(re)builds a fresh inner `VerifierPort` on each `provision()` (launch sibling → ship setup → await
healthy), destroying it on each `teardown()`. It is re-launchable across a resident task's runs.

This composes with the existing `VerifierPort.provision`/`teardown` contract (§3.4): in-process
verifiers (the fake, the tests) implement them as no-ops, so the shift is **behavior-preserving for
non-sibling verifiers** and the change is uniform across verifier types. Runs are serialized per task
by the daemon's run lock, so there is no intra-task provision/teardown overlap.

## Consequences

- **Bounded resource use:** steady-state running verifiers ≈ **active runs**, not tasks-ever-created.
  This is the acceptance bar for #96 and a prerequisite for #3.
- **Cost:** a verifier is re-launched and re-shipped its setup payload (~230 MB of CSV for full-data
  `fe-kaggle`) **per run** — the warm-verifier-across-runs optimization is given up. Irrelevant under
  the `run.sh`=new-task pattern (a fresh task gets a fresh verifier regardless); the removal of the
  over-the-wire bulk transfer is tracked separately (the networked-data-plane track).
- **API contract change:** callers must not assume the verifier is live after `configure()` — it is
  live only during `run()`. Reflected in [docs/api-surface.md](../api-surface.md) (`configure` /
  `teardown` / `run`).
- **Alternative considered:** an idle-TTL reaper that keeps the task-scoped verifier warm and reaps it
  after a no-active-run TTL. Rejected as the default — more moving parts in the long-lived daemon
  (a background loop + last-activity tracking) for a warmth benefit the dominant usage doesn't need.
  Run-scoping also matches the spec's "ephemeral, regenerated" spirit for workers.
