# Verity Roadmap

What is still outstanding, and where each open issue folds in. `README.md` and `CLAUDE.md` point
here for status rather than restating it.

**Phases 0 through 9 are complete.** What each one delivered is summarized under *What has been
built* below; the detail lives in git history and in the ADRs. Everything after that section is work
that has not been done.

---

## How to use this roadmap (hygiene)

- **The roadmap reflects reality, not intentions.** When a PR changes the plan's reality — a phase
  starts, a milestone lands, scope shifts — it **updates this file in the same PR**. A roadmap that
  has drifted from the code is worse than none.
- **Status legend:** ⬜ not started · 🚧 in progress · ✅ done.
- **One source of truth.** If you want to know what to build next, read the lowest ⬜/🚧 item here.
- **Completed phases are a record, not a map.** A landed sprint's entry describes the code as it was
  when it landed, so it may name files that have since been retired. Retired so far: the one-shot
  `composition/fe_run.py` entrypoint and `infra/compose.fe.yml` (removed with the basic-`fe` task,
  #68); `tools/benchmark_models.py` and `tests/test_fe_via_config_acceptance.py`; and the five
  per-image `Dockerfile.*` files, consolidated into one multi-stage `Dockerfile` with a target per
  image. For what exists now, read the tree or `README.md`, not a completed entry.

## Tracking off-roadmap items (use GitHub issues)

The roadmap tracks the **planned arc**. Everything else — anything that surfaces mid-stream but
doesn't need immediate resolution and **must not be lost** — goes into a **GitHub issue**, never a
buried `TODO` in the code or a line in someone's head. Deferred decisions, known limitations, a spec
ambiguity, a future refactor, a "we should revisit this": **open an issue.**

- Suggested labels: `deferred`, `tech-debt`, `spec-question`, `enhancement`.
- Link issues from the phase they bear on, so the arc and the backlog stay connected.
- The "Explicitly deferred past MVP" list below should each become an issue/epic when it comes alive.

## Cross-cutting principles (always on)

These hold in every phase (see `CLAUDE.md` → *Coding habits*):

- **uv** for everything; **structured logging from day one**; **tests alongside the code**.
- **Branch + PR** for all changes; **never a PR on buggy or embarrassing code.**
- **Container-per-service readiness** — the architecture keeps each service independently imageable
  (async ports, no shared mutable state). The **sandbox**, the **verifier** (its own image since
  Phase 9.1 / #73 — gates + `kaggle`, launched as a sibling), and the **control plane** each ship as a
  container. Since Phase 9.2 (#74 / ADR 0005) the CP image carries **no domain data-prep** either —
  data prep is user-side and the CP routes opaque role-keyed blobs.
- **Design traceability** — `§N` citations in the code and the ADRs point at the
  self-revising-discovery specification the design came from, kept in the sibling `NL-specs` repo.
- **Invariants are property tests**, not prose aspirations (spec §5).

---
## What has been built

Phases 0 through 9 are complete. One row each; the sprint-level record lives in git history and in
the ADRs, which are the durable account of why each shape was chosen.

| Phase | Delivered |
|---|---|
| **0 — Bootstrap** | Repo, uv project, CI, the working norms in `CLAUDE.md`. |
| **1 — Control plane** | The kernel: typed-provenance store, commit path, lifecycle, context assembly, registries, loop control, workspace contract. Proven against throwaway agent and verifier doubles. |
| **2 — Verifier service** | The opaque verifier and its gate-primitive SDK (deterministic check, numeric scorer, LLM judge, auto code runner, human-in-the-loop), staged cheap → `tentative`, hard → `accepted`. [ADR 0001](docs/adr/0001-opaque-verifier-and-control-plane-boundary.md) |
| **3 — Sandbox service** | The real agent runtime on Deep Agents, in-process and container drivers, the outbox harvest path. [ADR 0002](docs/adr/0002-sandbox-runtime-deep-agents.md) |
| **4 — Feature-engineering domain → MVP** | The FE domain end to end, reaching MVP against the twelve acceptance criteria as runnable checks. |
| **5 — Reliability & observability** | Degrade-don't-crash on both boundaries, typed errors, bounded-backoff retries, atomic commits, structured logging and telemetry, the store-derived `RunReport`. |
| **6 — Model breadth** | The provider-agnostic `ModelSpec` seam, live-validated across ten hosted and local models over batches run 2026-06-30 → 2026-07-16 (indexed in `verity-analysis/data/REGISTRY.md`). |
| **7 — Service split & containerization** | The wire codec and transport seam, then ephemeral per-cycle worker containers behind a backend-agnostic provisioning seam, ending with a task-agnostic control plane running in a container and launching sibling workers. [ADR 0003](docs/adr/0003-control-plane-and-ephemeral-worker-provisioning.md) |
| **8 — The standing daemon (v1)** | `verity serve` multiplexing many tasks behind one process, a `verity` CLI over a Unix socket, an authenticated network HTTP API with a byte data plane, durable across restart, with the exchange and store volumes provably off every worker. [ADR 0004](docs/adr/0004-long-lived-configurable-control-plane.md) |
| **9 — The dumb control plane** | The verifier extracted to a sibling container; task-specific data prep moved out of the CP ([ADR 0005](docs/adr/0005-data-prep-out-of-the-control-plane.md)); task and verifier types discovered from entry points at boot ([ADR 0006](docs/adr/0006-plugin-loader-entry-point-types.md)). A new type ships as a container plus an installed entry point, with no control-plane rebuild. |

Also landed outside the phase sequence: the run-scoped verifier lifecycle
([ADR 0007](docs/adr/0007-run-scoped-verifier-lifecycle.md)), which binds verifier provisioning to the
run rather than the task so idle verifiers stop accumulating.

---

## Outstanding

### Pre-ablation audit hardening (2026-06-29) 🚧

Remediation of `docs/audit/pre-ablation-audit.md` ahead of the ablation runs. **Landed:** restart-safe
score baseline (the verifier reads incumbents' durably-recorded scores from `VerifierRequest.scores`,
so a verifier restart can't collapse the bar to 0 and break monotonicity — G3); control-plane
degrade-don't-crash on every boundary (typed store-IO error, base-`TransportError`→`GateUnavailable`,
a `run_cycle` catch-all that records+regenerates, isolated child-harvest — S1–S3); verifier kernel
guards (no hard-less/empty pipeline → no silent accept G1; single authoritative supersession G2;
payload-key allowlist at intake so free-text can't ride to a gate R1); answer-key isolation backstop
(content-hash guard + agent `test.csv` target-strip — I1/I2); always-on **transcript** (stderr-log-reconstructed host-side)
instrumentation referenced from the RunReport (F); capped subprocess output (`verity.proc`, V3); and
the selection-margin noise-floor lever wired through the sweep (`Budgets.selection_margin`, C/V2).

- *Deferred (tracked here / file issues):*
  - **pip-freeze / resolved-env recording (V6).** Record the installed dependency set per run
    (auditable drift) rather than pinning a closed wheelhouse (which would reject legitimate
    submissions). Belongs with the code-runner config work below.
  - ~~**ε measurement + the `selection_margin` value (C).**~~ **Done 2026-07-05**
    (`verity-analysis/data/calibration_and_prototyping/ablation-gpt5mini/ablation-ladder-calibration/epsilon-calibration/`): 5× reruns on both an sklearn and a lgbm+xgb submission were
    bit-identical → ε = 0.0 on the study host; loop-rung `selection_margin` pinned at 0.0
    (`spec.loop.gpt5mini.*.json`).
  - **Post-audit addendum (2026-07-07):** the audit missed the *image* boundary — the sandbox image
    baked the repo (incl. the answer key) into `/app` (**#137**, fixed: dockerignore + purity
    guards; transcript audit found zero exploitation, but the 07-05..07 gpt-5.4-mini batches are
    re-designated smoke/calibration). The calibration transcripts also drove the
    **commodious-workspace affordance package** (**#136** pip seeded into the sandbox venv; **#138**
    workspace map names `scratch/provided/`+INDEX.md, outbox gloss directional, testing-regime
    split + runner budget stated per task; workspace contract v2, pins in
    `tests/test_prompt_freeze.py`). See experimental-design §7 for the validity record.
  - **fe-kaggle production should-fixes:** offline run-phase (network only for `pip`, V1) and optional
    single-thread determinism — out of the ablation path; the public-leaderboard exfil risk is real
    there. Part of the code-runner hardening item below.
  - **Nice-to-knows:** transcript+telemetry are now salvaged on a hard wall-clock kill (F4 — shipped); reap orphaned
    code-runner containers if the verifier crashes mid-run (V5); a two-directional bundle-consistency
    assertion (G5); bound the kaggle `_run_cache` growth (G7).

### Woven through the build

- **Code-runner hardening / generalization** — a declarative runner config (image, network policy,
  resource + output caps, mount layout, deps strategy), an offline / pinned-wheels install option to
  reclaim reproducibility, and a cleaner validity-rung vs scoring-rung split. Revisits the accepted
  network-on tradeoff. *(The output cap — V3 — and the network/deps items above intersect this; fold
  the deferred V1/V6 work in here.)*
- **Configuration guide + a Claude Agent Skill for setting up the system** — now that the configuration
  surface has stabilized through Phase 6, make standing up a new task / domain / model low-friction.
  (a) A **configuration guide** in `docs/` covering the whole surface: building a domain (schema +
  gate pipeline + shape validator + harvester), registering a sandbox via `ModelSpec` /
  `local_spec` / `openai_spec`, registering a verifier + code runner, assembling a `TaskConfig`, the
  `OrchestrationPolicy` (cycles / refine cap / stop-on-accept), and mounting data via
  `static_contents` vs `data_sources` (the §13.12 isolation note). (b) A **Claude Agent Skill** that
  scaffolds that configuration interactively — gathers the task/domain/model intent and emits a
  working registration + `TaskConfig` (and a benchmark arm), so a user (or Claude) can wire a new
  domain or model arm without reverse-engineering the test setups. The live-validation work
  (`docs/local-models.md`, `tools/benchmark_models.py`) is the seed material.

### Multi-tenancy & run-control

The control plane grows into a multi-tenant **job server**. The typed-provenance store stays the
system of record (artifacts / operations / decisions / objects) — this adds the *operational* layer
*around* it, not a second copy of it. Built **seams-first, engine-later**, so multi-tenancy is an
added adapter, not a reshape.

- **Run-control seams (with Phase 5).** A `JobQueue` port (`enqueue` / `claim` / `complete` /
  `status`) with a trivial **in-process default** (run synchronously, as today); a `RunRecord` /
  run-results store keyed by `(tenant_id, run_id)` that holds the **5.3 RunReport** + run status +
  *pointers* to the accepted artifacts (operational metadata referencing the provenance store, **not**
  a copy of it — the RunReport read-side and this store are the same thing); and a `tenant_id`
  threaded through the API / `TaskConfig` while there is still only one tenant.
- **Multi-tenancy engine (with Phase 7).** The real queue + worker model — commits stay **serialized
  per tenant** so the sole-mutator audit guarantee is preserved; the **store-engine upgrade** the
  queue forces (SQLite is single-writer → Postgres, or per-tenant DBs) — the natural moment for
  **object retention / GC** of bytes referenced only by terminal artifacts (**#9**, §5.5); **tenant
  isolation** (namespaced stores; no cross-tenant reads in retrieval / slices / object-provisioning) —
  the actual hard part; and the async **submit → job_id → poll/fetch** API a standing, networked
  control plane wants (the re-homed **#3** data plane). Naturally siblings with the service split
  (7.1–7.4). **This is the home for ADR 0003's deferred concurrency engine:** the level-triggered
  reconciliation loop + per-tenant concurrency slots/fairness (ADR c/h), and the commit-under-concurrency
  correctness (re-read incumbents inside the commit `transaction()`, ADR j). The `WorkerBackend` (7.4)
  is the seam it drives; a **`K8sBackend`** is the second impl of that port for a cluster deployment.
- **Phase 8 takes the first engine increment.** The long-lived control-plane service (ADR 0004) wires
  the `JobQueue`/`RunRecord` seams into a *standing daemon* (serial, single default tenant) and
  activates the #3 control half. It already isolates **per task** (a store per task instance, 8.1);
  the residual **cross-tenant** isolation (namespaced / Postgres stores, no cross-tenant reads) +
  concurrency stay this deferred engine (#58).
- **Not here:** authentication / authorization (still *further out*); parallel agents against one task
  (a related but separate concurrency concern, below).

### Open issues → roadmap home

Where each open issue folds in, so the backlog and the plan stay linked. Regenerated from
`gh issue list --state open` on 2026-09-19; re-derive it rather than editing rows by hand.

**Delivered on `develop`, still open on GitHub.** A PR merged to `develop` does not fire a `Closes`
keyword, which only triggers on the default branch. These fourteen are fixed in the code and will
close when `develop` merges to `main`. Do not plan work against them.

#32, #96, #102, #103, #125, #126, #129, #132, #133, #134, #136, #137, #138, #142.

**Agent-loop reliability**

| Issue | Folds into |
| --- | --- |
| **#141** tool-call parse recovery is Ollama-specific (vLLM / llama.cpp still crash) | the residual of the #126 fix |
| **#144** wire the image-purity test into the build path | the enforcement half of the #137 fix |
| **#69** harden outbox harvest to the declared proposal objects | sandbox hardening, standalone |
| **#125** synthesized stop-reason on hard timeout | *delivered, see above* |

**Task & verifier extensibility (epic #97)**

| Issue | Folds into |
| --- | --- |
| **#97** declarative / manifest-driven task types | the umbrella: a generic loader, no per-task Python |
| **#98** expose the task-type authoring contract (the four seams) | the docs slice of #97 |
| **#99** regulated hot-drop / runtime registration in the daemon | gated on #97 |
| **#64** make the domain concept optional | a generic default domain + verifier criteria in the prompt |
| **#65** per-task agent skills in the sandbox (harness-agnostic) | sandbox extensibility |
| **#66** extender's guide for sandboxes and verifiers | the configuration-guide item above |
| **#42** extensible stop/continuation predicate | acceptance-modes extensibility on `OrchestrationPolicy` |
| **#106** make the acceptance dimension config-driven | the deferred half of decoupling it from provisioning |

**Multi-tenancy & run-control engine (epic #27)**

| Issue | Folds into |
| --- | --- |
| **#27** the epic | the Multi-tenancy & run-control track above |
| **#3** standing control-plane data plane | the over-the-wire file API (removes the exchange volume) |
| **#58** store-isolation tiers | per-task is done; the cross-tenant residual is the engine |
| **#57** crash recovery for in-flight runs | durable run-status, with the engine |
| **#9** object retention / GC | the engine, with the store-engine upgrade |
| **#77** graduate verifier launch to the full fleet lifecycle | the verifier half of the same lifecycle work |
| **#96** run-scoped verifier teardown | *delivered, see above* |

**Verifier & gate quality**

| Issue | Folds into |
| --- | --- |
| **#117** harden the fe-kaggle verifier (exfil, calibration restart-safety, cache growth) | the code-runner hardening item above |
| **#11** LLM-judge reproducibility | when a judge enters a live gate stack |
| **#79** seed the fe-kaggle incumbent from the live leaderboard | fe-kaggle gate work |
| **#80** steer agents off `id` as a feature | FE domain instructions |

**Experiment-driven**

| Issue | Folds into |
| --- | --- |
| **#112** separate fails-to-run rate from balanced accuracy | the model + instruction selection protocol |
| **#113** cheap-gate reject feedback: a "test your code first" steer | agent feedback quality |

**Model hosting & housekeeping**

| Issue | Folds into |
| --- | --- |
| **#40** revisit Docker Model Runner as the managed local-model host | the model-server hardening track |
| **#5** spec sync: collapse the §8.2 tool registry | already true in code; sync back to NL-specs |


### Further out / seamed

Designed-for, not yet scheduled; each becomes an issue/epic when its time comes:

- Authentication / authorization on the control plane (pairs with the multi-tenancy track above).
- Parallel agents against one task (diversity / race — N sandboxes → 1 control plane). The 7.4
  worker model + labels (`run`, `cycle`) are the placed seam (ADR 0003); building it = the concurrency
  engine in the Multi-tenancy track above.
- Over-the-wire **file ingestion/egress API** — removes Phase 8's shared exchange volume (ADR 0004 (f)
  seam / the #3 networked data plane).
- Spec sync: collapse the §8.2 tool registry into operation signatures (**#5**) — already true in code.
- Secrets / egress policy once local + networked services land.
- True schema migration / regime transition (spec §15).
- The soft-gate consulting domain (spec §14).
