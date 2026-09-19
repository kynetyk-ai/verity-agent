# Verity

Implementation of the **Self-Revising Discovery Harness** — a domain-agnostic kernel that
maintains an agent's state as a *typed provenance record* (artifacts, the operations that produced
them, and the gate decisions about them) rather than a chat transcript or a vector store, behind a
verifier-aware commit lifecycle. Shorthand: *git + a type system + a verifier-aware lifecycle for
agent artifacts.*

The name carries the point: **verity** = truth, and *verify* — the system exists to make what an
agent believes trustworthy and auditable.

## The spec is the source of truth

This repo implements a finalized natural-language specification, **vendored locally** so the repo is
self-contained (it may be worked on on a different machine):

```
spec/self-revising-discovery-harness.md      the spec (v0.2) — authoritative
spec/references/                              the grounding paper + prior-art analyses
```

Treat `spec/self-revising-discovery-harness.md` as authoritative. Implementation choices should trace
back to a spec section (cited as `§N`); if the code needs to diverge from the spec, raise it rather
than drifting. The `spec/` tree is a copy synced from the sibling spec repo (`NL-specs`,
`agents-and-harnesses/self-revising-discovery/`) — re-sync it if upstream changes.

## Architecture in brief

Three cooperating services (spec §3.3–§3.6):

- **Control plane** — the *only* service that mutates durable state; deliberately unintelligent. Owns
  the typed-provenance store, the commit path, the lifecycle, context assembly, the registries, the
  task config, the orchestration policy (loop control), and the invariant **workspace contract** +
  composed system prompt.
- **Sandbox** — the agent runtime + workspace where the loop runs and tools execute; ephemeral
  (regenerated each cycle); proposes but never writes; writes objects to the **outbox** for harvest.
- **Verifier** — advisory; hosts the gate plugins; takes a shaped proposal + a declared store-slice
  (+ object attachments) and returns a verdict; never writes. *(Per the spec, a separate service; since
  **Phase 9.1 / #73** it runs as a **sibling container** — its own image carries the gates + the
  `kaggle` extra, launched on demand and reached over the wire; the CP image carries no gate code.)*

The loop is **read → propose → gate → commit**. Load-bearing rules: **no implicit accept** (a type
with no declared gate cannot be committed), the proposer is never its own gate, status is richer than
accept/reject (`proposed → tentative → accepted`, plus `rejected` / `superseded` / `revised`), and
the `refine` verdict recovers a mostly-sound artifact with a localized defect.

## Roadmap and build order

The live plan is **[ROADMAP.md](ROADMAP.md)** — the canonical "where are we, what's next." Read the
lowest ⬜/🚧 item there before starting work, and keep it current (update it in the same PR that
changes the plan's reality).

The arc: **Phase 1** builds the control plane and proves it against bespoke test doubles (a stub agent
and a stub verifier in `tools/harness/`); **Phases 2–3** swap in the real verifier and sandbox;
**Phase 4** wires the feature-engineering domain (§12) and reaches MVP when the twelve §13 acceptance
criteria pass. This follows the spec's §17 build order (kernel contracts → extension interfaces →
context assembly → service scaffolding → domain → acceptance).

## Experiments: the whole experiment tree is out of this repo

Nothing under `experiments/`, `results/` or `figures/` is tracked here. All three are gitignored
outright. This repo is the harness; the experiments that use it, and everything they produce, are
versioned elsewhere.

- **`experiments/`** — the sweep machinery (`ablation/sweep.py`, `analyze.py`, `figures.py`,
  `run_sweep_container.sh`) and the `spec.*.json` catalog. Operator-local: keep it on disk, not in a
  commit here. `tests/test_ablation_*.py` `importorskip` it, so a checkout without it still runs
  green. The runnable experimental design it implements is versioned next to the analysis, at
  `../verity-analysis/docs/experimental-design.md`.
- **`results/`** — raw sweep output: RunReport JSONs, `manifest.json`, `transcripts/`, `submissions/`,
  and the durable `store/`. Always scratch. Transcripts record agent tool output verbatim, which
  includes rows of whatever dataset the agent read, so a result batch is **never** committed here:
  we hold access rights to the competition data and no right to redistribute it.
- **`figures/`** — analysis output, authored in `verity-analysis`.

**Do not `git add` any of these, and do not force past the gitignore.** If something in these trees
needs to be versioned, it belongs in `verity-analysis`, not here.

The **experiment data, the R/renv analysis, and the paper writing** live in the sibling
**`../verity-analysis`** repo (`data/<group>/`, notebooks + `scripts/`, `docs/paper-outline.md` +
`docs/related-work.md`).

### Keeping a finished batch

Import it into `verity-analysis`, passing the exact spec that produced it:

`cd ../verity-analysis && scripts/import_results.sh <batch-subdir> <group> ../verity/experiments/ablation/<spec>.json`

This `rsync`s the batch into `data/<group>/` **excluding `store/` and `sweep.log`** (only the
extracted artifacts — RunReports, transcripts, submissions — plus any figures are versioned; stores
are regenerable and never committed anywhere), **snapshots the producing spec into the batch dir as
`spec.json` + records this repo's git sha in `PROVENANCE.txt`** (so each run ties back to its exact
parameters), then you **add a row to `verity-analysis/data/REGISTRY.md`** and commit it in
`verity-analysis`. Always pass the spec arg.

Once a batch is imported and committed there, its `verity/results/<batch>/` (stores included) can be
deleted to reclaim disk. So: run → `results/` scratch here → `import_results.sh` into
`verity-analysis/data/` → add a `REGISTRY.md` row → commit there → delete the scratch.

## Coding habits

Non-negotiable working norms for this repo:

- **Python**, managed with **uv**. All dependencies and runs go through `uv` (`uv add`, `uv sync`,
  `uv run`). **No system-level or global `pip install`** — ever.
- **Don't defer basic infrastructure to reach an MVP.** The plumbing that makes a system debuggable
  and trustworthy is built from the start, not retrofitted:
  - **Structured logging from day one** — every service logs; no `print`-and-hope.
  - **Tests written alongside the code** — good coverage as we go, not bolted on at the end.
- **Always work on a branch.** Never commit directly to `main`; branch, then open a PR.
- **The unit of work for a sprint is a commit, not a PR.** Land each sprint as its own focused,
  green commit on the working branch; a single PR then carries several related sprints. Prefer
  **fewer, meaningful PRs** over one-PR-per-sprint churn. **Always align before issuing a PR** —
  confirm the scope and timing with the user rather than opening one unprompted.
- **Never open a PR on buggy or embarrassing code.** It runs, it's tested, and it's clean before it
  goes up for review. A PR is a finished thought, not a work-in-progress dump.
- **Keep the error-handling bar the hardening pass set (Phase 5.1).** These are now defaults, not
  one-off work:
  - **Typed errors wherever failure is predictable** — every service/IO boundary surfaces a
    domain-specific error, never a raw `OSError`/`KeyError` escaping to abort a run. Distinguish
    *recoverable* (degrade-don't-crash: record the failed cycle, feed it back, continue) from *misuse*
    (fail fast and loud).
  - **Retry transient external calls** with bounded backoff (the shared `retry_async` /
    `RetryPolicy`), and classify what's transient vs. fatal explicitly.
  - **"A failed step is recorded, not fatal"** stays an invariant — assert it with property tests, not
    just examples. Mutations stay atomic (commit-path `transaction()`).

## Status

**Post-MVP, deep into the Post-MVP roadmap.** Phases 0–5 are complete: the control plane, the verifier
service + gate-primitive SDK, the sandbox service (in-process + container Deep Agents), and the
feature-engineering domain (§12) that reached MVP against the twelve §13 acceptance criteria — plus the
Phase 5 reliability & observability hardening (degrade-don't-crash on both boundaries, typed errors,
retries, atomic commits, the store-derived `RunReport`).

**Phase 6 — model breadth (complete)** (the thesis payoff): **live-validated across ten models**,
hosted (`gpt-5.4-mini`/`nano`, `haiku-4-5`, `sonnet-4-6`) and local/open (`gpt-oss`, `gemma4`, `qwen`,
`granite`, `glm-4.7-flash`, `fugu`), over ablation batches run 2026-06-30 → 2026-07-16 and indexed in
`../verity-analysis/data/REGISTRY.md`.
**Phase 7 — service split & full containerization** reached its done-line (7.5): a **task-agnostic
control plane runs in a container** and launches ephemeral sandbox + code-runner **worker containers**
as siblings on the host daemon (ADR 0003), running the FE task by configuring the CP through its
API (the task catalog + per-task builders in `composition/`).

**Phase 8 — the long-lived control-plane service (v1 complete)** (ADR 0004): the CP now runs as a
**standing daemon** (`verity serve`) that multiplexes many tasks (per-task `ControlPlane` + durable
`SqliteStore`) behind one process, with a `verity` **CLI** as a thin client. Two control surfaces: a
local **Unix socket** (`docker exec verity-cp verity …` / `just cp …`, unauthenticated, local-only) and
an authenticated **network HTTP API** (`verity serve --http HOST:PORT` + bearer `VERITY_API_TOKEN`),
plus a **byte data plane** (raw `POST /objects`, `GET /artifacts/{run_id}/{path}`). Tasks are created,
run (async — poll `status`, read `results`, `export` artifacts), and survive daemon restart; no rebuild
to drive a new task. A **task catalog** self-describes each installed type's output-shape contract,
`verifier_approach`, and `sandbox_notes`. The feature-engineering task is **`fe-kaggle`**: a
goal-seeking ladder (cheap local hold-out proxy → a **competitive top-N% bar** read from the live
leaderboard, below which an attempt is *refined* — the gap fed back, no submission spent — rather than
submitted → a hard real **Kaggle-leaderboard** gate that accepts only what climbs our best public
score, with a self-calibrating proxy→public estimate; trusted-submitter so creds never reach a
worker), **live-validated** at 0.92253 balanced accuracy on `playground-series-s6e6` with local Qwen
(under the pre-competitive-bar config). (The earlier single-tier basic-`fe` task type was removed; its §12 domain — schema, shape,
scoring, and the standalone verifier exercised by `tests/test_feature_engineering_*.py` — is reused by
both `fe-kaggle` and `fe-holdout`. The installed catalog is `{code, fe-kaggle, fe-holdout}` — `fe-holdout`
is the ablation task: the same §12 FE domain scored on a **local reserved hold-out** (no Kaggle, no
competitive bar; the `holdout-experiment` verifier), driven by the operator-local `experiments/` tree. Prefer
`verity catalog` for the live set.) The three **acceptance modes** (optimizer / accumulate / first-acceptable) are documented as
emergent from verifier gate composition × `--stop-on-accept` × object-provisioning mode (README + the
`verity-run-task` skill).

**Phase 9 — the dumb control plane (complete)** (epic #27): the founding concept (§3.3–§3.6) that the
CP *dumbly* provisions containers and makes mechanical context updates is now true of the *image and
process*, not just the kernel. **#73 / 9.1** extracted the **verifier into a sibling container** (its
own image carries the gates + `kaggle`; the CP image dropped `--extra kaggle` + gate code), reached
over the wire via the `RemoteVerifier` seam. **#74 / 9.2** removed **task-specific data-prep** from the
CP ([ADR 0005](docs/adr/0005-data-prep-out-of-the-control-plane.md)): the user prepares per-role inputs
outside Verity and the CP routes opaque **role → {filename: blob}** maps (the split + answer-key
derivation moved to `tools/`), so the answer-key isolation invariant holds **by construction at the prep
boundary**. **9.3** built the **plugin loader** ([ADR 0006](docs/adr/0006-plugin-loader-entry-point-types.md)):
task **and** verifier types are **discovered** from the `verity.task_types` / `verity.verifier_types`
entry-point groups at boot (built-ins dogfooded through the same loader), not compiled in. Litmus test —
now met: a new verifier or task type ships **a container + an installed entry point, no CP rebuild**.

**Remaining / open tracks:** the multi-tenancy engine (real queue +
tenant isolation, issue #3); a `K8sBackend`; and three research-driven docs/feature tracks — making the
domain concept optional (#64), per-task harness-agnostic agent skills via the CLI (#65), and extender
docs for adding sandboxes & verifiers (#66). See [ROADMAP.md](ROADMAP.md) for the live plan and
[docs/api-surface.md](docs/api-surface.md) for the control-plane API reference.

**Stack: Python, managed with uv** (see *Coding habits*).
