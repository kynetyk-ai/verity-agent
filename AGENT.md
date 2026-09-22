# Verity

An ongoing experiment in how to build agentic systems that run unattended and produce trustable
results for knowledge work. The name comes from the same idea: **verity** = truth, and *verify*.

An ephemeral agent in a sandbox proposes; an isolated verifier the agent cannot reach judges the
proposal; a deterministic control plane records the decision and its reasons as durable, typed state
and decides what context the next sandbox sees. The kernel is domain-agnostic: a task supplies a
sandbox and a verifier, both of which are extension points. Feature engineering against a live Kaggle
leaderboard is the prototyping domain it was built and validated on. `README.md` carries the full
argument and the usage recipes.

## Where `§N` citations point

The design originated in a natural-language specification, which is why `§N` citations appear
throughout the code, the tests and the ADRs. That document is **not vendored here**; it lives in the
sibling `NL-specs` repo at `agents-and-harnesses/self-revising-discovery/`. Treat the citations as
design provenance, not as a live contract: where the build has moved past the spec, the build is
what's true. Keep existing citations when you edit nearby code, and do not add new ones.

## Architecture in brief

Three cooperating services (spec §3.3–§3.6):

- **Control plane** — the *only* service that mutates durable state; deliberately unintelligent. Owns
  the typed-provenance store, the commit path, the lifecycle, context assembly, the registries, the
  task config, the orchestration policy (loop control), and the invariant **workspace contract** +
  composed system prompt.
- **Sandbox** — the agent runtime + workspace where the loop runs and tools execute; ephemeral
  (regenerated each cycle); proposes but never writes; writes objects to the **outbox** for harvest.
- **Verifier** — advisory; hosts the gate plugins; takes a shaped proposal + a declared store-slice
  (+ object attachments) and returns a verdict; never writes. It runs as a **sibling container** whose
  own image carries the gates + the `kaggle` extra, launched on demand and reached over the wire; the
  control-plane image carries no gate code.

The loop is **read → propose → gate → commit**. Load-bearing rules: **no implicit accept** (a type
with no declared gate cannot be committed), the proposer is never its own gate, status is richer than
accept/reject (`proposed → tentative → accepted`, plus `rejected` / `superseded` / `revised`), and
the `refine` verdict recovers a mostly-sound artifact with a localized defect.

## Roadmap

**[ROADMAP.md](ROADMAP.md)** is the single home for status and outstanding work. Read the lowest open
item there before starting, and update it in the same PR that changes the plan's reality. Do not
restate phase status here or in the README; both point at the roadmap so it cannot drift out of sync
with itself.

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
