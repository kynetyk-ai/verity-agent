# First end-to-end `fe-kaggle` run — full data, gpt-5.4 (2026-06-21)

The first genuine end-to-end run of the `fe-kaggle` task on the hardened infrastructure: a hosted
frontier model writing real ML pipelines, each scored by the **live Kaggle leaderboard gate**, over
ten read → propose → gate → commit cycles on the full competition dataset.

This folder is a recording of that run — the data we observed and the agent's nine submissions, kept
together so the progression is legible.

| | |
|---|---|
| run_id | `488a4fb6e7084beeaf4b9600a551b9c8` |
| task_id | `7f9fa8e0e5e649b5930f41be5e8f6714` |
| date | 2026-06-21 |
| model | `gpt-5.4-2026-03-05` (hosted OpenAI) |
| competition | `playground-series-s6e6` (stellar classification, balanced accuracy) |
| data | **full** — train 577,347 rows · test 247,435 · hold-out 86,601 (15% stratified) |
| config | [`../../task.gpt.json`](../../task.gpt.json) (`max_cycles=10`, `stop_on_accept=false`) |
| totals | 10 cycles · ~120.5 min wall · 3,349,832 tokens |
| outcomes | 6 revised · 3 rejected · 1 sandbox-failed · **0 accepted / 0 Kaggle submissions** |

## Infrastructure exercised

Everything below ran for real, end to end, for the first time in one run (the fixes that made each
possible landed this session, PRs #83–#86):

- **Full-data training** — the agent trained on all 490,746 agent-train rows (not a subsample).
- **16-core code-runner** — `VERITY_CODE_CPUS=16` (the worker default of 1 CPU pinned training to a
  single core before this).
- **Real full-leaderboard bar** — the competitive gate read the **whole 2,155-team** leaderboard
  (`competition_leaderboard_download`) and set the top-10% target at **0.9714** (the paginated
  `_view` had returned only the top ~20, giving a near-podium 0.9725).
- **16 GB** code-runner memory; **4 GB / 3 h** verifier sibling (memory + dispatch timeout).
- **libgomp ML stack** — XGBoost, LightGBM, and CatBoost all importable in the runner image.
- **Remote verifier sibling** over the wire; **degrade-don't-crash** on every failure below.

## What we observed

| cyc | outcome | proxy bal-acc | wall | tokens | submission |
|---:|---|---:|---:|---:|---|
| 0 | revised | **0.9658** | 1m51s | 287,603 | single full-data XGBoost + color/redshift/sky FE, balanced weights |
| 1 | sandbox-failed | — | 1m08s | — | (no proposal — OpenAI `invalid_prompt`) |
| 2 | revised | **0.9661** | 1m52s | 246,356 | 2-model XGBoost average, explicit preprocessing |
| 3 | rejected | — | 40m50s | 566,633 | XGBoost + CatBoost (CatBoost couldn't create its working dir) |
| 4 | revised | **0.9659** | 3m27s | 551,443 | 3-model XGBoost soft-vote (CatBoost removed) |
| 5 | rejected | 0.9586 | 14m58s | 386,124 | 3-model XGBoost + ordinal encoding + hand-set class weights |
| 6 | revised | **0.9661** | 3m20s | 255,335 | XGBoost + LightGBM soft-vote |
| 7 | revised | **0.9664** | 18m26s | 363,446 | **stacked** — 2 XGB + 2 LGBM → out-of-fold → LogisticRegression meta |
| 8 | revised | **0.9661** | 19m34s | 420,368 | the cycle-7 stack + OOF-tuned per-class probability scaling |
| 9 | rejected | — | 15m04s | 272,524 | 6-model stack (2 XGB + 2 LGBM + 2 CatBoost) + tuned blend (hung) |

Trajectory of the scored cycles:

```
0.9658 → ✗ → 0.9661 → ✗ → 0.9659 → ✗(0.9586) → 0.9661 → 0.9664 → 0.9661 → ✗
```

Best hold-out balanced accuracy: **0.9664** (cycle 7). Competitive bar: **0.9714**. No cycle's
estimate reached the bar, so the gate spent **zero** Kaggle submissions across the run.

## The agent learned and adapted across cycles

The agent visibly changed its approach in response to the harness's reject reasons and refine
feedback. Each step below quotes the agent's own recorded rationale (see `results.json`) next to the
signal it was answering:

- **Recovered from a runtime crash, then re-integrated the fix.** Cycle 3's XGBoost + CatBoost
  submission was rejected (`runs-clean`: CatBoost could not create its `catboost_info` working
  directory in the locked-down runner). Cycle 4 — *"removing the failing CatBoost dependency path
  entirely… directly addresses the prior runtime rejection while preserving a competitive
  boosted-tree approach"* — dropped CatBoost. By cycle 9 it had **re-added CatBoost correctly**
  (`allow_writing_files=False`).
- **Replaced a guess with an optimization after a regression.** Cycle 5 hand-set class weights
  (`{QSO 1.25, STAR 1.1}`) and the hold-out score went *down* to 0.9586 (it was then retired at the
  refine cap). Cycle 8 returned to the same idea done properly — *"added OOF-derived class-specific
  probability scaling tuned directly for balanced accuracy"* — optimizing the per-class scaling on
  out-of-fold predictions instead of guessing.
- **Escalated from voting to stacking.** Cycles 0–6 were fixed-weight averages/votes of boosted
  trees. Cycle 7 — *"replaced fixed-weight argmax voting with 4-fold out-of-fold stacking via
  multinomial logistic regression blended with base-average probabilities"* — moved to a proper
  stacked ensemble, which produced the run's best score.

Across the run the agent accumulated fixes and reached for progressively more standard ensembling
techniques, driven by the per-cycle feedback the harness fed back.

## Failure modes observed (and how the harness handled each)

| cyc | failure | how the harness handled it |
|---:|---|---|
| 1 | OpenAI `invalid_prompt` (content-filter flag on a refine cycle) | recorded as a failed cycle; **the run continued** (degrade-don't-crash) |
| 3 | CatBoost `Can't create train working dir: catboost_info` | `runs-clean` reject; the reason was fed back, and the agent fixed it |
| 5 | hand-set class weights regressed to 0.9586; lineage hit the `refine_cap` (3) | terminated the refine lineage as `rejected`; the agent started a fresh lineage |
| 9 | training hung at 0% CPU after ~11 min, never produced output | the run-record was killed manually; the runner's 60-min timeout was the standing safety net |

On the cycle-9 hang, the **evidence** in this run: the identical out-of-fold stacking pipeline
completed successfully **twice** with XGBoost + LightGBM (cycle 7, 18m26s; cycle 8, 19m34s; both
exit 0). Cycle 9 was the same pipeline **plus CatBoost**, and it froze at 0% CPU with memory steady
at ~4.45 GiB (no crash, no OOM). *Hypothesis* (not confirmed in this run): running three libraries
that each ship their own OpenMP/threading runtime in one process can deadlock — left to verify with
a thread dump.

## The submissions

The nine scripts the agent actually wrote, as the gate ran them (cycle 1 produced none):

| script | cycle | result |
|---|---:|---|
| `submissions/cycle0-submission.py` | 0 | revised · 0.9658 |
| `submissions/cycle2-submission.py` | 2 | revised · 0.9661 |
| `submissions/cycle3-submission.py` | 3 | rejected · CatBoost FS crash |
| `submissions/cycle4-submission.py` | 4 | revised · 0.9659 |
| `submissions/cycle5-submission.py` | 5 | rejected · 0.9586 + refine-cap |
| `submissions/cycle6-submission.py` | 6 | revised · 0.9661 |
| `submissions/cycle7-submission.py` | 7 | revised · **0.9664 (best)** |
| `submissions/cycle8-submission.py` | 8 | revised · 0.9661 |
| `submissions/cycle9-submission.py` | 9 | rejected · hung |

`results.json` is the full machine-readable RunReport (every cycle's decisions, scores, timings, and
the agent's rationales).

## Reproduce

Per [`../../PROTOCOL.md`](../../PROTOCOL.md): bring up the daemon, then
`./prototyping_datasci_test/run.sh prototyping_datasci_test/task.gpt.json` (full data, hosted model).
Runs are non-deterministic (hosted model + live leaderboard).
