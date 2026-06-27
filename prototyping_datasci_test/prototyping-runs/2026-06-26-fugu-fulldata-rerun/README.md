# End-to-end `fe-kaggle` run — full data, Sakana **fugu**, raised time budget (2026-06-26)

A **re-run of the Sakana `fugu` full-data run** ([2026-06-23](../2026-06-23-fugu-fulldata/README.md))
with two changes: a **funded API key** (the prior run aborted at cycle 8/10 on credit exhaustion) and
a **raised agent time budget** — `sandbox_timeout_s` **2400 → 4800** (40 → 80 min), `code_timeout_s`
3600 → 4800, `recursion_limit` default → 300, and the verifier HTTP dispatch timeout
`VERITY_VERIFIER_TIMEOUT` 10800 → 14400. The prior run lost cycles 1 and 3 to the 40-minute wall; this
run set out to see whether more headroom converts those into scored proposals.

It did — and then some. **All ten cycles completed** (no external abort), **five of the eight scored
cycles needed more than 40 minutes** and so would have died under the old wall, and the run still ended
with the same disciplined outcome: a flat ~0.966 plateau, **zero Kaggle submissions spent**, because no
attempt's calibrated estimate reached the live top-10% bar.

| | |
|---|---|
| run_id | `656b3b11a7e54a338c747be26764d8cf` |
| task_id | `b3e00f1ce65b44dc81cce7ef2edafd2e` |
| date | 2026-06-26 → 2026-06-27 |
| model | `fugu` (Sakana, OpenAI-compatible at `api.sakana.ai/v1`; key via `OPENAI_API_KEY`) |
| competition | `playground-series-s6e6` (stellar classification, balanced accuracy) |
| data | **full** — train 577,347 · test 247,435 · agent-train 490,746 · hold-out 86,601 (15% stratified) |
| config | [`task.fugu.json`](../../task.fugu.json) — `max_cycles=10`, `stop_on_accept=false`, **`sandbox_timeout_s=4800`**, `code_timeout_s=4800`, `recursion_limit=300` |
| totals | 10 cycles · **~9 h 46 m wall** · **9,507,835 tokens** (9.24M in · 0.27M out) · 370 model steps · 433 tool calls |
| outcomes | **8 revised · 2 sandbox-failed (both timeouts)** · **0 accepted / 0 Kaggle submissions** |
| best hold-out | **0.9666** (cycle 4) · live competitive bar **0.9718–0.9719** · gap never below **0.0053** |

## What changed vs. the 2026-06-23 run, and why

The prior fugu run lost two cycles to the **2400 s (40-min) `sandbox_timeout_s`** wall (a full-data
LightGBM/OpenMP stall) and then aborted at cycle 8 when the Sakana account's prepaid credit ran out.
This run funded the key and **doubled the agent's per-cycle budget to 4800 s (80 min)** (README of the
prior run recommended exactly this). The verifier-side `code_timeout_s` was raised to match, and the
CP→verifier HTTP dispatch timeout was lifted to 14400 s to clear `2 × code_timeout_s` + the Kaggle
wait. `recursion_limit` was set to 300 (the agent never came close — it used ≤70 model steps/cycle).

## What we observed

Per-cycle, from the machine-readable [`results.json`](results.json) (cycles 0-indexed):

| cyc | outcome | proxy bal-acc | bar | gap | wall | tokens | steps | models / what changed |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | revised | 0.9660 | 0.9718 | 0.0059 | 42.2 m | 847,199 | 35 | XGBoost + sklearn HGB, OOF-tuned multipliers |
| 1 | revised | 0.9662 | 0.9719 | 0.0057 | 76.5 m | 809,462 | 42 | **+ LightGBM** (3 families) + target encoding |
| 2 | revised | 0.9663 | 0.9719 | 0.0056 | 70.1 m | 1,124,077 | 48 | richer colour×redshift interactions |
| 3 | revised | 0.9663 | 0.9719 | 0.0056 | 76.8 m | 1,407,776 | 51 | more features / combo categoricals |
| 4 | revised | **0.9666** | 0.9719 | 0.0053 | 71.3 m | 1,472,199 | 51 | **best** — 2×LGB + 2×XGB + HGB weighted blend |
| 5 | **timed out** | — | — | — | 80.1 m | — | — | (no proposal — multi-harness tuning hit the wall) |
| 6 | revised | 0.9665 | 0.9719 | 0.0054 | **9.5 m** | 838,420 | 37 | **LightGBM-only; self-calibrating multipliers baked into the script** |
| 7 | **timed out** | — | — | — | 80.0 m | — | — | (no proposal — K-fold OOF *stacking* experiment hit the wall) |
| 8 | revised | 0.9665 | 0.9719 | 0.0054 | **10.4 m** | 788,292 | 36 | **root-cause recovery** — bounded threads, vectorized target-encoding |
| 9 | revised | 0.9661 | 0.9719 | 0.0057 | 69.4 m | 2,220,410 | 70 | final — robust 2-model, fixed multipliers (XGB re-added) |

Trajectory of the scored cycles:

```
0.9660 → 0.9662 → 0.9663 → 0.9663 → 0.9666 → ✗timeout → 0.9665 → ✗timeout → 0.9665 → 0.9661
```

Best hold-out balanced accuracy **0.9666** (cycle 4). The live top-10% bar sat at **0.9718–0.9719**
(read fresh each cycle; it drifted up ~0.0001 over the run). The gap stayed **~0.0053–0.0059**
throughout — the gate spent **zero** Kaggle submissions, exactly as designed.

## The headline: the raised time budget did its job

This run is the clean before/after on `sandbox_timeout_s`:

- **Five of the eight scored cycles ran longer than the old 40-min wall** (cycles 1, 2, 3, 4 at
  70–77 min, plus the two timeouts at 80 min). Under the prior `2400 s` config, cycles 1–4 would all
  have been **killed with no proposal** — the same loss that wiped out two cycles last run. With 80
  minutes they each produced a scored attempt, including the run's best (cycle 4).
- **But 80 minutes is not unlimited headroom.** Two cycles (5, 7) *still* timed out — both on the
  agent's most elaborate experiments (a multi-harness multiplier search; a K-fold OOF *stacking*
  bake-off on a 200k subsample). More time helped, but **the agent does not reliably self-budget**:
  faced with the wall mid-experiment, the soft 80%-of-budget nudge can't land while it's inside a long
  training call, and the hard kill is the backstop. Both were handled cleanly (record `sandbox_failed`,
  feed back, continue) — never close to the 3-consecutive-failure abort, since a successful cycle sat
  between them.

## The agent matured its own ML methodology across the run — with no human guidance

The most interesting story is an autonomous climb in *methodology*, steered only by the gate's
refine-with-gap feedback:

1. **Guessed → tuned multipliers, more model families** (cycles 0–4): from XGBoost+HGB to a
   2×LGB+2×XGB+HGB weighted blend with redshift-binned **target encoding**; the per-class
   balanced-accuracy multipliers went from hardcoded to OOF-tuned. This carried it to the run's best,
   0.9666 — but the gains were sub-thousandth and plateauing.
2. **Self-calibration promoted into the artifact** (cycle 6): after a timeout, it *simplified*
   (LightGBM-only) and **baked the multiplier tuning into the submission itself** — a bounded
   coordinate-descent on an internal hold-out, so the script self-calibrates its decision rule. Ran in
   **9.5 min** and matched the plateau (0.9665).
3. **Stacking** (cycle 7): it then tried proper K-fold OOF stacking (a logistic-regression
   meta-learner over three diverse LightGBM variants), with multiplier tuning done *correctly* —
   tuned on OOF, validated on a held-out split. This experiment **ran out the 80-min clock** (timeout).
4. **Root-cause recovery** (cycle 8): after the second timeout it performed real **failure analysis** —
   its own comments name the causes ("many OpenMP threads can thrash"; "the row-by-row target-encoding
   python loop"; "an extra calibration fit") — and engineered them out (bound all four thread pools,
   vectorize the target encoding, drop the in-script calibration fit). It ran in **10.4 min** and
   scored 0.9665.
5. **Strategic retreat under deadline** (cycle 9): the final cycle banked a robust 2-model average with
   *fixed* multipliers — trading the plateau-breaking methods (self-calibration, stacking) for
   certainty of finishing. It scored **0.9661**, slightly below the best.

The arc — *hardcoded → tuned → self-calibrating artifact → OOF stacking → root-cause recovery → safe
retreat* — is a concrete illustration of the loop working: the agent got more sophisticated **and** more
rigorous about validation on its own, but a hard wall-clock budget pushes it to sacrifice its best
methods for robustness. (`requirements.txt` tracks the model-family moves: XGB+HGB → +LightGBM →
LightGBM-only at cycles 6/8 → LightGBM+XGBoost again at cycle 9.)

## Failure modes observed (and how the harness handled each)

| cyc | failure | how the harness handled it |
|---:|---|---|
| 5 | sandbox worker hit the 4800 s wall mid-experiment (no proposal) | recorded `sandbox_failed`, reason fed back; run continued (degrade-don't-crash) |
| 7 | sandbox worker hit the 4800 s wall during the OOF-stacking experiment | recorded `sandbox_failed`, fed back; the agent then performed root-cause analysis in cycle 8 |

Both timeouts were **non-consecutive** (cycle 6 succeeded between them), so
`max_consecutive_sandbox_failures=3` never triggered. The soft `DeadlineMiddleware` nudge (80% of the
budget = 64 min) fires only *between model steps*, so it cannot land while the agent is inside a long
full-data training/CV call — the hard kill is the backstop.

## A note on the incumbent (the ratchet caveat, observed live)

Provisioning is `last_revised_or_accepted` — **newest by recency, not highest-scoring**. Because nothing
was ever `accepted`, the carried-forward incumbent could drift *below* the best seen: cycle 6's 0.9665
became the base over cycle 4's higher 0.9666, and the run ended on cycle 9's 0.9661. The "never
regresses" ratchet is a property of the `accepted` lifecycle, not an all-`revised` run — report
best-so-far as the `max` over scored cycles (0.9666), not "the incumbent."

## The submissions

The eight scripts fugu actually wrote, as the gate ran them (cycles 5 and 7 produced none):

| script | cyc | proxy | note |
|---|---:|---:|---|
| `submissions/cycle0-submission.py` | 0 | 0.9660 | XGBoost + sklearn HGB |
| `submissions/cycle1-submission.py` | 1 | 0.9662 | + LightGBM, target encoding |
| `submissions/cycle2-submission.py` | 2 | 0.9663 | richer interactions |
| `submissions/cycle3-submission.py` | 3 | 0.9663 | + combo categoricals |
| `submissions/cycle4-submission.py` | 4 | **0.9666 (best)** | 2×LGB + 2×XGB + HGB blend |
| `submissions/cycle6-submission.py` | 6 | 0.9665 | self-calibrating multipliers in-script |
| `submissions/cycle8-submission.py` | 8 | 0.9665 | bounded threads + vectorized target-encoding |
| `submissions/cycle9-submission.py` | 9 | 0.9661 | final robust 2-model |

Each cycle's `requirements.txt` is alongside its script. [`results.json`](results.json) is the full
machine-readable RunReport (every cycle's gate decisions, proxy scores, competitive estimates, timings,
and per-cycle telemetry).

## Reproduce

Per [`../../PROTOCOL.md`](../../PROTOCOL.md): bring up the daemon with a **funded** `OPENAI_API_KEY` for
Sakana in `.env` (and the raised `VERITY_VERIFIER_TIMEOUT=14400` default, already in
`infra/compose.daemon.yml`), then `./prototyping_datasci_test/run.sh prototyping_datasci_test/task.fugu.json`
(full data, hosted fugu, 80-min agent budget). Runs are non-deterministic (hosted model + live
leaderboard read fresh each cycle), and fugu auto-routes across backend providers, so per-cycle latency
varies. To curb the two remaining timeouts, either raise the budget further or constrain the agent's
in-cycle experimentation (the binding limit here was the agent over-exploring, not the wall being too
low).

## Next steps (decision for the next run)

**Raise `sandbox_timeout_s` 4800 → 9600 (80 → 160 min).** The two cycles that timed out (5, 7) were the
agent's *most promising* experiments — the multi-harness multiplier search and the K-fold OOF
**stacking** bake-off, i.e. exactly the methods most likely to break the ~0.966 plateau — and 80 min
wasn't enough to finish them. 160 min gives them room to complete, which is the bet worth making before
concluding the plateau is a ceiling. Already applied to [`task.fugu.json`](../../task.fugu.json); the
soft wrap-up nudge moves to 128 min (80% of budget).

Only `sandbox_timeout_s` changes — it bounds the agent's *development* budget. `code_timeout_s` (the
verifier's per-script run budget) and `VERITY_VERIFIER_TIMEOUT` stay put: submission scripts run on the
verifier in ~1–3 min, far inside the existing 80-min `code_timeout_s`.

Two caveats going in: (1) this raises the ceiling the agent explores up to but does **not** fix the
root cause that it doesn't self-budget; (2) total wall grows — cycles that *use* the full budget cost
up to 160 min each, so plan for a longer run (this 80-min run was ~9.8 h; the next could land ~12–16 h).
