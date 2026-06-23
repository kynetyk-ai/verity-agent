# Third end-to-end `fe-kaggle` run — sonnet-4-6 with the parallel-library prompt nudge (2026-06-22)

A re-run of [the sonnet run](../2026-06-21-sonnet-fulldata/README.md) with one intended change: the
prompt now warns the agent that *"CatBoost loaded alongside LightGBM or XGBoost … the run can STALL"*
and to train such libraries in separate processes ([PR #89](https://github.com/kynetyk-ai/verity/pull/89)).
Same model, same task/data/infra otherwise. The question: does the nudge change behavior — fewer
CatBoost stalls — and does it still land ~0.967?

| | |
|---|---|
| run_id | `2e34d6b98d194c60b82ddeab89e9f3e8` |
| task_id | `cb89c7e147334ae9ab764c85606980e1` |
| date | 2026-06-22 |
| model | `claude-sonnet-4-6` (hosted Anthropic) |
| change vs run-1 | the CatBoost/parallel-library prompt nudge (#89) baked into the CP image |
| competition | `playground-series-s6e6` (stellar classification, balanced accuracy) |
| data | **full** — train 490,746 (agent) / hold-out 86,601 / real test 247,435 |
| config | [`../../task.sonnet.json`](../../task.sonnet.json) (`max_cycles=10`, `stop_on_accept=false`) |
| totals | 10 cycles · ~168 min wall · 4,846,600 tokens |
| outcomes | 5 revised · 4 sandbox-failed · 1 rejected · **0 accepted / 0 Kaggle submissions** |

> Unlike run-1, **no cycle was operator-killed** — every failure here was the harness's own: four
> recursion-cap degrades and one clean gate-timeout. The ~168-min wall is inflated by that timeout
> (cycle 8 ran the full 60-min code budget before the gate cut it).

## What we observed

| cyc | outcome | proxy | steps | note |
|---:|---|---:|---:|---|
| 0 | revised | 0.9646 | 49 | LightGBM |
| 1 | revised | 0.9658 | 31 | |
| 2 | revised | 0.9651 | 39 | |
| 3 | sandbox-failed | — | — | `GraphRecursionError` (200-step cap) |
| 4 | sandbox-failed | — | — | `GraphRecursionError` |
| 5 | revised | 0.9644 | 46 | 6×LightGBM + 2×XGBoost + OOF threshold opt |
| 6 | sandbox-failed | — | — | `GraphRecursionError` |
| 7 | sandbox-failed | — | — | `GraphRecursionError` |
| 8 | rejected | — | 18 | 5-fold stack of a big LGB+XGB ensemble → **timed out** (60-min gate budget) |
| 9 | revised | 0.9656 | 20 | |

Trajectory of the scored cycles:

```
0.9646 → 0.9658 → 0.9651 → ✗ → ✗ → 0.9644 → ✗ → ✗ → ✗timeout → 0.9656
```

Best hold-out balanced accuracy **0.9658** (cycle 1). Competitive bar **0.9714**. No cycle's estimate
reached the bar, so the gate spent **zero** Kaggle submissions.

## On the prompt nudge — what we can and can't claim

**What we saw:** zero CatBoost stalls across all 10 cycles (run-1 had two), and the run's biggest
ensembles (cycles 5, 9) got their diversity from **many LightGBM configs + XGBoost** — the
libgomp-only combination that runs clean — never co-loading CatBoost.

**What we cannot claim:** that the nudge *caused* this. The agent's rationales (in `results.json`)
**never mention** CatBoost, separate processes, stalls, or memory — it simply never reached for
CatBoost this run, and never narrated the warning. A clean CatBoost record is equally consistent with
ordinary run-to-run variance (sonnet's runs are non-deterministic; it wrote different scripts than
run-1). **One run can't isolate the prompt as the cause** — it would take several runs with and
without the nudge to separate signal from variance. So: suggestive, not established.

## The recursion cap was the dominant failure

The clearer story this run is the **200-step recursion cap**: it took out **4 of 10 cycles** (vs 1 in
run-1). That is the failure mode the nudge does *not* address — it's about how iterative the agent is
(sonnet routinely exceeds 200 langgraph steps), not about libraries. Plus one cycle (8) was lost to a
genuinely over-heavy script (a 5-fold stack of a big ensemble) that the gate timed out cleanly. Net:
only 5 of 10 cycles produced a score, and best (0.9658) came on cycle 1.

## Three-run comparison (same task/data/infra)

| | gpt-5.4 | sonnet (no nudge) | sonnet (nudge) |
|---|---|---|---|
| best proxy | 0.9664 | **0.9667** | 0.9658 |
| Kaggle submissions | 0 | 0 | 0 |
| CatBoost stalls | 1 | 2 | **0** |
| recursion-cap fails | 0 | 1 | **4** |
| operator kills | 0 | 2 | **0** |
| clean gate-timeouts | 0 | 0 | 1 |
| scored cycles | 6 | 6 | 5 |
| tokens | 3.35M | 7.32M | 4.85M |
| wall | ~120 min | ~110 min | ~168 min |

All three parked in the **0.9644–0.9667** band, none crossed the 0.9714 bar, all spent **zero**
Kaggle submissions — the competitive gate behaved identically across two models and a prompt change.
The standout difference this run is **not** the score but the failure mix: no CatBoost stalls, but
the recursion cap dominated. The honest next lever (a *future* run, not this one) is raising
`recursion_limit` — sonnet is simply more iterative than 200 steps allows, and it's costing nearly
half the cycles.

## The submissions

The 6 scripts that committed a proposal (the four recursion-cap cycles produced none):

| script | cycle | result |
|---|---:|---|
| `submissions/cycle0-submission.py` | 0 | revised · 0.9646 |
| `submissions/cycle1-submission.py` | 1 | revised · **0.9658 (best)** |
| `submissions/cycle2-submission.py` | 2 | revised · 0.9651 |
| `submissions/cycle5-submission.py` | 5 | revised · 0.9644 (6×LGB + 2×XGB, no CatBoost) |
| `submissions/cycle8-submission.py` | 8 | rejected · timed out (5-fold stack) |
| `submissions/cycle9-submission.py` | 9 | revised · 0.9656 |

`results.json` is the full RunReport (every cycle's decisions, scores, timings, and rationales).

## Reproduce

Per [`../../PROTOCOL.md`](../../PROTOCOL.md):
`./feature-engineering-test/run.sh feature-engineering-test/task.sonnet.json` (full data, hosted
model), with the nudge merged in `feature_engineering.py`. Non-deterministic (hosted model + live
leaderboard).
