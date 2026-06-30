# Second end-to-end `fe-kaggle` run — full data, claude-sonnet-4-6 (2026-06-21)

The second full-data end-to-end run of `fe-kaggle`, identical in task/data/infra to the
[gpt-5.4 run](../2026-06-21-gpt5.4-fulldata/README.md) earlier the same day — only the model
changed (and the writable-CWD fix #88 had since landed). Run as an apples-to-apples comparison of how
a different frontier model writes and adapts across the read → propose → gate → commit loop.

| | |
|---|---|
| run_id | `cffff78cce9847bb870081c04ef48c60` |
| task_id | `f71fc80660464e6abdecde648cc03f64` |
| date | 2026-06-21 |
| model | `claude-sonnet-4-6` (hosted Anthropic) |
| competition | `playground-series-s6e6` (stellar classification, balanced accuracy) |
| data | **full** — train 490,746 (agent) / hold-out 86,601 / real test 247,435 |
| config | [`../../task.sonnet.json`](../../task.sonnet.json) (`max_cycles=10`, `stop_on_accept=false`) |
| totals | 10 cycles · ~109.6 min wall · **7,316,709 tokens** |
| outcomes | 5 revised · 4 rejected · 1 sandbox-failed · **0 accepted / 0 Kaggle submissions** |

> **Two cycles were terminated by an operator kill, not by the gate.** Cycles 3 and 8 (the CatBoost
> hangs) were `docker rm -f`'d while stuck rather than left to hit the 60-minute timeout. This is
> flagged up front because it shaped the run: the kill exits `137`, which the agent then repeatedly
> misread as an out-of-memory kill (see *Failure modes*). The wall-clock above is inflated by the
> time those hung cycles ran before the kill (cycle 8 alone was ~42 min).

## What we observed

| cyc | outcome | proxy | steps | tokens | wall | submission |
|---:|---|---:|---:|---:|---:|---|
| 0 | revised | 0.9627 | 45 | 880,669 | 4m14s | LightGBM, 1000 trees |
| 1 | sandbox-failed | — | — | — | 4m16s | (no proposal — `GraphRecursionError`, 200-step cap) |
| 2 | revised | 0.9664 | 31 | 607,072 | 5m07s | XGBoost + LightGBM ensemble |
| 3 | rejected | — (137) | 26 | 502,715 | 13m55s | +CatBoost +LGB-DART → **0%-CPU hang (killed)** |
| 4 | revised | 0.9665 | 30 | 563,569 | 4m00s | dropped XGBoost ("OOM") |
| 5 | rejected | **0.9667** | 47 | 1,194,766 | 9m27s | XGBoost + LightGBM (best score; refine-cap) |
| 6 | revised | 0.9661 | 44 | 1,286,485 | 8m48s | LightGBM + ExtraTrees |
| 7 | rejected | — | 46 | 1,422,309 | 11m52s | LGB **+CatBoost** → `subsample`/bayesian param crash |
| 8 | rejected | — (137) | 21 | 381,142 | 42m23s | LGB **+CatBoost** +ET → **high-CPU wedge (killed)** |
| 9 | revised | 0.9664 | 26 | 477,982 | 5m33s | dropped ExtraTrees ("OOM"), two LightGBMs |

Trajectory of the scored cycles:

```
0.9627 → ✗ → 0.9664 → ✗ → 0.9665 → 0.9667 → 0.9661 → ✗ → ✗ → 0.9664
```

Best hold-out balanced accuracy **0.9667** (cycle 5). Competitive bar **0.9714**. No cycle's estimate
reached the bar, so the gate spent **zero** Kaggle submissions.

## The agent adapted across cycles (evidence)

Sonnet visibly changed approach each cycle in response to the gate's feedback — quoting its own
recorded rationales:

- **Built up an ensemble, then optimized calibration.** Single LightGBM (c0) → XGBoost+LightGBM with
  *"threshold calibration dividing by class priors to maximize balanced accuracy"* (c2) → richer
  features and stronger LightGBM (c5, the best score).
- **Recovered from every failure and kept moving** — after each rejected cycle it produced a fresh,
  runnable submission the next cycle (degrade-don't-crash held throughout).

But the adaptation ran on a **wrong causal model** for the hangs (see below) — the agent attributed
the killed cycles to memory, not to a library conflict, and "fixed" them by dropping libraries on
memory grounds. That happened to avoid the conflict (removing a library dissolves it), so it muddled
to clean runs — but never diagnosed the actual cause.

## Failure modes observed (and how the harness handled each)

| cyc | failure | handling |
|---:|---|---|
| 1 | `GraphRecursionError` — the agent took >200 graph steps without finishing | recorded as a failed cycle; **run continued** (degrade-don't-crash) |
| 3 | XGBoost + LightGBM + CatBoost → froze at **0% CPU** | **operator-killed**; `runs-clean` reject |
| 7 | CatBoost `subsample` set with default `bayesian` bootstrap (unsupported) | clean exit-1; `runs-clean` reject, reason fed back |
| 8 | LightGBM + CatBoost + ExtraTrees → **spun ~13 cores for 34 min with no progress** (flat memory, empty `/out`) | **operator-killed**; `runs-clean` reject |

**The OOM misdiagnosis.** Both kills (cycles 3, 8) exit `137` — the same code the Linux OOM-killer
uses. The agent read that as out-of-memory and optimized for it, three separate cycles:

> c4: *"rejected with exit 137 (**OOM kill**) … exceeded the 2GB RAM budget … removed XGBoost (biggest RAM consumer)"*
> c6: *"safe now that XGBoost was removed, eliminating the **OOM pressure**"*
> c9: *"exited with code 137 (**OOM**) … ExtraTrees consumes too much memory … removed ExtraTrees"*

The runner has 16 GB (the "2 GB budget" is invented), and neither hang was an OOM — cycle 3 was a
0%-CPU deadlock, cycle 8 a high-CPU spin, both at ~2 GiB with memory flat. The agent never diagnosed
the real cause; a bare `137` from an operator/no-progress kill is indistinguishable from an OOM kill,
so it confabulated a plausible-but-wrong story and ran with it.

## The CatBoost finding (cross-run, cross-model)

Lining up every multi-library attempt across both runs:

| libraries in one process | result |
|---|---|
| XGBoost + LightGBM | ✅ runs clean (sonnet c2/c5; gpt) |
| LightGBM + CatBoost | ⚠️ **stalled** (sonnet c8 — high-CPU spin) |
| XGBoost + LightGBM + CatBoost | ❌ **hung** (sonnet c3 — 0% CPU; gpt cycle 9) |

The common factor in every stall is **CatBoost co-loaded with a LightGBM/XGBoost process** — two
independent models reached for it and both got stuck, in two different signatures (a 0%-CPU block and
a high-CPU spin). The raw evidence for the cycle-8 wedge is saved here:
`sub8-catboost-wedge.console.log` (frozen at CatBoost iter 200) and `sub8-catboost-wedge.proctop.txt`
(one process, 1305% CPU, 7.5 CPU-hours, flat memory). The prompt nudge added afterward
([PR #89](https://github.com/kynetyk-ai/verity/pull/89)) steers the agent to isolate such libraries
in separate processes; it was **not** in this run.

## Comparison to gpt-5.4 (same task/data/infra)

| | gpt-5.4 | claude-sonnet-4-6 |
|---|---|---|
| best proxy | 0.9664 | **0.9667** |
| gap to bar (0.9714) | 0.0050 | **0.0047** |
| **Kaggle submissions** | **0** | **0** |
| outcomes | 6 rev / 3 rej / 1 sbx | 5 rev / 4 rej / 1 sbx |
| wall | ~120 min | ~110 min (incl. the 42-min wedged cycle) |
| tokens | 3.35M | **7.32M (2.2×)** |

Both models parked in the 0.966–0.967 band, neither crossed the bar, both spent **zero** real
submissions — the competitive gate behaved identically across two very different models. Sonnet edged
slightly higher (0.9667 vs 0.9664) at ~2.2× the token cost, and was more iterative (the only run to
hit the 200-step recursion cap). Both independently wrote the XGBoost+LightGBM+CatBoost combination
and both got stuck on it.

## The submissions

The 9 scripts sonnet wrote, as the gate ran them (cycle 1 produced none):

| script | cycle | result |
|---|---:|---|
| `submissions/cycle0-submission.py` | 0 | revised · 0.9627 |
| `submissions/cycle2-submission.py` | 2 | revised · 0.9664 |
| `submissions/cycle3-submission.py` | 3 | rejected · 0%-CPU hang (killed) |
| `submissions/cycle4-submission.py` | 4 | revised · 0.9665 |
| `submissions/cycle5-submission.py` | 5 | rejected · **0.9667 (best)** · refine-cap |
| `submissions/cycle6-submission.py` | 6 | revised · 0.9661 |
| `submissions/cycle7-submission.py` | 7 | rejected · CatBoost param crash |
| `submissions/cycle8-submission.py` | 8 | rejected · high-CPU wedge (killed) |
| `submissions/cycle9-submission.py` | 9 | revised · 0.9664 |

`results.json` is the full machine-readable RunReport (every cycle's decisions, scores, timings, and
rationales).

## Reproduce

Per [`../../PROTOCOL.md`](../../PROTOCOL.md): bring up the daemon, then
`./prototyping_datasci_test/run.sh prototyping_datasci_test/task.sonnet.json` (full data, hosted
model). Runs are non-deterministic (hosted model + live leaderboard).
