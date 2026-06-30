# End-to-end `fe-kaggle` run — full data, Sakana **fugu**, 160-min budget (2026-06-27)

A follow-on to the [80-min fugu run](../2026-06-26-fugu-fulldata-rerun/README.md) that **doubled the
agent time budget** — `sandbox_timeout_s` **4800 → 9600 (80 → 160 min)** — to test a specific
hypothesis: the prior run's two timeouts were on the agent's *most promising* experiments (multi-harness
tuning, K-fold OOF stacking), so maybe more time lets them finish and break the ~0.966 plateau.

**The hypothesis half-held, and the experiment is a cautionary result.** One long experiment did finish
and set a new best (cycle 1, **0.9668** — beating the entire 80-min run). But doubling the budget bought
**+0.0002 on the headline metric for +6 h of wall and *more* instability** — three failed cycles instead
of two, including a **new failure mode**: with 160 minutes the agent took so many steps it exhausted the
`recursion_limit` and crashed (see below). Net: **more wall-clock was the wrong lever.**

| | |
|---|---|
| run_id | `36fa2867f6cd452aaf574b07960d7f0e` |
| task_id | `8b62150b9caa4c19a324549d50b0dced` |
| date | 2026-06-27 |
| model | `fugu` (Sakana, OpenAI-compatible at `api.sakana.ai/v1`; key via `OPENAI_API_KEY`) |
| competition | `playground-series-s6e6` (stellar classification, balanced accuracy) |
| data | **full** — train 577,347 · test 247,435 · agent-train 490,746 · hold-out 86,601 (15% stratified) |
| config | [`task.fugu.json`](../../task.fugu.json) — `max_cycles=10`, `stop_on_accept=false`, **`sandbox_timeout_s=9600`**, `code_timeout_s=4800`, `recursion_limit=300` |
| totals | 10 cycles · **~16.0 h wall** · **8,700,886 tokens** · 314 model steps · 363 tool calls |
| outcomes | **7 revised · 3 sandbox-failed (1 timeout · 2 recursion-limit crashes)** · **0 accepted / 0 Kaggle submissions** |
| best hold-out | **0.9668** (cycle 1) · live competitive bar ~0.9719 · gap never below **0.0052** |

## What we observed

Per-cycle, from the machine-readable [`results.json`](results.json) (cycles 0-indexed; "wall" is the
full cycle — agent sandbox **+** the gate's full-data re-run):

| cyc | outcome | proxy bal-acc | gap | wall | tokens | note |
|---:|---|---:|---:|---:|---:|---|
| 0 | revised | 0.9663 | 0.0056 | 84 m | 994,543 | |
| 1 | revised | **0.9668** | 0.0052 | 190 m | 1,614,523 | **best** — ~155-min experiment + ~35-min gate run |
| 2 | revised | 0.9657 | 0.0062 | 37 m | 1,256,484 | |
| 3 | **timed out** | — | — | 160 m | — | hit the 160-min wall (no proposal) |
| 4 | revised | 0.9656 | 0.0064 | 25 m | 1,037,452 | fast post-timeout recovery |
| 5 | revised | 0.9652 | 0.0068 | 135 m | 1,921,361 | |
| 6 | **crashed** | — | — | 140 m | — | **`GraphRecursionError` — recursion_limit=300 exhausted** |
| 7 | revised | 0.9652 | 0.0068 | 18 m | 777,033 | |
| 8 | revised | 0.9663 | 0.0057 | 26 m | 1,099,490 | |
| 9 | **crashed** | — | — | 145 m | — | **`GraphRecursionError` — recursion_limit=300 exhausted** |

Trajectory of the scored cycles:

```
0.9663 → 0.9668 → 0.9657 → ✗timeout → 0.9656 → 0.9652 → ✗crash → 0.9652 → 0.9663 → ✗crash
```

Best **0.9668** (cycle 1), set early and **never beaten**. After the peak the scores **drifted down**
(0.9668 → 0.9657 → 0.9656 → 0.9652) then partially recovered to 0.9663 — see the incumbent note below.

## The headline findings

### 1. More time was the wrong lever (vs. the 80-min run)

| | [80-min run](../2026-06-26-fugu-fulldata-rerun/README.md) | this 160-min run |
|---|---|---|
| best proxy | 0.9666 | **0.9668** |
| wall | 9.8 h | **16.0 h** |
| failures | 2 (both timeouts) | **3 (1 timeout · 2 recursion crashes)** |

Doubling the budget produced a **+0.0002** score gain for **+6 h** and more failures. The one genuine
win (cycle 1's 0.9668) proves a long experiment *can* pay off — but nothing after it beat it, and the
overall ROI was poor. The binding constraint is methodology/ceiling and the provisioning regression
(below), **not** wall-clock.

### 2. A new failure mode: the agent hits the *step* wall, not just the *time* wall

Cycles 6 and 9 did **not** time out — they crashed with
`langgraph.errors.GraphRecursionError: Recursion limit of 300 reached without hitting a stop condition`
(`deepagents_driver.py:201`, `agent.invoke`). Given 160 minutes, the agent took so many model↔tool
steps that it exhausted the `recursion_limit=300` we set earlier and the worker died with **no proposal
and no soft-nudge warning** (the wall-clock `DeadlineMiddleware` nudge has no recursion-limit analog
wired for this task). So with a larger time budget the binding wall **shifted from wall-clock to
step-count** — and the step wall is an *ungraceful* hard crash. (Filed as a follow-up issue.)

### 3. "The agent expands to fill the budget" — confirmed again, harder

Cycle walls: 84, **190**, 37, 160(✗), 25, 135, 140(✗), 18, 26, 145(✗). The agent repeatedly consumed
most of the 160-min budget, and the two longest non-timeout experiments (cycles 5, then 6/9) hit the
step ceiling. This is the same "doesn't self-budget" finding as the 80-min run, now with a second,
harder ceiling.

## The incumbent regression (issue #95), observed live across the whole run

Provisioning was `LAST_REVISED_OR_ACCEPTED` (newest-by-recency). Cycle 1 set 0.9668, but because every
later attempt was the newest `revised`, the agent kept building on its *most recent* (often worse) base,
not its best — the 0.9668 peak was **orphaned for the entire back half** while scores drifted to ~0.9652.
This is the exact behavior of **issue #95**, whose fix (`BEST_REVISED_OR_ACCEPTED`, now the fe-kaggle
default) merged in #100 *after* this run started — so it will apply to the **next** run, not this one.

## Failure modes and how the harness handled them

| cyc | failure | handling |
|---:|---|---|
| 3 | sandbox hit the 9600 s wall (no proposal) | recorded `sandbox_failed`, fed back, run continued |
| 6, 9 | `GraphRecursionError` (recursion_limit=300) after ~140 min | recorded `sandbox_failed` with the traceback, fed back, run continued |

All three were non-consecutive, so `max_consecutive_sandbox_failures=3` never tripped. Degrade-don't-crash
held on a failure mode it had not seen before (a worker exception vs. a timeout kill).

## The submissions

The seven scored scripts (cycles 3, 6, 9 produced none):

| script | cyc | proxy |
|---|---:|---:|
| `submissions/cycle0-submission.py` | 0 | 0.9663 |
| `submissions/cycle1-submission.py` | 1 | **0.9668 (best)** |
| `submissions/cycle2-submission.py` | 2 | 0.9657 |
| `submissions/cycle4-submission.py` | 4 | 0.9656 |
| `submissions/cycle5-submission.py` | 5 | 0.9652 |
| `submissions/cycle7-submission.py` | 7 | 0.9652 |
| `submissions/cycle8-submission.py` | 8 | 0.9663 |

[`results.json`](results.json) is the full RunReport (per-cycle gate decisions, proxy scores,
competitive estimates, timings, telemetry, and the two recursion-limit tracebacks under `sandbox_error`).

## Conclusion / next-run guidance

- **Revert the time budget** — 160 min did not earn its cost; the 80-min budget is the better operating
  point. The bottleneck is not wall-clock.
- **The step ceiling needs a graceful path** (issue filed): either a recursion-limit soft-nudge analog
  of `DeadlineMiddleware`, or harvest an in-progress proposal on `GraphRecursionError`, so a long cycle
  doesn't lose all its work to a hard crash.
- **#95 fix should help the next run** — the agent will build on its best (0.9668-class) artifact instead
  of drifting down from a recency incumbent.
- The ~0.005 gap to the competitive bar looks like a **methodology/ceiling** limit for this feature/model
  family, not a budget limit — the path forward is the ablation + better methods, not more time.
