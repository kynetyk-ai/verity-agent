# Code-runner timeout calibration (issue #111)

**Question:** the fe-kaggle gate runs each agent script on the code-runner under `code_timeout_s`, a
hand-guessed budget (currently **1800s / 30 min**). Is that fair, or does it silently kill valid,
heavy-but-correct scripts? This measures the real cost directly.

## Method (existing infra, minimal driver)

A thin host-side harness (`calibrate.py`) drives the **exact** component the gate uses —
`verity.verifier.ContainerCodeRunner` on the `verity-code-runner` image — at the gate's real caps
(**memory 16g, cpus 16, tmpfs 2g, pids 128**), on the **full-data kaggle-rung workload**
(`verifier/full_train.csv` ~101 MB + real `verifier/test.csv` ~41 MB). No daemon, verifier service, or
agent. Wall-clock = `pip install` + script run (exactly what `code_timeout_s` bounds), timed externally.

**Subset:** the 8 heaviest **non-CatBoost** submission scripts collected across the prototyping runs.
CatBoost was excluded deliberately — its co-load-with-LightGBM/XGBoost stall and its
`catboost==1.2.5`-vs-`numpy 2.x` ABI break register as a hang/crash, not a runtime, and would poison the
measurement. A single uniform, conflict-free `requirements.txt` was used for every script so the number
is *runtime*, not dependency-install variance.

Candidate 4 hit the harness's 3600s cap on the first pass — the *same* arbitrary 60-min value under test,
so that told us nothing. It was re-measured alone with a 6h non-clipping cap (`followup_candidate4.py`).

## Results — 8/8 ran clean

| script | model mix | wall (install + full-data train) |
|--------|-----------|----------------------------------|
| fugu-rerun-c9 | LGBM+XGB | **148s** (2.5m) |
| fugu160-c5 | LGBM | **219s** (3.6m) |
| fugu160-c8 | LGBM+XGB | **315s** (5.2m) |
| fugu160-c2 | LGBM | **326s** (5.4m) |
| gpt5.4-c8 | LGBM+XGB | **1270s** (21m) |
| sonnet-nudge-c5 | ExtraTrees+LGBM+XGB | **1776s** (30m) |
| qwen-cyc8 | 4-config LGBM ensemble | **1889s** (31m) |
| sonnet-nudge-c8 | 5-fold RF+ExtraTrees+LGBM+XGB stack | **4312s** (72m) † |

† clipped at 3600s on the first pass (the budget under test); true runtime measured separately at
**4312s, clean**.

Every one of these scripts was a real agent submission that ran to completion. None are pathological.

## Findings

1. **Huge spread among valid scripts: 148s → 4312s (29×).** The cost is driven by **fold count ×
   ensemble size**, not by library family — a 4-config LightGBM ensemble (1889s) is as slow as a sklearn
   ExtraTrees one (1776s), and the heaviest is a 5-fold stack of four model types (4312s). There is no
   clean "GBDT = fast, sklearn = slow" line to cut on.
2. **The current 1800s budget kills valid work.** Two of the eight (qwen-cyc8 1889s, sonnet-c8 4312s)
   exceed it and would be cut mid-train despite being correct; a third (sonnet-c5 1776s) survives by only
   24 seconds. So ~25–37% of these real submissions are at or over the current line — the guessed budget
   is too tight, exactly the silent-failure risk #111 was about.
3. **The heaviest valid script needs 72 min.** It was being clipped at 60 min in its original run and
   recorded as a "timeout reject" — which read as *too heavy to allow* but was really *12 minutes short
   of a guessed budget*. Measuring it removed that mislabel.

## Recommendation

**Permissive bound = 3× the slowest clean completion = 3 × 4312s ≈ 12,936s ≈ 3.6 h.**

That is the literal answer to "set it so nothing valid gets cut, with headroom." But it surfaces a real
tension to decide, not a number to paste in: **3.6 h is far larger than the agent's entire cycle**
(`sandbox_timeout_s` = 1800s / 30 min). A gate budget that dwarfs the cycle is impractical, so the true
output of this calibration is an **informed policy choice** about how heavy a script the gate should
permit:

| target | budget (3× the class max) | covers |
|--------|----------------------------|--------|
| GBDT + single-model-type ensembles | 3 × 1889s ≈ **5,667s ≈ 94 min** | 7 / 8 (all but the 5-fold 4-type stack) |
| everything measured, incl. the 72-min stack | 3 × 4312s ≈ **12,936s ≈ 3.6 h** | 8 / 8 |
| (current) | 1800s / 30 min | 5 / 8 |

The key shift vs. before: whichever line is chosen, scripts above it are now cut **by an informed
decision about cost we have measured** — not by an accident of a guessed number. If the gate stays near
the cycle length, that's a deliberate "we don't permit >30-min training," not an oversight.

## Files

- `calibrate.py` — the 8-script harness (the minimal driver over `ContainerCodeRunner`).
- `followup_candidate4.py` — the non-clipping re-measure of the clipped script.
- `requirements.txt` — the uniform, no-CatBoost dependency set.
- `results/` — per-script `stdout`/`stderr`, `summary.json` (main pass), `candidate4_followup.json`,
  and the run logs.

*Memory note:* the measurement covers wall-clock only (a clean-vs-OOM check via the 16g cap; no fine
peak-memory profiling). All eight fit in 16g without OOM.
