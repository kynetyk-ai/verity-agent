# Follow-up run — qwen, top-50% bar, full-lineage provisioning, post-report steers

**A follow-up to [`PROTOTYPING_REPORT.md`](../../PROTOTYPING_REPORT.md).** Same task/data/infra as the
runs recorded there; this one was taken *after* the changes that report motivated landed on `develop`
(PRs #105/#107/#108/#109), to see how the same local model behaves with them in place. It is a single
run — read it as one more data point next to the report's runs, not as a verdict on the changes.

## What was different vs the report's runs

- **Model / bar:** local `qwen3.6:27b-coding-mxfp8` (Ollama), `target_percentile: 50` (top-50% bar),
  `sandbox_timeout_s` and `code_timeout_s` both 1800 s, `max_cycles: 10`.
- **Steers/hooks now live** (rebuilt images from `develop`): the sharpened FE system prompt (#107),
  the propose-tool **outbox file-presence check** + soft **`tested` attestation** + the two-step **80%
  deadline nudge** (#108), and the **provisioning decoupling** (#109).
- **Provisioning:** `policy.provisioning = {statuses: [accepted, superseded, revised], select: all}` —
  i.e. provision *every* script that ever cleared the gate (the "lineage of bests").

## Headline outcome

10 cycles, **~185 min** wall-clock. Outcomes: **1 accepted, 5 rejected, 4 sandbox-failed.**

- The one accepted submission (cyc1) scored **proxy 0.9649 / Kaggle public 0.9653**.
- **The lineage provisioning never triggered:** with only one accept there was nothing to supersede
  (`superseded: []`), so the workspace layout stayed the flat single-artifact form all run. The config
  parsed and resolved correctly; it simply had ≤1 qualifying artifact every cycle (the same arithmetic
  the decoupling fixed — just never exercised past one accept here).

## Per-cycle trajectory

| cyc | outcome | gate reached / reason | proxy | steps | tokens | script |
|----|---------|------------------------|-------|-------|--------|--------|
| 0 | sandbox-failed | no proposal (empty outbox) | — | — | — | — |
| 1 | **accepted** | runs-clean → proxy → competitive → **kaggle** | 0.9649 | 33 | 555k | 122 ln, single LightGBM |
| 2 | sandbox-failed | no proposal (empty outbox) | — | — | — | — |
| 3 | rejected | runs-clean: `AttributeError: 'numpy.ndarray' object has no attribute 'values'` | — | 31 | 532k | 175 ln, +xgboost |
| 4 | rejected | proxy-improves: no improvement (0.9653 vs best 0.9649, margin 0.0010) | 0.9653 | 22 | 326k | 214 ln |
| 5 | rejected | runs-clean: `ValueError: numpy.dtype size changed … binary incompatibility` | — | 28 | 530k | 251 ln, +catboost, early-stop |
| 6 | sandbox-failed | no proposal (empty outbox) | — | — | — | — |
| 7 | rejected | proxy-improves: no improvement (0.9645 vs best 0.9649) | 0.9645 | 29 | 479k | 209 ln, +xgboost |
| 8 | rejected | proxy-improves: no improvement (0.9639 vs best 0.9649) | 0.9639 | 25 | 405k | 268 ln, 4-config ensemble |
| 9 | sandbox-failed | no proposal — **outbox had `submission.py`** (wrote it, never called submit) | — | — | — | — |

The six proposing cycles' scripts are saved under [`submissions/`](submissions/); `results.json` and the
run's `task.json` are alongside.

## What happened, cycle to cycle (and how it tracks the feedback received)

Each cycle was served the prior reject reason as feedback plus — via the lineage config — the **accepted
cyc1 script** in `scratch/provided/` (with one accept, the lineage = just cyc1). Two patterns are visible
in the scripts and how they follow that feedback:

1. **Crashes came from *scaling up*, and the next attempt fixed the specific crash.** The accepted cyc1
   is a lean 122-line single-LightGBM model. cyc3 scaled to 175 lines + XGBoost and **crashed**
   (`ndarray … has no attribute 'values'` — a feature-build slip); cyc4 (214 ln) then ran **clean** and
   scored 0.9653 — the crash class was gone. cyc5 added **CatBoost** (+ early-stopping) and crashed on a
   **CatBoost/NumPy binary-incompat** (`numpy.dtype size changed`, from `catboost==1.2.5` against
   `numpy==2.1.0` in its `requirements.txt`); cyc7 then **dropped CatBoost** (back to LightGBM+XGBoost),
   pinned `numpy==2.1.0`, and ran clean. So the agent did respond to specific failures — a crash one
   cycle was generally absent the next — but it kept *re-introducing new* failure classes as it reached
   for more model libraries.

2. **The "no improvement" feedback drove more *complexity*, not higher scores.** The four clean,
   scoring scripts (cyc1, 4, 7, 8) land at **0.9649 / 0.9653 / 0.9645 / 0.9639** — a tight cluster, all
   within the **0.0010 noise margin** of the accepted 0.9649 (cyc4's 0.9653 is the highest raw but only
   +0.0004, correctly read as noise by the proxy gate, so no Kaggle submission spent). Told repeatedly
   "you didn't beat your best," the agent kept *elaborating* — 122 → 214 → 209 → 268 lines, adding color
   gradients, ratios, log-ratios, redshift interactions, coordinate sin/cos, polynomial terms, and by
   cyc8 a **4-config LightGBM ensemble** — rather than minimally editing the provided cyc1 script. The
   added machinery did not move the score off the plateau. (Note: the cyc8 4-config ensemble ran *within*
   the 30-min gate budget — it did not time out.)

3. **Four cycles produced no proposal.** Three (cyc0/2/6) left an empty outbox; cyc9 **wrote
   `submission.py` to the outbox but never called the submit tool** — the exact "did the work, didn't
   submit" failure the new two-step 80% nudge targets. (The run-scoped worker logs aren't retained, so
   whether the nudge fired that cycle can't be read back here.)

## Against the report's observations

- **The ~0.96 plateau holds.** The report found a model-agnostic ceiling around ~0.966; this run's clean
  scripts sit at 0.9639–0.9653 with materially more feature engineering, consistent with that ceiling
  being a feature/data/compute property rather than a prompt or model one.
- **The proxy/competitive gate behaved as designed** — it declined to spend a Kaggle submission on a
  within-noise "improvement" (cyc4's +0.0004) three separate times, exactly the budget discipline the
  ladder is for.
- **Bug *classes* shifted but did not disappear.** The report's runs failed on crashes (e.g.
  mixed-label, API-misuse) and timeouts; here the crashes were `ndarray.values` and a CatBoost/NumPy ABI
  mismatch, and the heavy ensemble that earlier would have risked a timeout finished in budget. The
  no-proposal failures (incl. wrote-but-didn't-submit) persisted.
- **Comparison point:** the report's `fugu`/hosted and the prior-day qwen run reached two accepts with a
  documented climb; this run reached one accept (0.9653 public) and then plateaued within noise. Whether
  that delta is the model, the bar, the steers, or run-to-run variance is **not** determinable from one
  run — recorded here as an observation, not a conclusion.

## Provisioning decoupling — status

The decoupling (#109) is in and correct: `policy.provisioning` with an explicit `{statuses, select}`
parsed with no warning and resolved to the lineage selection. Its *effect* (multiple prior scripts
namespaced under `provided/01-…/`, `02-…/` with the INDEX) requires ≥2 artifacts in the selected set,
which needs a second accept (to produce a `superseded` sibling) — not reached this run. So the capability
is verified at the config/resolution layer here; the multi-artifact materialization remains to be
observed on a run that banks a second accept.

## Next steps (filed)

Forward tracks this run surfaced — to get *fair gates* and *clean signal* before reading more into
model/prompt comparisons:

- **[#111] Calibrate the code-runner gate budget from measured runtimes.** The `code_timeout_s` budget is
  guessed; a too-tight budget would fail valid heavy scripts and pollute the fails-to-run signal. (This
  run's crashes were genuine code/dep bugs, not timeouts — but the budget is still uncalibrated, and the
  agent's 4 GB sandbox differs from the gate's 16 GB runner.) A one-shot calibration: measure
  representative full-data scripts' wall-time + peak memory on the gate's runner, set the budget from the
  distribution.
- **[#112] A model + instruction selection protocol that separates fails-to-run rate from BA.** qwen's
  high runtime-bug rate (it doesn't self-test) swamps signal; and the FE prompt's stellar-domain
  background (colour indices / redshift / encode the categoricals) may be doing the feature engineering
  *for* the agent — inflating the floor and compressing BA so model/system contribution is invisible.
  Next: a small grid of {model × background-rich vs background-lean instructions} scored on
  (fails-to-run rate, clean-run BA) as separate axes, to pick a baseline where BA actually reflects the
  model + system rather than the prompt's hints. (The verifier may also be too simple to separate
  configs.)
- **[#113] On a cheap-gate (runs-clean) crash, return a "test your code first" steer in the outcome.**
  The unenforceable prompt nudge isn't landing on qwen; relaying the traceback *plus* an explicit
  "run it end-to-end before resubmitting" puts the steer at the moment of failure. The stronger
  alternative — a better model bug-reviews the failed code into a `revise` message — risks the reviewer
  leaking non-trivial *improvement* (not just bug-fix) information, which would confound the small-model
  thesis; it would need to be constrained to bug-identification and measured.
