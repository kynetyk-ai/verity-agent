# Verity `fe-kaggle` prototyping report — what six runs taught us

**Scope.** Six end-to-end `fe-kaggle` runs (stellar classification, `playground-series-s6e6`, metric =
**balanced accuracy**), recorded under [`prototyping-runs/`](prototyping-runs/), across three model
families and three agent time budgets. This report combines the **objective run data** (the harness's
own `results.json` RunReports — see the companion notebook
[`run_data_explorer.ipynb`](run_data_explorer.ipynb)) with a **qualitative comparison of the modeling
strategies** the agents actually used (read from every submission script). The system prompt and its
variants are in [Appendix A](#appendix-a--the-composed-system-prompt-and-its-variants).

## TL;DR

- **Every model plateaus at ~0.966 balanced accuracy and no run ever reached the top-10% bar (~0.972).**
  Best of all six runs: **0.9668** (fugu). The plateau is **model-agnostic** (gpt-5.4 0.9664, sonnet
  0.9667, fugu 0.9668 — a 0.0004 spread), which points to a **feature/data/compute ceiling**, not a
  model-capability gap.
- **The task is simultaneously too easy and too hard.** Too easy: every model reaches ~0.966 by cycle 0–1
  with the textbook GBDT-on-colors-and-redshift play. Too hard: the last ~0.005 to the bar is unreachable
  by *anyone*. There is almost **no usable gradient in between** — so the gate accepted **nothing**, spent
  **zero** Kaggle submissions, and the entire `accepted`/submission/calibration half of the lifecycle was
  **never exercised**.
- **The time budget is too long, not too short.** More time bought *elaboration and instability*, not
  accuracy: fugu 40→80→160 min moved the best score **+0.0003 for 4× the wall**, while **failures rose**
  and the 160-min run literally **spun** (three identical scripts). Shorter budgets would force focus and
  likely make the task more discriminating.
- **Timeouts/crashes are an agent time-management problem, and the warning lands too late.** The agents
  *expand to fill any budget* and don't self-budget; the soft wrap-up nudge fires at 80% (and can't land
  mid-training), and the **step budget has no graceful path at all** — it crashes with `GraphRecursionError`
  ([issue #103](https://github.com/kynetyk-ai/verity/issues/103)).

---

## 1. Setup

Each run is the same task driven for up to 10 cycles of **read → propose → gate → commit**. The gate is a
ladder: *runs-clean* (the script executes on a hold-out) → *proxy-improves* (balanced accuracy beats the
best **accepted** attempt) → *competitive* (calibrated estimate vs the **live top-10% bar**; below it →
`refine`, gap fed back, **no submission spent**) → *kaggle* (regenerate on full data, submit, accept iff
it beats our best public score). The agent never sees the hold-out; it only gets the gate's structured
gap feedback.

What varied across the six runs:

| run | model | agent budget (`sandbox_timeout_s`) | turn cap (`recursion_limit`) |
|---|---|---:|---:|
| 2026-06-21 gpt-5.4 | gpt-5.4 | 40 min | 200 |
| 2026-06-21 sonnet | claude-sonnet-4-6 | 40 min | 200 |
| 2026-06-22 sonnet+nudge | claude-sonnet-4-6 (+ parallel-library prompt nudge) | 40 min | 200 |
| 2026-06-23 fugu | Sakana fugu | 40 min | 200 |
| 2026-06-26 fugu 80min | Sakana fugu | 80 min | 300 |
| 2026-06-27 fugu 160min | Sakana fugu | 160 min | 300 |

---

## 2. Objective analysis

### 2.1 The plateau, and the bar nobody reached

| run | best proxy bal-acc | best at cycle | gap to bar | accepted | Kaggle submissions |
|---|---:|---:|---:|---:|---:|
| gpt-5.4 | 0.9664 | 7 | ~0.005 | 0 | 0 |
| sonnet | **0.9667** | 5 | ~0.005 | 0 | 0 |
| sonnet+nudge | 0.9658 | 1 | ~0.006 | 0 | 0 |
| fugu 40min | 0.9665 | 0 | ~0.005 | 0 | 0 |
| fugu 80min | 0.9666 | 4 | ~0.005 | 0 | 0 |
| fugu 160min | **0.9668** | 1 | ~0.005 | 0 | 0 |

Two facts dominate everything else:

1. **The whole field lands in a 0.0004-wide band (0.9664–0.9668)** regardless of model. A frontier model
   (gpt-5.4), a strong open-weights-style hosted model (sonnet), and a smaller third-party model (fugu)
   are *indistinguishable* on this task. When the model doesn't matter, the **task is measuring the
   feature/data ceiling, not the agent.**
2. **The best score almost always arrives in the first one or two cycles** (sonnet c5, sonnet+nudge c1,
   fugu-80 c4, fugu-160 c1) and then *never improves*. Across 8 scored gpt cycles the climb was **+0.0006
   total.** The remaining cycles buy architecture, not accuracy.

The gate held its discipline perfectly — because no calibrated estimate reached the bar, it withheld
every submission. That is the harness working **as designed**, but it also means the climb/accept/
self-calibrating-estimate machinery got **zero exercise** across all six runs.

### 2.2 Cost — the agent expands to fill the budget

Wall-clock per cycle scales with the budget it's *given*, not with the work the task *needs*: the 40-min
runs finished cycles in minutes-to-40, the 80-min run routinely used 70–77 min, and the 160-min run ran
cycles up to ~155 min — for the **same ~0.966 result**. The agents do not self-budget; given more runway
they explore more, not better.

### 2.3 Failures — two walls, and raising one just pushes you into the other

(See notebook §9 for the plots.) Sandbox failures by run and mode:

| run | budget | turn cap | timeout | recursion-limit crash | api-credit | total |
|---|---:|---:|---:|---:|---:|---:|
| gpt-5.4 | 40m | 200 | 0 | 0 | 0 | 1* |
| sonnet | 40m | 200 | 0 | 1 | 0 | 1 |
| sonnet+nudge | 40m | 200 | 0 | **4** | 0 | 4 |
| fugu 40min | 40m | 200 | 2 | 0 | 3 | 5 |
| fugu 80min | 80m | 300 | 2 | 0 | 0 | 2 |
| fugu 160min | 160m | 300 | 1 | **2** | 0 | 3 |

\* gpt's lone sandbox failure was a content-filter `invalid_prompt` (c1); it also lost cycles **gate-side**
to CatBoost crashes/hangs (c3, c9) that the table above doesn't count (those are gate rejects, not sandbox
failures).

The pattern is the headline finding on budgets:

- **At 40 min, the agent tends to hit the *step* wall first** (sonnet/sonnet+nudge: 200-turn
  `GraphRecursionError`) — or an external limit (fugu's credit exhaustion).
- **At 80/160 min, it runs long enough to hit the *time* wall** (timeouts).
- **At 160 min it hits *both*** — a timeout *and* recursion-limit crashes, because the longer runway lets
  the agent take enough steps to exhaust even the raised 300-turn cap.
- **Raising the turn cap 200 → 300 did not stop the recursion crashes** — it only delayed them. They occur
  at *both* caps. More time *increased* total failures; more turns just moved the wall.

---

## 3. Qualitative strategy comparison

Read across all 42 submission scripts, the agents converge on the **same playbook** and differ mostly in
*temperament* and *discipline*.

### 3.1 Common ground (every model, every run)

- **Gradient-boosted trees** (LightGBM / XGBoost / HistGradientBoosting; CatBoost attempted but hazardous —
  below). Nobody used anything exotic; the prompt steers hard toward CPU GBDTs.
- **Physics-literate features**: the 10 pairwise photometric **colors** (u-g … i-z), **redshift**
  transforms (log1p/sq/sqrt, negativity flags, binning), magnitude aggregates, cyclic sky coordinates,
  and the two native categoricals. All three models independently reason that *redshift separates QSO/STAR
  and colors separate the rest* — and gpt/fugu both build explicit redshift "physical-window" features.
- **Per-class decision multipliers** before `argmax`, because balanced accuracy is recall-sensitive — this
  is the single most-tuned lever in the whole corpus, and (see below) the one that actually moved scores.

### 3.2 The four strategy axes that distinguish the runs

**(a) Model families & the OpenMP/CatBoost hazard.** The biggest *infrastructure* story. CatBoost (its own
threading runtime) **co-loaded with LightGBM/XGBoost stalls at 0% CPU** — gpt's most ambitious cycle (c9,
a 6-model XGB+LGBM+CatBoost stack) **deadlocked**; sonnet's baseline hit it **twice** and *misdiagnosed it
as an OOM*, "fixing" it by dropping libraries on a self-invented "2 GB budget" theory. fugu's 40-min run
abandoned LightGBM entirely (→ HistGradientBoosting) to dodge the same stall. The agents' biggest losses
here were **engineering/infra, not modeling**.

**(b) Ensembling.** A shared escalation ladder — single model → uniform average → hand-weighted soft-vote
→ **K-fold OOF stacking** (a LogReg meta-learner). Notably, **the stacking step is where ambitious cycles
go to die**: gpt's 6-model stack deadlocked, sonnet+nudge's 2-level stack timed out (c8), fugu-80's stacking
attempt timed out (c7). The most sophisticated move repeatedly collides with the resource wall.

**(c) The decision rule — the one lever that mattered, and a clean budget experiment.** How the per-class
multipliers are *obtained* is the most revealing axis:
- *gpt-5.4*: implicit (sample weights) → a **bad hardcoded guess** (c5, regressed) → **OOF-tuned**
  coordinate-ascent (c8–c9). Matured correctly.
- *sonnet*: baseline used a **static inverse-prior division** (never tuned); the **nudge run upgraded to
  Nelder-Mead thresholds that directly maximize balanced accuracy on its own OOF** — a genuinely better,
  data-driven decision rule.
- *fugu, across budgets, is the cleanest natural experiment*: **only more time converted the decision rule
  from hardcoded constants to OOF-tuned, redshift-*segmented*, self-calibrating multipliers baked into the
  script.** The 40-min squeeze did the **opposite** — it *regressed* (OOF → single-split → hardcoded
  constants, with the best-scoring submission carrying the *least* principled rule). 80 min produced the
  standout invention (c6's in-script coordinate-descent self-calibration). 160 min made segmented OOF
  tuning the durable default.

**(d) Validation discipline.** Most runs **fly blind** for model selection — they train on full data and
lean on the harness's hold-out as their only feedback, never carving an internal validation set to
estimate the gate score before submitting. The exceptions track *budget and prompting*: the sonnet **nudge**
run and the fugu **80/160-min** runs build their own OOF CV and even fold-safe target encoding; the rushed
40-min fugu run does **no validation at all** by its last cycle.

### 3.3 Per-model character (one line each)

- **gpt-5.4 — the by-the-book journeyman.** Arrives knowing the GBDT playbook cold, climbs the recognized
  ladder by *accretion* (features only grow, ensembles only widen), repairs its own crashes — but never
  builds a holdout to check whether any of it helped, so it churns architecture while pinned at its cycle-0
  ceiling, and over-reaches into a deadlock at the top.
- **sonnet — the physics-literate maximalist.** Long, heavily-commented scripts, domain-named astro features
  (Lyman-break, stellar-locus), big ensembles — but prone to **confident misdiagnosis** (the "OOM" theory)
  and so iterative it routinely blows the 200-turn cap. The nudge made it a cleaner library citizen and a
  better *validator*, but **no more accurate and more failure-prone** (1 → 4 recursion crashes).
- **fugu — the rigor-seeking over-explorer.** The only model that *visibly matures its own methodology*
  (hardcoded → OOF-segmented self-calibration; leaky → fold-safe target encoding) — but it **cannot
  self-budget**: too little time crudifies it, the right amount produces its best work, too much funds
  elaboration-without-payoff (duplicate scripts, ever-finer segmentation) until it slams into the step wall.

---

## 4. Findings

### F1 — The competitive bar (top 10%) is too high. *Relax it.*

Across six runs and three models, **nobody ever reached it**, and the gap (~0.005) never closed. A bar
that is never cleared yields **no positive signal**: every cycle of every run is `revised`, the `accepted`
lifecycle never fires, and no Kaggle submission is ever spent — so we never test the climb, the public-score
calibration, or the accept path at all. **Recommendation:** lower the bar to something reachable (e.g.
top-25–40%, or a fixed target a strong attempt actually hits), so the harness produces a *documented climb*
and exercises its full lifecycle.

### F2 — The task is both too easy and too hard; the plateau is a ceiling, not a skill gap.

Every model hits ~0.966 in the **first cycle or two** with the obvious play (the "too easy" entry), and
**no model can pass ~0.9668** no matter how elaborate it gets (the "too hard" ceiling) — and crucially the
**plateau is model-agnostic** (0.0004 spread across gpt/sonnet/fugu). That is the signature of a
**feature/data/compute ceiling** for a single bounded CPU script on this dataset, not a measure of agent
capability. With almost no usable gradient between trivial-entry and unreachable-bar, the task **does not
discriminate** between models or strategies. **Recommendation:** either (a) relax the bar into the band
where the gradient exists (F1), and/or (b) move to a harder / more-discriminating dataset or a looser
compute envelope (GPU, larger time-on-real-signal) so there is genuine headroom to separate models.

### F3 — The time budget is too long, not too short. *Shorten it.*

The data is unambiguous: fugu **40 → 80 → 160 min** moved the best score **+0.0003 for 4× the wall**, and
the longer budgets produced **more failures** (§2.3) and outright **spinning** — the 160-min run emitted
**three byte-for-byte identical scripts (c4 ≡ c5 ≡ c7)**, the agent re-proposing the same submission rather
than progressing. More time funds over-engineering (ever-finer segmentation, doomed stacking ambitions),
not better models, because the **ceiling (F2) is reached almost immediately**. Shortening the budget would:
(i) cut wasted compute and the failures that grow with runtime; (ii) force the agents to *focus* (the rushed
fugu run was cruder but no worse on score); and (iii) make the task **more discriminating** — under a tight
budget, *time management and prioritization* become real differentiators, which helps F2. A budget in the
**20–30 min** range looks right to test next (and pairs naturally with shortening to address F1/F2's "too
easy to grind" problem).

### F4 — Timeouts/crashes reflect poor agent time management *and a warning that comes too late.*

The agents "expand to fill the budget" and don't pace themselves. Two compounding harness gaps make that
fatal rather than merely wasteful:
- The **wall-clock** soft nudge (`DeadlineMiddleware`) fires at **80%** of the budget and only *between
  model steps* — so it lands late, and **can't land at all** when the agent is inside a long full-data
  training call as the wall arrives (exactly when it matters).
- The **step budget has no graceful path whatsoever** — exhausting `recursion_limit` throws an uncaught
  `GraphRecursionError`, killing the worker with **no proposal and no warning**. This is
  **[issue #103](https://github.com/kynetyk-ai/verity/issues/103)**, filed from these runs: wire a
  step-budget wrap-up nudge (the `StepBudgetMiddleware` that already exists but isn't enabled) **and**
  harvest any in-progress proposal on the error, mirroring what the time budget already does. **An earlier
  nudge (e.g. 60–70%) plus #103 is the concrete fix** — and it matters more as budgets shrink (F3), since a
  tighter budget makes good time-management the whole game.

### F5 — (of note) The accept lifecycle was never exercised.

Because nothing cleared the bar (F1), `accepted` status, real Kaggle submission, the self-calibrating
proxy→public estimator, and the daily-cap blocking path got **zero coverage** across all six runs. We have
validated the *revise* loop thoroughly and the *accept* loop **not at all**. Relaxing the bar (F1) is also
what would finally test that half of the system.

### F6 — (of note) The recency-incumbent regression stranded the peak (now fixed).

In every all-`revised` run the agent built on its **newest** attempt, not its **best** — so the 160-min
run's 0.9668 peak (c1) was orphaned and scores **drifted down** (0.9668 → 0.9657 → 0.9656 → 0.9652), and
the duplicate-script spinning (F3) was abetted by it. This is
**[issue #95](https://github.com/kynetyk-ai/verity/issues/95)**, **already fixed and merged**
(`BEST_REVISED_OR_ACCEPTED`, now the fe-kaggle default) — future runs will build on their best. (Related
housekeeping: **[#96](https://github.com/kynetyk-ai/verity/issues/96)**, idle verifier containers
accumulating.)

### F7 — (of note, and a positive) The adversarial verifier did its job.

Multiple runs show the agent **self-reporting a higher score than the gate measured** (e.g. fugu's
self-reported ~0.967 vs gated 0.9665 — the tell of multipliers tuned on its own split). The independent
gate caught the optimism every time and kept the reported numbers honest. The harness's core thesis held
even though the *task* didn't crack the bar.

---

## 5. Recommendations (next-phase task config)

1. **Lower the acceptance bar** into the reachable band (F1) — top-25–40% or a fixed achievable target — so
   the climb is observable and the accept/submission/calibration lifecycle is exercised.
2. **Shorten the agent budget to ~20–30 min** (F3) — less waste, fewer runtime-correlated failures, more
   discriminating.
3. **Ship the step-budget graceful path (#103)** and **move the soft nudge earlier** (~60–70%) (F4).
4. **For real model-discrimination, change the problem, not the agent** (F2): a harder/larger-signal dataset
   or a looser compute envelope, since this dataset's single-CPU-script ceiling is ~0.9668 for everyone.
5. **Keep #95's fix** (done) and reduce/repair the recursion-limit handling rather than just raising the cap
   (raising 200→300 didn't help).
6. **Re-run the cleanest comparison** (one model, several seeds) under the new config to get the ablation the
   plateau prevented here.

---

## Appendix A — the composed system prompt and its variants

The control plane assembles the system prompt **mechanically in three layers** (`compose_system_prompt`,
`control_plane/config.py`): an **invariant kernel orientation** (rendered from the workspace contract), the
**domain instructions** (the FE layer), and the **task instructions** (the per-task goal). The kernel +
domain layers are the slow-changing, prompt-cached prefix.

### A.1 Layer 1 — Kernel orientation (invariant, all runs)

```
# Kernel orientation (invariant)
You operate a single repeating loop: read -> propose -> gate -> commit.
You PROPOSE artifacts; you never write durable state, and you never contact the verifier.
A proposal is a new artifact plus the operation that produced it, with a short note on how and
why you produced it.

Your workspace has a fixed layout (read-only inputs; writable-ephemeral working areas):
  data/     (read-only)            the task's data sources (read-only, §3.4)
  context/  (read-only)            skills, SOPs, gold-standard examples
  tools/    (read-only)            the task's registered tools (§8.2)
  spec/     (read-only)            the proposal-shape spec / output schema (§3.4)
  scratch/  (writable-ephemeral)   the agent's working space
  outbox/   (writable-ephemeral)   proposal + object attachments; harvest source

Write your proposal and any object attachments (a script, a data file) to outbox/. The harness
harvests the outbox; do not rely on any other path. The output schema lives under spec/.
Put ONLY your proposal's declared objects in outbox/ — nothing else; the harness keeps exactly
those and discards any other file you leave there.

Two corrections may come back:
- A shape-error means the proposal is malformed (wrong parts/types). Fix the formatting and
  resubmit; nothing is recorded.
- Refine feedback means the proposal is mostly sound but carries named defects. Produce a
  tracked revision that addresses exactly those defects.
```

### A.2 Layer 2 — Domain instructions (the `fe-kaggle` FE layer)

Verbatim from `domains/feature_engineering.py` (`FEATURE_ENGINEERING_INSTRUCTIONS`). `{TRAIN_INPUT}`,
`{TEST_INPUT}`, `{PREDICTIONS_OUTPUT}`, `{ENTRYPOINT}`, `{REQUIREMENTS}` are filled per task.

```
# Domain instructions
You are a discovery agent on a tabular prediction task. Each cycle you produce ONE submission: a
self-contained Python script that reads the data, builds a predictive pipeline, and writes a
prediction for every test row. Any technique that fits in a single script and improves the held-out
score is fair game — engineered features, the model you pick, an ensemble, calibration, handling
class imbalance. There is no mandated method. Your aim across cycles is to climb into the TOP TIER
of the real leaderboard: each cycle either improves on your best or is sent back to revise toward a
competitive target (how that works is below).

The script contract (the gate runs your script; honour it exactly):
- Resolve the data directory as os.environ.get("VERITY_DATA", "data") and the output directory as
  os.environ.get("VERITY_OUT", "out"). (The defaults work in your sandbox; the gate sets these.)
- Read the labelled training data from <VERITY_DATA>/{TRAIN_INPUT} and the unlabelled rows to
  predict from <VERITY_DATA>/{TEST_INPUT}. The target is `class`; {TEST_INPUT} has no `class`.
- Clean the data, build your pipeline, train on the training data, then predict a `class` for every
  row of {TEST_INPUT}.
- Write predictions to <VERITY_OUT>/{PREDICTIONS_OUTPUT} with exactly two columns: id,class.
- The script must be deterministic (fix every random seed) and self-contained.

The runner (where the gate executes your script): a CPU-only Linux container, Python 3.12, with your
pinned {REQUIREMENTS} pip-installed (network on for the install). The common system libraries for
the CPU ML stack are present, so scikit-learn, LightGBM, XGBoost, pandas, and numpy all work.
Budget: a few minutes, ~2 GB RAM, ~1 GB scratch. So: prefer fast, wheel-installable CPU libraries
and a bounded model; do NOT use deep-learning frameworks (torch / tensorflow will not fit or
finish), and avoid libraries with no prebuilt wheel (there is no compiler in the runner).

Orient before you propose:
- If there is NO incumbent yet, EXPLORE THE DATA WITH CODE first — do not try to read the raw files,
  they are too large. Write a short script that loads the data and prints its shape, column dtypes,
  the target's class balance, summary statistics, missingness, and a few candidate signals
  (correlations or simple per-class means). Let what you find drive your first submission.
- If you have prior work, it is provided under scratch/provided/ (see its INDEX.md): your best
  accepted submission, or — on a REVISE cycle — the exact script you were just asked to revise.
  Read it, work out WHY it scores as it does and where it is WEAK, then make a focused change that
  targets that weakness rather than starting from scratch. You may edit your workspace freely.

What to deliver to outbox/ each cycle:
- {ENTRYPOINT} — the script above.
- {REQUIREMENTS} — the pinned package list your script needs (e.g. pandas==2.2.2), one per line.
  The gate installs exactly these before running your script, so pin versions for reproducibility.
- The proposal payload (via your submit/revises tool): entrypoint = "{ENTRYPOINT}" and
  requirements = "{REQUIREMENTS}". The submission is the unit — you do not report individual
  features or changes.

How you are judged (you never see the judge's data):
- Your goal is the TOP TIER of the real leaderboard. The gate runs your script on data you cannot
  see (a reserved hold-out) and scores BALANCED ACCURACY, then estimates whether that score would
  reach the competitive bar — a target read live from the leaderboard. If it would NOT, your
  submission is sent back to REVISE: you are told your estimated score, the target, and the gap —
  close it and resubmit. No real leaderboard submission is spent until you are competitive, so keep
  improving the held-out score.
- Once the estimate clears the bar, the script is regenerated on the full data and submitted to the
  REAL leaderboard; it is accepted only if its public score beats your best so far. Each accepted
  submission is a real step up the leaderboard — keep climbing across cycles.
- Anything that peeks at the target looks great on your own split but fails on the hold-out and the
  real data. A submission that errors, times out, or fails to predict every row is rejected
  outright; there is no partial credit. Confirm your script runs cleanly before you submit.

Data + speed (the amount of training data is the real lever here — use ALL of it):
- TRAIN ON THE FULL training data provided; do NOT subsample it. With balanced accuracy on this
  problem the quantity of training data is the dominant driver of the score, and a gradient-boosted
  tree model (LightGBM / XGBoost / CatBoost) fits the whole set in a couple of minutes — comfortably
  within the time budget.
- USE ALL THE CPU CORES. The runner has ~16 cores — set n_jobs=-1 (scikit-learn / XGBoost /
  LightGBM) and thread_count=-1 (CatBoost) so training is parallel. A single-threaded fit (the
  library default for some estimators) on the full data WILL time out — this is the most common
  reason a good model fails to finish.
- The runner is an automated, isolated sandbox — not your interactive workspace, and with tighter
  limits. Some libraries that manage their own thread pools do not share one process cleanly —
  CatBoost loaded alongside LightGBM or XGBoost is a known case — and the run can STALL (it stops
  making progress and the gate eventually times it out). If you want several such models, train each
  in its own process (multiprocessing / ProcessPoolExecutor) and combine their predictions,
  rather than co-loading them in one process.
- Optimize for BALANCED accuracy, not raw accuracy: validate with stratified cross-validation and
  handle the class imbalance (class weights / resampling / threshold tuning).
- The gate runs your script under a generous time budget, but a model that does not finish scores
  nothing — keep any ensemble or search bounded. To check it runs without burning your own budget,
  test on a small sample (e.g. a few thousand rows); the gate trains the real thing on the full
  data. Then submit.

Useful domain knowledge: differences between photometric bands ("colour indices", e.g. u-g, g-r,
r-i, i-z) and redshift carry most of the signal; encode the categorical spectral_type and the
galaxy_population column (both are present in train AND test — explore the columns to see exactly
what is available); and because scoring is balanced accuracy, handle class imbalance (class weights
/ resampling).
```

### A.3 Layer 3 — Task instructions (per-run goal)

```
# Task instructions
Reach the top 10% of the real Kaggle leaderboard for the stellar competition, improving across cycles.
```

### A.4 Variants

1. **Parallel-library nudge** *(added 2026-06-22; present in sonnet+nudge and all fugu runs; absent in the
   2026-06-21 gpt-5.4 and sonnet baselines).* The domain-layer block headed *"USE ALL THE CPU CORES"* plus
   the paragraph warning that thread-pool-managing libraries (**CatBoost co-loaded with LightGBM/XGBoost**)
   can stall, advising separate-process training. Added in response to the OpenMP/CatBoost hangs in the
   baselines. *Observed effect (§3.2a):* agents **avoided CatBoost** rather than implementing process
   isolation — it removed the CatBoost stalls but did not reduce overall failures (sonnet's recursion
   crashes rose 1 → 4), and one A/B pair cannot separate the nudge from run-to-run variance.

2. **Soft deadline nudge** *(runtime-injected, all runs; `DeadlineMiddleware`).* At **80%** of the agent's
   wall-clock budget, between model steps, one message is injected:
   > *"Time budget almost spent: ~{remaining}s of {total}s left. Stop exploring now — finalize your script,
   > run it once to confirm it works, and submit your best proposal immediately."*

   It fires late and cannot land mid-training-call (F4). There is **no analogous nudge for the step/turn
   budget** — that limit crashes ungracefully (issue #103).

3. **Refine feedback** *(per-cycle, from the gate, into the next cycle's context).* The structured gap that
   drives the loop, e.g.:
   > *"below the top-10% bar: estimated public score 0.9660 vs target 0.9718 (gap 0.0059); keep improving
   > the held-out score before we spend a submission."*

*(Per-run task configs that set `sandbox_timeout_s` / `recursion_limit` are in `task.gpt.json`,
`task.sonnet.json`, `task.fugu.json`; the values are tabulated in §1.)*
