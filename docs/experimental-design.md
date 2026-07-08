# Verity — experimental design

> Detailed experimental plan for the practical paper. Companion to
> [paper-outline.md](paper-outline.md) (expands its §6 evaluation sketch into a runnable design) and
> [related-work.md](related-work.md) (positioning). Status: **design locked for Exp 1–5; Exp 6 and
> all compute/length budgets are placeholders** pending prototyping experience.

## 0. What we are testing (and what we are not)

The paper's thesis is that unattended self-improvement needs two things most agent loops lack:

1. an **independent, unfudgeable success signal** (`proposer ≠ gate`), and
2. **structural context hygiene** (failed paths recorded as provenance, never injected into the next
   working context).

The experiments are built to **isolate each mechanism separately** and show each contributes, then to
show the assembled system (a) lifts a weak model disproportionately on a real external benchmark and
(b) extends to a *fuzzy* knowledge task where the verifier enforces methodology and veracity. We are
**not** trying to win Kaggle or beat a SOTA model; the contribution is the harness and its discipline.

Two claims map to two measurable deltas in the ablation ladder below:

- **Claim A (the gate helps):** adding an independent reject-capable gate improves outcomes →
  measured by **Exp 3 − Exp 2**.
- **Claim B (hygiene helps):** excluding failed paths from context improves outcomes, *holding the
  gate fixed* → measured by **Exp 4 − Exp 3**.
- **Claim B′ (selection helps):** building on the *best* prior rather than *all* good priors improves
  outcomes → measured by **Exp 4b − Exp 4** (the #95 best-vs-recency selection effect).

---

## 1. The ablation ladder (Exp 1 → 4b) — the core study

**Shared task:** stellar-class prediction on `playground-series-s6e6`, metric **balanced accuracy** —
the same dataset as the Kaggle capstone (Exp 5), so the ladder (local proxy) escalates naturally into
Exp 5 (real leaderboard) on one task.

**Shared scoring:** the **code runner scores a held-out split locally** — **no Kaggle submission, no
competitive bar**. The verifier stack is truncated to `runs-clean → holdout-score` (the original §12
FE holdout scorer, not the full fe-kaggle ladder). The held-out labels live verifier-side (answer-key
isolation by construction, ADR 0005); the agent never sees them.

**Shared data (test-set sizing).** Prepared once outside Verity (`tools/prepare_fe_data.py`, ADR
0005) and passed to every cell. The agent's role gets `train.csv` plus a **small unlabelled `test.csv`
sample** (default **200 rows**, `--agent-test-rows`) — a *wiring/format check only* (does the script
run and write a well-formed predictions file over the real schema?). It is unscoreable (no labels),
so the agent estimates balanced accuracy via **stratified cross-validation on the training data**. The
verifier role holds the full `train.csv` + the reserved `holdout.csv` and its answer-key
`holdout_labels.csv`, and the gate scores the agent's script on that **full** reserved hold-out. The
small sample is the deliberate decision: it verifies wiring as well as the full set at a fraction of
the cost and keeps a capable model from over-fitting to a peek at the test schema. Identical across
all conditions, so it never confounds a rung.

**Shared models:** `sonnet`, `gpt-5.4-mini` (`openai:gpt-5.4-mini`), local `qwen`. **N = 10 runs per cell.**
→ 5 conditions × 3 models × 10 = **150 ablation runs**.

Each rung adds exactly **one** mechanism over the rung below it:

| # | Name | Loop | Gate | Provisioning mode | Isolates |
|---|---|---|---|---|---|
| 1 | One-shot calibration | no (1 cycle) | score-and-**always-accept** | `none` | per-model **headroom** |
| 2 | Unguarded loop | yes | score-and-**always-accept** | `all` | does naive iteration help? |
| 3 | Guarded, all context | yes | **accept iff runnable ∧ beats best prior; else reject** | `all` | **Claim A** (the gate) |
| 4 | Guarded, curated | yes | (same as 3) | `all_revised_or_accepted` | **Claim B** (hygiene) |
| 4b | Guarded, curated, best | yes | (same as 3) | `best_revised_or_accepted` | **Claim B′** (selection) |

Notes on the conditions:

- **Even the "always-accept" rungs use a *declared* gate** (a permissive one that runs the holdout
  scorer, records the score, and accepts unconditionally). This respects "no implicit accept" — the
  difference across rungs is the gate's *verdict policy*, never the absence of a gate.
- **The 3/4/4b gate rejects, it does not `refine`.** A `refine` marks an artifact `revised`, which the
  `accept-revise` provisioning modes *include* — that would leak failures back into Exp 4 and erase
  Claim B. Binary accept/reject keeps rejected failures out of the curated context. (Because nothing
  is refined here, `all_revised_or_accepted` ≡ `all_accepted` and `best_revised_or_accepted` ≡
  `best_accepted` for these runs; we use the `_revise` forms for forward-compatibility with Exp 5,
  which *does* refine.)
- **"Beats best prior"** compares the candidate's holdout score against the **best prior attempt of
  any status** (not best *accepted*) — the #95 "gate against best prior" mechanism. This is required:
  in an all-`revised`/`rejected` run there is no other ratchet.

### 1.1 Instruction text (held constant within the loop rungs)

- **Exp 1 (one-shot):** *"Write a script that optimizes stellar-class prediction for balanced
  accuracy. Submit your best work."*
- **Exp 2 / 3 / 4 / 4b (loop):** *byte-identical* across all four — *"Write a script that optimizes
  stellar-class prediction for balanced accuracy. Your submission should improve upon any previous
  submissions."* The manipulation across 2→4b is **only** gate + provisioning, never wording.

### 1.2 The decomposition logic

- **2 − 1:** value of naive looping alone (the strawman everyone assumes works). May be small or even
  negative — that's a *result*, not a failure.
- **3 − 2:** Claim A. The independent gate's value, holding context (`all`) and budget constant.
- **4 − 3:** Claim B. Context hygiene's value, holding the gate constant. (This bundles two
  sub-effects — *excluding failures from view* and *building on the good set*; 4b decomposes the
  second.)
- **4b − 4:** Claim B′. Building on the **best** good prior vs **all** good priors.

### 1.3 Equal-compute control

Rungs 2/3/4/4b run the **same cycle count and the same per-cycle wall/token budget** (values are a
*placeholder*, §5), so deltas reflect mechanism, not attempt count. Exp 1 is the single-shot baseline
by design and is not compute-matched to the loops.

---

## 2. Exp 5 — FE-Kaggle capstone (punch above its weight)

The assembled system on the **real external leaderboard**, configured from the ablation findings
(provisioning = `best_revised_or_accepted`, the full `runs-clean → proxy → competitive → kaggle` gate
ladder, refine enabled). The headline 2×2:

|  | single-shot (cycle 1) | after-loop (best incumbent) |
|---|---|---|
| **small local model (Qwen)** | a | b |
| **large hosted model** | c | d |

**Paper-making result:** `b − a ≫ d − c` (the loop lifts the weak model disproportionately), ideally
`b ≈ c` (small-in-loop ≈ large-single-shot). **Test-set hygiene** (paper-outline §6.3): gate on the
verifier hold-out, **report the headline number on the truly-held-out public leaderboard** (never fed
back).

---

## 3. Exp 6 — second domain: explainable knowledge work (methodology + veracity verifier)

The generalization evidence **and** the showcase for the inspectable-criteria thesis where it matters
most: a **fuzzy** task with no objective metric, graded by a verifier that enforces **methodology and
veracity** rather than a number. This is the experiment that converts "the harness is domain-agnostic"
from an architectural claim into a demonstrated one, and that operationalizes §4 (an explicit,
auditable criterion replacing the model's private stopping rule).

**Candidate task:** competitive landscaping — *"identify the relevant competitors for X."* Each run
produces a **ranked list with backing artifacts** (sources/snapshots per claim) for fact-checking.

**Verifier (LLM-as-judge enforcing a written rubric):**
- **Methodology** — did the agent follow the declared process (coverage of sources, search breadth,
  de-duplication, inclusion criteria)?
- **Veracity** — is each listed competitor backed by a cited artifact, and does the artifact actually
  support the claim? (The verifier re-checks artifacts, not the agent's say-so — the analog of the FE
  gate re-running the *script* on gold rather than trusting the agent's CSV.)

**Build implications (largest of the program):** a new sandbox with a **headless-browser tool**, a new
**LLM-judge verifier type**, a new task type whose output shape is *a list + per-item backing
artifacts*. **Design is a placeholder** pending the Exp 1–5 results and a sandbox-tooling spike.

**Honest risk:** Exp 6 is the *only* second-domain evidence and the least-built; it carries the
generalization claim alone. Mitigation: keep it a real experiment (not a vignette), and front-load the
browser-sandbox + judge-verifier spike so slippage is visible early.

---

## 4. Metrics, instrumentation, and figures

All metrics derive from the store / the store-derived RunReport (provenance is the source of truth).

**Primary**
- **Final quality** — best held-out balanced accuracy per run (distribution over the 10 seeds).
- **Best-so-far trajectory** — per-cycle best held-out score (loops only); mean ± band over seeds.

**Mechanism (the figures that explain *why*, not just *that*)**
- **Context growth over cycles** — served-context size per cycle. Expectation: `all` rungs (2, 3)
  balloon; curated rungs (4, 4b) stay bounded. The direct evidence for hygiene.
- **Self-reported vs independent divergence ("fudge")** — the agent reports its own estimated balanced
  accuracy (added to the instruction as a measurement instrument); we compare it to the independent
  holdout score. Largest in Exp 2 (no gate to check it). **Gate catch-rate** = fraction of
  self-claimed "improvements" the Exp 3 gate rejects as non-improvements vs the same population in
  Exp 2 — the "agent fools itself, gate catches it" thesis, finally *measured* (paper-outline §7 has
  this only as anecdote).
- **Regression incidence** — does best-so-far ever drop? Expected: impossible under reject-gate +
  `best` provisioning (4b); possible under the unguarded loop (2).
- **Compute / tokens to a quality threshold.**

**Step-level instrumentation (pre-ablation audit, workstream F)**
- Every cycle's agent **transcript** is captured — written by the sandbox driver each cycle (success
  *and* failure, including a recursion/step-budget limit-end), content-addressed into the object
  store, and referenced from the cycle's RunReport entry as `transcript_ref` (a content hash; read it
  from the per-task object store, `…/tasks/<task_id>/objects/<hash>`, and render it with
  `tools/render_transcript.py`). Each entry records, per message, an index, the type/content, and tool
  calls with their **arguments** (large fields capped). This is the step-by-step record for
  qualitative analysis — previously only a no-proposal failure diagnostic, now always-on for both the
  in-process and container sandboxes.

**Figure catalogue**
- **F1** (Exp 1) — one-shot score distribution by model, with a headroom annotation (gap to a
  reference best). *The calibration gate: do not advance a model with no room to improve.*
- **F2** (Exp 2–4b) — best-so-far trajectory bands, faceted by model, one line per condition.
- **F3** (money plot) — final score by condition × model, showing the 1→2→3→4→4b deltas (Claims A/B/B′).
- **F4** — context size vs cycle by condition (bounded vs ballooning).
- **F5** — self-vs-independent divergence by condition (the fudge result).
- **F6** (Exp 5) — the 2×2 punch-above-weight bars (`b−a` vs `d−c`).
- **F7** (Exp 6) — one qualitative arc + the methodology/veracity rubric scores.

---

## 5. Compute & length budgets — **PLACEHOLDER**

Per-cycle wall-clock cap, per-cycle token cap, and max cycles per loop run to be set from current
prototyping experience (not yet complete). Constraints to honor when filled in: (a) identical budget
across Exp 2/3/4/4b; (b) feasible to run 150 ablation runs + replicates; (c) Qwen-local seedable for
replicates, hosted models acknowledged non-deterministic.

---

## 6. Build vs. configure

The experimental variation lives on **two axes that already map to two existing config surfaces**, so
most of the program is *configuration* over what exists. The only genuinely new core piece for the
whole ablation ladder is a small, purpose-built verifier (§6.2).

- **Provisioning** (`none` / `all` / `all_revised_or_accepted` / `best_revised_or_accepted`) →
  `TaskConfig.object_provisioning`. **Already built** (#95). Pure config.
- **Verdict policy** (always-accept vs improve-or-reject) → the **verifier**. The one new piece — and
  it has only **two** values across all five conditions.

So the entire custom-code surface for Exp 1–4b is *two verdict policies over a shared holdout scorer*;
everything else (model selection via the OpenAI-compatible seam, loop-vs-one-shot via cycle budget +
`--stop-on-accept`, instruction text, the sandbox, the `SUBMISSION` output shape) is already-built
config.

### 6.1 The build-vs-configure ledger

| Item | Build / Config | Size | Notes |
|---|---|---|---|
| Provisioning modes | **built** | — | #95, done — the ablation knobs |
| Model / provider selection | config | — | OpenAI-compatible seam, 4 providers proven |
| Loop / one-shot, budgets, instructions | config | — | `TaskConfig` + orchestration policy |
| Sandbox (Exp 1–5) | **reuse** | — | existing Deep Agents container; no new sandbox |
| **`holdout-experiment` verifier** | **build** | **S–M** | reuses §12 FE scorer; 2 policies; + best-prior baseline (§6.2) |
| Experiment / sweep orchestrator | build | **M** | N seeds × M models × K conditions over the async API |
| Analysis + plotting + context-size instrumentation | build | **M** | F1–F5 from the RunReport; self-vs-independent parse |
| Exp 5 (fe-kaggle capstone) | mostly config | **S** | existing system, tuned + 2×2 driver + public-LB reporting |
| Exp 6 sandbox (headless browser) | **build** | **L** | new sandbox tool |
| Exp 6 LLM-judge verifier (methodology + veracity) | **build** | **L** | new verifier type + rubric + artifact re-checking |
| Exp 6 task type (list + backing artifacts) | **build** | **M** | new output shape |

**Critical path for Exp 1–4b:** the whole quantitative ladder is unblocked by **two modest builds** —
the `holdout-experiment` verifier (S–M) and the sweep orchestrator + analysis tooling (M). Everything
else on that ladder is config. **Exp 6 is a separate, large track** (new sandbox + judge verifier +
task type) and is the only part that needs a different *kind* of sandbox and verifier.

### 6.2 The `holdout-experiment` verifier (the one core build)

A new `verity.verifier_types` plugin, purpose-built for controlled, deterministic, local, seedable
experimentation (the §6.1/outline requirement) — kept **distinct from the production fe-kaggle gate
stack** so experiments neither perturb nor depend on it, and so each condition is a declarative config
diff (a reproducibility win). It is also the natural first prototype of the declarative-verifier epic
(#97). It **reuses the existing §12 FE holdout scorer** (run the agent's script on the reserved split →
balanced accuracy) and rides the **existing verifier image** (needs only sklearn, already present — no
`kaggle` extra); "build" here = one plugin module + an entry point, **no new container**.

Declarative policy surface:

```
score:         holdout balanced-accuracy        # reused from the §12 FE verifier, fixed
runnable:      required                          # the runs-clean precondition (existing primitive)
accept_policy: always | improve_over_best_prior  # the only verdict axis; 2 values cover all 5 conditions
baseline:      best_prior_any_status             # for improve_over_best_prior
margin:        0.0                               # what counts as an improvement
```

Condition → policy mapping:

- **Exp 1, 2:** `accept_policy = always` (score is recorded, the artifact commits unconditionally).
- **Exp 3, 4, 4b:** `accept_policy = improve_over_best_prior` → **accept iff** `runnable ∧
  score > best_prior + margin`, **else reject** (binary; never `refine`, so failures stay `rejected`
  and out of the curated context).

The **only non-trivial new logic** is `improve_over_best_prior` reading the prior best holdout score
from the declared store-slice. The plumbing already exists — fe-kaggle's `proxy-improves` already
receives an incumbents slice — we change the **baseline from best *accepted* to best *prior of any
status*** (the verifier-side #95 mechanism, earlier scoped out of the code change; the experiments now
require it). This stays control-plane-generic: it reads a recorded score, not verifier-internal state.

### 6.3 Supporting builds

- **Experiment / sweep orchestrator (M)** — drives N seeds × M models × K conditions, each a Verity
  task via the existing async create→run→poll→export API; collects metadata. (`tools/` or
  `experiments/`.) Conditions are expressed as `(verifier policy) × (provisioning mode)` config, not
  code.
- **Analysis + plotting + context-size instrumentation (M)** — per-cycle score extraction from the
  store-derived RunReport, self-vs-independent divergence (parse the agent's reported estimate),
  served-context size per cycle, cross-condition aggregation, figures F1–F5.

---

## 7. Threats to validity

- **Single domain for the quantitative ladder.** Exp 1–5 are all the stellar DS task; generalization
  rests on Exp 6. Stated and accepted; mitigation is to resource Exp 6 as a real experiment.
- **Construct validity of the conditions.** The "always-accept" and "reject" gates must be *fair*
  representations of an unguarded vs guarded loop, not a hobbled strawman — document each precisely so
  a reviewer can judge fairness.
- **LLM-judge reliability (Exp 6).** The methodology/veracity verifier is itself a model; mitigate by
  re-checking artifacts (not the agent's claims) and by reporting judge agreement / spot human audits.
- **Non-determinism.** Hosted models and (Exp 5) a live leaderboard; mitigate with seeded replicates
  where possible, acknowledge where not.
- **Scoring noise → the bar-ratchet (pre-ablation audit, C/V2).** The holdout scorer trains
  multi-threaded (`n_jobs=-1`), so an identical submission's balanced accuracy wobbles by ~ε at the
  bit level (only boundary-row argmax flips). Under `improve_over_best_prior` (strict `score > best`)
  that wobble can *manufacture* a spurious "improvement" or *block* real progress, distorting the
  consistent-improvement signal and the gate-catch/regression metrics. **Mitigation:** the selection
  gate's `margin` (a noise floor; an improvement must exceed it) is wired as a shared knob across the
  guarded conditions (`Budgets.selection_margin` in `experiments/ablation/sweep.py`). We do **not**
  pin single-thread determinism — full-data single-core training is prohibitively slow and would
  force a script-contract change. **Pre-flight:** measure ε once (run a fixed submission ~5× through
  the real holdout coderunner — the `docker`-marked path in `tests/test_feature_engineering_gates.py`
  is the substrate — and look at the balanced-accuracy spread) and set `selection_margin` to ~2–3×ε.
  **Sizing update (2026-06-30):** the dataset was resized to an 18K stratified hold-out
  (`data/ablation-120k-seed42`), which drops the *sampling* component of ε to ≈ **0.002** (one
  boundary-row flip ≈ 1/18000·class-weight) — down from ≈ 0.02 at the old 45-row smoke. So the
  hold-out is no longer the dominant noise source; what remains is the *training-nondeterminism*
  component (the `n_jobs=-1` thread races), still to be quantified by the 5× rerun above. **The
  bottom rung (exp1) is `always`-accept so `margin` is irrelevant there; pin `selection_margin` for
  the loop rungs (exp2–4b) from that 5× measurement before launching them.** If ε proves large, fall
  back to scoring each submission N× and averaging.
- **Goal wording differs between the bottom rung and the loop (X1).** `spec.exp1.json` ends "Submit
  your best work" while `spec.loop.json` ends "improve upon any previous submissions" — intentional
  (exp1 is single-cycle, so "improve on previous" is inapplicable) and the layer-2/3 instruction text
  is otherwise identical across conditions. We deliberately **do not** unify it; flagged here so the
  one-clause difference is on record (run all conditions from `spec.example.json` if a single goal is
  preferred). Dependency drift across days is recorded, not pinned (a closed wheelhouse would reject
  legitimate submissions); see the deferred-items note in ROADMAP.
- **N.** 10/cell makes the ablation deltas estimable; report distributions + nonparametric effect
  sizes, pre-register 3−2 and 4−3 as the primary comparisons.
- **Sandbox-image leakage (found 2026-07-07; #137).** Until the fix, `Dockerfile.sandbox` baked the
  whole repo into `/app` — including `data/*/verifier/holdout_labels.csv` (the answer key) and
  `results/` (prior accepted submissions for the same task). The gate's measurement was never at
  risk (the code-runner image carries no repo), but a sandbox agent *could* have read the key.
  **Audit of every recorded batch (~340 cycles): zero answer-key access; one benign features-only
  read of `holdout.csv`; one directory listing.** Consequence: the 2026-07-05..07 gpt-5.4-mini
  batches (`results/exp1-gpt5mini-120k-v2/`, `results/loop-gpt5mini-120k-exp2/`,
  `results/loop-gpt5mini-120k-exp4b/`) are **re-designated smoke/calibration** — believed sound but
  not publishable ladder data. Fixed by construction at the image boundary
  (`Dockerfile.sandbox.dockerignore` + `tests/test_sandbox_image_purity.py`).
- **Pre-registered environment change between calibration and the real ladder (#136/#138).** The
  calibration transcripts motivated an affordance package applied 2026-07-07 — pip seeded into the
  sandbox venv, the workspace map naming `scratch/provided/` + INDEX.md, an unambiguous outbox
  gloss, the testing-regime split, and the runner budget stated in the instructions (workspace
  contract v2; pins in `tests/test_prompt_freeze.py`). All rungs of the real ladder run on this
  one surface; the calibration batches are not comparable to it and are not pooled with it.

---

## 8. Open placeholders (to fill before running)

- Compute/length budgets (§5) — from prototyping.
- The "large hosted model" for Exp 5 (the `gpt-5.4-mini` id is pinned: `openai:gpt-5.4-mini`).
- Exp 5 tuned configuration — finalized from the Exp 1–4b findings.
- Exp 6 full design — task scope, rubric text, sandbox tooling, output schema.
- ~~**Measured ε + chosen `selection_margin`** (§7)~~ — **measured 2026-07-05**
  (`results/epsilon-calibration/`): 5× reruns of a fixed submission through the real code runner,
  on both an sklearn (HistGB+LogReg) and a LightGBM+XGBoost stack, were **bit-identical** →
  training-nondeterminism ε = 0.0 on the study host (fixed seeds + the 16-cpu cap). Loop-rung
  **`selection_margin` pinned at 0.0**, justified by measurement; the knob stays wired should a
  future host measure ε > 0.
