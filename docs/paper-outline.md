# Verity — practical implementation paper: working outline

> Status: working notes. Captures the thinking from the design discussion during the 2026-06-26
> fugu fe-kaggle run. This is an outline + rationale, not a draft. It is **not** a theory paper —
> it is a practical / artifact / experience paper about a working system.

## 1. Thesis (one sentence)

Autonomous, unattended self-improvement needs two things most agent loops lack — an **independent,
unfudgeable success signal** and **structural context hygiene** — and Verity is a working system that
provides both by maintaining agent state as *typed provenance behind a verifier-aware commit lifecycle*
(git + a type system + a verifier-aware lifecycle for agent artifacts).

The contribution is the **harness and its discipline**, not a state-of-the-art result on any one task.

### Contribution type: an integration / systems paper (state this explicitly)

The literature scan (§9, [related-work.md](related-work.md)) makes one thing clear and we should own
it: **every building block already exists** — independent verifiers (Cobbe; Lightman), the empirical
case that self-grading fails (Huang et al.) and that irrelevant context hurts (Shi et al.;
Lost-in-the-Middle), typed/structured memory (CoALA; PROV-DM), and verifier-gated improvement loops
(FunSearch; SWE-bench; AIDE). **None of these is our contribution.** The contribution is the
**assembly** — putting them together into something that *works, runs unattended, and is composable*,
which is a first-class systems-paper contribution (the ethos of OSDI/SOSP/MLSys: the pieces existed;
making them compose is the hard part). This *complements* the conceptual argument (§3–§5) rather than
competing with it: **the theory says *why* this assembly matters; the systems work shows *that* it can
be built and holds together.**

Why the assembly is non-trivial (say this so "we glued known parts" doesn't read as weak):
- The pieces come from **different research communities** (LLM self-correction, RL verifiers, DB
  provenance, long-context ML, agent frameworks) and were **never designed to compose**.
- The **trust invariants are *emergent* from the assembly, not from any single piece** — `proposer ≠
  gate`, "no implicit accept", answer-key isolation *by construction* at the prep boundary (ADR 0005),
  and proposer-rationale segregation from the verifier exist only because of how the services are wired
  together.
- Running **unattended** takes the plumbing most demos skip: degrade-don't-crash on both boundaries,
  atomic commits, bounded retries, a store-derived run report.
- **Composability is a deliberate design outcome**, not an afterthought: the plugin loader
  (task/verifier = container + entry point, no CP rebuild), the model-agnostic OpenAI-compatible seam
  (4 providers, no code change), and **typed, gated outputs that carry their own provenance + criteria**
  (composability-of-trust, §10).

This framing also keeps the paper honestly **math-free**: an integration/artifact paper needs a
working system, a clear account of the assembly, and honest evaluation — not theorems.

## 2. The problem

A naive loop is trivial to write: *"do X, then improve it; you have Y attempts; no human guidance."*
But that loop makes the agent **both the worker and the judge of its own work**. The two canonical
failure modes of unattended agents follow directly:

- **False success** — the agent declares victory on a metric it controls (goalpost-moving,
  optimistic self-grading, reward-hacking a self-defined criterion).
- **Doom loops / going down a hole** — one bad assumption compounds because nothing external ever
  checks it, and the failed path stays in context and anchors the next attempt.

Underneath both is a deeper issue: in a black-box agent, the judgments that define success —
*have I done enough? what counts as strong evidence? am I done?* — live **implicitly inside the
model**. They are unstated, non-reproducible, and explainable only by the LLM that produced them.
(Deep-research agents, incl. Perplexity-style systems and Claude's own deep-research skill, are the
clearest case: the research is real, but the **stopping rule and the evidence bar are the model's
private opinion** — un-auditable, and at some level unreliable.)

## 3. The contribution

1. **An independent, adversarial verifier the agent cannot fudge.** `proposer ≠ gate` (spec
   §3.3–§3.6): the grader is never the proposer, so the agent cannot grade its own homework.
   "No implicit accept" — a type with no declared gate cannot be committed. The verdict is richer
   than accept/reject (`proposed → tentative → accepted`, plus `rejected`/`superseded`/`revised`),
   and `refine` recovers a mostly-sound artifact with a localized defect by feeding back only a
   **structured gap**.
2. **Structural context hygiene.** State is a typed provenance record, not a transcript; the control
   plane assembles each cycle's context **mechanically from the store** (status/recency — e.g.
   `last_revised_or_accepted`, "build on the best/newest incumbent"). Failed paths are recorded as
   provenance but **never injected into the next working context**. This is a design fix, not a
   prompt-engineering band-aid — and it is the part most agent frameworks cannot easily retrofit.
3. **Inspectable success criteria → bounded *and* explainable outcomes** (see §4).

## 4. Inspectable criteria: why the outcome is bounded and explainable

The verifier's success criteria are a **first-class, inspectable artifact** rather than a hidden
judgment inside the model. This is what makes an autonomous outcome:

- **Bounded** — the agent cannot declare success outside the criteria; it cannot charm the gate.
- **Explainable** — the criteria, their assumptions, and their limitations are written down and
  auditable by someone other than the LLM that produced the work.

Key nuance (state it as a strength, not hide it): independence does not *eliminate* trust — it
**relocates** it from the agent's hidden self-assessment to an **explicit, external, auditable
criterion**. A bad rubric just moves the fudge from the agent to the criteria author — but an
explicit imperfect criterion is inspectable and improvable, whereas an implicit one is neither.

- **Objective criteria** (Kaggle leaderboard): maximally honest — an external grader that can't be
  gamed. The ideal realism capstone.
- **Softer / knowledge tasks**: not every task has an objective metric, and that's fine. The system
  still applies **as long as the verifier definition names the *how*, the *assumptions*, and the
  *limitations*.** That converts the black-box research-agent's private stopping rule and evidence
  bar into an accountable, restate-able contract. The difference between a bounded, accountable
  system and a confident black box is precisely whether those judgments are written down.

## 5. The adversarial-verification spectrum

`proposer ≠ gate` is the invariant that makes any verification *adversarial*. Two points on the
spectrum:

- **Criteria-grading** (the current fe-kaggle gates: runs-clean → proxy-improves → competitive →
  kaggle): grade a proposal against fixed, declared criteria.
- **Refutation** (strong form, future work): a verifier that actively tries to *break* the proposal
  — find the failing test, the counterexample, the input that breaks it.

Naming the spectrum gives a tight contribution today and a clean future-work hook.

## 6. Evaluation plan

### 6.1 Ablation — isolate the one novel piece (the independent grader + its context effects)

Run on a **simple, cheap, auto-graded task** (not full Kaggle) so it can be replicated across many
seeds. Requirements for the task's grader: cheap, deterministic, **local** (no external submission,
no rate limit), short cycles, seedable.

- **T (treatment):** propose → independent verifier grades → CP commits to typed store → next
  context = best incumbent + structured gap; failed paths excluded.
- **C (control):** **no** independent grade — same model, same attempt budget, same compute, one
  accumulating context, agent self-decides when done. The vanilla long-transcript loop.
- **(optional) C2:** grades present but context **not** curated (full transcript retained) — to
  decompose "the grade helps" vs "the context hygiene helps."

Candidate simple tasks: unit-tested coding task (grade = fraction of tests passing — reuses the
existing `code` task type, least new plumbing); small tabular task with a fixed local hold-out
(grade = held-out metric, no leaderboard); rubric / LLM-judge-graded generation task.

**Metrics — report the *mechanism*, not just a final bar:**
- Final quality at equal compute.
- **Context growth over cycles** (T bounded; C balloons) — the direct evidence for "failed paths
  don't pollute context."
- **Derailment / recovery** — after a bad attempt (or a deliberately injected one), does C anchor /
  repeat while T cleanly drops it?
- **Regressions** — does quality go backward? (T resists this *once something is `accepted`*; see the
  ratchet caveat in §8 — in an all-`revised` run the carried-forward incumbent is newest-by-recency and
  *can* drift down. For the 6.2 "after-loop" cell, take the **best-scoring** artifact explicitly, not
  the recency incumbent.)
- Compute / tokens to reach a quality threshold.

### 6.2 "Punch above its weight" — the headline study

Same harness, 2×2:

|                     | single-shot (cycle 1) | after-loop (best incumbent) |
|---------------------|-----------------------|-----------------------------|
| **small local model** | a                   | b                           |
| **large hosted model**| c                   | d                           |

Paper-making result: **b − a ≫ d − c** (the loop lifts the weak model disproportionately), ideally
**b ≈ c** (small-model-in-loop ≈ large-model-single-shot). Cheap to run on the existing
OpenAI-compatible seam (local Qwen + hosted models).

### 6.3 Test-set hygiene

The competitive gate feeds a hold-out-derived gap back each cycle, so the agent is *steered* by
hold-out signal — making the verifier hold-out mildly **optimistic** as a *reported* metric. Report
the headline number on something **never fed back**: the **public leaderboard** (truly held out) or a
third untouched split. Use the verifier hold-out for *gating*, a separate set for the *claim*.
Selling point: the architecture already provides the clean instrument — answer key isolated to the
verifier by construction (ADR 0005), agent never sees labels, gate re-runs the *script* on gold
rather than trusting the agent's CSV.

### 6.4 Kaggle as the realism capstone

External, honest, rate-limited grade at full scale with hosted models. Shows the same mechanism
survives contact with reality, and doubles as the venue for the 6.2 study.

## 7. Evidence already in hand

- **Cross-model / cross-provider runs on an identical harness**: gpt-5.4, sonnet, sonnet+nudge,
  fugu (Sakana, over the OpenAI-compatible seam) under `feature-engineering-test/agent-runs/`.
- **Live instances of the agent fooling itself, caught by the independent gate** — the adversarial
  thesis demonstrated in the wild:
  - fugu run README: the script *self-reported ~0.967* while the held-out gate measured **0.9665**
    ("multipliers tuned on the agent's own split").
  - 2026-06-26 run, cycle 5: the agent's tuning harness optimized against *its own 25% split*; the
    gate's **independent** hold-out confirmed only 0.9663 → 0.9666.
- **Disciplined withholding** — the competitive gate spent **zero** real Kaggle submissions across
  runs until it estimated an attempt was competitive (autonomy *with* discipline).
- **Degrade-don't-crash** on two boundaries (sandbox timeout, external API error) demonstrated live;
  the 3-consecutive-failure safeguard firing cleanly.
- **Narrated qualitative arc** (great for a discussion section): plateau at ~0.966 → agent
  independently diagnoses it and builds its *own* validation/tuning harness → modest **gate-confirmed**
  gain (0.9663 → 0.9666). The independent measurement is what kept the reported number honest.

## 8. Limitations / threats to validity (state plainly)

- **Single domain so far.** "Domain-agnostic kernel" is claimed but evidenced mainly by one real
  domain (FE/Kaggle) + the trivial `code` type. The ablation's simple task + a second domain are
  what back the claim; this is the biggest gap for reviewers.
- **Trust relocated, not removed** (§4) — only as good as the criteria definition.
- **Non-determinism** — hosted models + live leaderboard; mitigate with replicates/seeds on the
  simple task, acknowledge for the Kaggle capstone.
- **N.** Ablations need replicates; one run per arm proves nothing. The simple-task design exists to
  make N feasible.

### Mechanism caveats observed in the 2026-06-26 run (don't overclaim these)

- **The "monotonic ratchet" is a property of the `accepted` lifecycle, not the `revised` one.**
  Incumbent provisioning is `last_revised_or_accepted` — **newest by recency, not highest-scoring**.
  Observed live: cycle 7's **0.9665** became the carried-forward incumbent over cycle 5's higher
  **0.9666**, and `proxy-improves` accepted it because that gate compares against the best *accepted*
  attempt — and in an all-`revised` run nothing is accepted, so it does **not** enforce monotonic
  improvement among revised attempts. Implication: report best-so-far as `max` over scored cycles, not
  "the incumbent"; reserve "never regresses" for runs that actually accept.
- **The soft deadline nudge does not always land.** It fires at 80% of `sandbox_timeout_s` *between
  model steps*; if the agent is deep inside a long training/experiment call when the wall arrives, the
  nudge can't take effect and the cycle is hard-killed. Observed live: even at the raised 80-min wall,
  cycle 6 ran the full 4800 s and **timed out with no proposal** — cleanly handled by
  degrade-don't-crash (recorded `sandbox_failed`, fed back, next cycle launched). A good *paired*
  example for the paper: it demonstrates the failure-handling invariant **and** an honest limitation of
  wall-clock budgeting (more time helps but isn't unbounded headroom for an agent that keeps exploring).

## 9. Related work — positioning

Full verified bibliography (grouped, with citation-hygiene flags) is in
[related-work.md](related-work.md). Compiled 2026-06-26 from three parallel literature scans. Summary
of what it means for us:

**The honest novelty framing (lead with this, or a reviewer will).** Verity is *not* first to gate an
agent loop with an external verifier (FunSearch, AlphaEvolve, SWE-bench, AIDE own that), nor first to
give an agent structured memory (CoALA, the provenance systems own that). The contribution is the
**integration into a domain-agnostic kernel**, plus two things no existing cluster has *together*:
(1) discovery loops have the external evaluator but are **domain-specific** with bespoke graders and
no general typed-provenance state; (2) memory/provenance systems have structured artifact records but
**no verifier-aware commit lifecycle and no notion of keeping failed paths out of context**.

Cluster → our claim → how we differ:

| Our claim | Anchor cites | How we differ |
|---|---|---|
| Proposer can't judge itself | Huang et al. "Cannot Self-Correct Yet" (ICLR'24); Self-Refine; Self-Preference Bias | `proposer ≠ gate` as a structural invariant, not a prompt |
| Independent verifier / gen-verifier split | Cobbe (verifiers); Lightman "Let's Verify Step by Step"; Mind the Gap | Domain-agnostic gate *composition*, not a trained reward model |
| Verifier-gated improvement loop | FunSearch (Nature); AlphaEvolve; AlphaCode | General harness, not one bespoke evaluator |
| Kaggle/ML-agent eval regime | **MLE-bench (OpenAI'24)**; AIDE; DS-Agent | MLE-bench *is* our benchmark frame; AIDE is closest sibling (held-out-gated tree search) |
| Structural context hygiene | Lost in the Middle (TACL'24); Shi et al. "Distracted by Irrelevant Context" (ICML'23) | We *exclude* failed paths by construction vs. hoping the model ignores them |
| Typed-provenance state | CoALA; PROV-DM; Ground; MLflow/ModelDB | We add the commit lifecycle + "no implicit accept" a log lacks |
| Why now: long-horizon reliability | METR "Long Tasks" (2025); GAIA/WebArena/AgentBench | Reliability-over-length is the binding constraint we target |

Every "how we differ" is a prose sentence — no math/theory section needed. Citation hygiene: cite DVC
and the Chroma "Context Rot" report by repo/docs (not as papers); 30-second-verify the 2025 arXiv-only
items (AlphaEvolve, AIDE, Weaver, Mind-the-Gap, and the two flagged reliability papers) before a
references list. Lean on Liu (TACL) + RULER (COLM) for the load-bearing context-degradation cites.

## 10. System properties / artifact value proposition

The four things that make this a credible *practical* paper (they map to artifact-track criteria:
available / functional / reusable). State each with its honest caveat:

- **Generalizable — architecturally yes, empirically one domain (so far).** The kernel is
  content-neutral (read→propose→gate→commit); the plugin loader is the evidence (new task *or* verifier
  type = a container + an installed entry point, no CP rebuild). Frame as *generalizable by
  construction, demonstrated on one domain* — and back it by making the **ablation's simple task a
  genuinely different domain** (e.g. unit-tested code), which doubles as the second-domain evidence
  that closes the biggest reviewer gap.
- **No vendor lock — proven live at the model layer.** Anthropic (sonnet), OpenAI (gpt-5.4), Sakana
  (fugu), and local Ollama/Qwen all drive the same harness over one OpenAI-compatible seam **with no
  code change** — a recorded result across four providers, not a promise. Standard parts elsewhere
  (Docker, SQLite, HTTP/UDS, Python entry-point plugins). *Honest caveat:* "no vendor lock" ≠ "no
  dependencies" — it assumes a container runtime and the sandbox leans on a specific agent framework
  (Deep Agents/LangChain), swappable behind the `BackendSandboxDriver` seam but one implementation in
  practice. Frame the driver seam as the mitigation.
- **It runs.** End-to-end, on an external benchmark, reproducibly, for hours, across providers — a bar
  many submissions fail. The recorded `agent-runs/` *are* the artifact-evaluation story; don't
  undersell it.
- **Composable — two layers.** *Mechanically:* already composable at the service boundary (long-lived
  daemon + CLI + HTTP API + byte data plane + async create→run→poll→results→export), so a Verity task
  is an addressable, scriptable unit today; making it a node in another workflow engine is a thin
  adapter (typed I/O contract + role-keyed routing already exist) — hence "made composable quickly" is
  fair. *Conceptually (the part worth putting in the paper):* **a Verity task's output is itself a
  typed, gated artifact with provenance and the criteria it met** — it arrives *pre-verified with an
  attached, inspectable contract*, where most agent output is untyped text you must re-verify. That's
  **composability-of-trust**, not just of data, and it extends the inspectable-criteria thesis:
  verified provenance is what makes an autonomous agent's output safe to compose into a larger system.
  *Cheap way to show (not just assert) it:* one small example — a Verity task's gated output consumed
  by a downstream step, or two task types chained — would be a convincing figure.

## 11. Venue framing

Practical / artifact / experience paper. Targets: MLSys (artifact track), agents / systems-for-ML
workshops at NeurIPS / ICML, or arXiv preprint + workshop. These venues reward a working,
reproducible artifact with honest evaluation — Verity's strength. **Do not** frame as "we win
Kaggle"; frame as "an autonomous harness whose outcomes are bounded, explainable, and
independently verified."

## 12. Spine (abstract-shaped summary)

Autonomous improvement requires an **adversarial, unfudgeable success signal** and **structural
context hygiene**; Verity provides both via typed provenance behind a verifier-aware commit
lifecycle, with **inspectable success criteria** that make outcomes bounded and explainable even
when the criteria are subjective. We show via ablation that each piece matters, that the loop lifts a
small model disproportionately, and — on an external Kaggle leaderboard — that it survives contact
with reality.
