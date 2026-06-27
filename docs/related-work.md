# Verity — related work & positioning

> Working bibliography for the practical paper (companion to [paper-outline.md](paper-outline.md)).
> Compiled 2026-06-26 from three parallel literature scans; every entry was confirmed to exist via
> web search with a real title/authors/venue/URL **except** the few explicitly flagged. This is a
> *positioning* document — a practical paper's related-work section is prose ("we're like X, differ
> in Y"), not theory. **No math required.**

## The honest framing (read this first)

Verity's novelty is **not** "first to gate an agent loop with an external verifier" — that lineage is
well established (FunSearch, AlphaEvolve, AlphaCode, SWE-bench, AIDE, MLE-bench). Nor is it "first to
give an agent structured memory" (CoALA, MemGPT, the provenance systems). The contribution is the
**integration** into a *domain-agnostic* harness, plus two things the existing clusters individually
lack:

1. The discovery-loop papers (FunSearch et al.) have an external evaluator but are **domain-specific**,
   with bespoke evaluators and no general typed-provenance state or commit lifecycle.
2. The memory/provenance systems (CoALA, MLflow, Ground, PROV-DM) have structured artifact records but
   **no verifier-aware commit lifecycle** ("no implicit accept", `proposer ≠ gate`, rich status) and
   **no notion of gating failed paths out of the agent's working context**.

Verity = **(independent adversarial verifier + commit lifecycle) × (typed-provenance state) ×
(structural context hygiene) × (inspectable criteria) as one reusable, domain-agnostic kernel.** Say
that plainly; don't claim a primitive someone else owns.

---

## Cluster A — "The proposer can't be its own judge" (motivation)

The empirical case for `proposer ≠ gate` and an external grader.

- **Large Language Models Cannot Self-Correct Reasoning Yet** — Huang et al., 2023 (ICLR 2024).
  arXiv:2310.01798. *Strongest single citation:* without external feedback, intrinsic self-correction
  fails to help and often **degrades** accuracy. Direct motivation for an independent verifier.
- **Self-Refine: Iterative Refinement with Self-Feedback** — Madaan et al., NeurIPS 2023.
  arXiv:2303.17651. The canonical "agent grades itself" loop Verity reacts against.
- **Reflexion: Verbal Reinforcement Learning** — Shinn et al., NeurIPS 2023. arXiv:2303.11366.
  Self-reflection works **only when grounded in an external feedback signal** — and its
  reflect-and-feed-forward mirrors Verity's `refine` verdict (feed the localized gap back, don't
  discard). Cross-cited under memory/autonomy.
- **Judging LLM-as-a-Judge (MT-Bench / Chatbot Arena)** — Zheng et al., NeurIPS 2023 D&B.
  arXiv:2306.05685. Names **self-enhancement bias** in LLM judges.
- **Self-Preference Bias in LLM-as-a-Judge** — Wu et al., 2024. arXiv:2410.21819. Quantifies that a
  model systematically favors its **own** outputs — the empirical reason to separate proposer & grader.

## Cluster B — Verifiers / generator–verifier split (the mechanism)

- **Training Verifiers to Solve Math Word Problems** — Cobbe et al., 2021. arXiv:2110.14168. Seminal
  generator-verifier split (GSM8K); a distinct verifier ranking outputs.
- **Let's Verify Step by Step** — Lightman et al., 2023. arXiv:2305.20050. Process reward models —
  argues for **explicit, inspectable per-step criteria** graded by a dedicated verifier over holistic
  self-scores. Good anchor for "inspectable criteria."
- **Mind the Gap: Self-Improvement Capabilities of LLMs** — Song et al., 2024 (ICLR 2025).
  arXiv:2412.02674. Formalizes the **generation–verification gap** — *when* an external/stronger
  verifier is necessary for an unattended loop. *(2024 arXiv; widely cited — verify venue line.)*
- **Shrinking the Generation-Verification Gap with Weak Verifiers (Weaver)** — Saad-Falcon et al.,
  2025. arXiv:2506.18203. Composing many weak verifiers into a strong grade — analogous to Verity's
  **gate composition** (runs-clean → proxy → competitive → kaggle). *(2025 arXiv — double-check.)*

## Cluster C — External-verifier-gated discovery / improvement loops (closest siblings)

Verity's lineage; differ by being **domain-specific** with bespoke evaluators and no general
typed-provenance / context-hygiene story.

- **FunSearch** — Romera-Paredes et al., *Nature* 625, 2023/24. LLM proposes program mutations; an
  evolutionary loop keeps only candidates an **automated scorer** rates higher. The canonical "LLM
  proposes, external evaluator disposes."
- **AlphaEvolve** — Novikov et al. (DeepMind), 2025. arXiv:2506.13131. Generalizes FunSearch to
  whole-file code under automatic evaluators. *(2025 tech report.)*
- **AlphaCode** — Li et al., *Science* 378, 2022. Execution-on-tests filters millions of candidates to
  ~10 submissions — verifier-as-selector.

## Cluster D — ML-engineering / data-science agents (your task family)

- **MLE-bench** — Chan et al. (OpenAI), 2024. arXiv:2410.07095. **75 Kaggle competitions graded
  against real leaderboards/medals** — *defines exactly Verity's evaluation regime.* Cite as the
  benchmark frame for the fe-kaggle capstone.
- **AIDE: AI-Driven Exploration in the Space of Code** — Jiang, Schmidt et al. (Weco), 2025.
  arXiv:2502.13138. ML engineering as **tree search gated by held-out validation** — the closest
  empirical analog to Verity's held-out-score ladder. *(2025 arXiv.)*
- **DS-Agent** — Guo et al., ICML 2024. arXiv:2402.17453. Case-based reasoning + performance-feedback
  loop for ML pipelines.

## Cluster E — Test/execution-feedback code agents (proposer≠gate in practice)

- **SWE-bench** — Jimenez et al., ICLR 2024. arXiv:2310.06770. Patches judged by the repo's **own test
  suite** — the defining external-verifier benchmark for code.
- **SWE-agent** — Yang et al., NeurIPS 2024. arXiv:2405.15793. Agent runs tests as its feedback signal.
- **Teaching LLMs to Self-Debug** — Chen et al., ICLR 2024. arXiv:2304.05128. Execution results are the
  objective check that triggers revision.
- **AgentCoder** — Huang et al., 2023/24. arXiv:2312.13010. Separate programmer / test-designer /
  **test-executor** agents — structurally mirrors `proposer ≠ gate`.

## Cluster F — Agent memory architectures (typed-provenance state vs transcript/vector)

- **CoALA: Cognitive Architectures for Language Agents** — Sumers et al., 2023. arXiv:2309.02427. The
  **strongest conceptual antecedent** — argues memory should be *typed and modular*, not an
  undifferentiated transcript or vector blob.
- **MemGPT / Letta** — Packer et al., 2023. arXiv:2310.08560. OS-style tiered memory the agent
  self-edits — structured but **agent-mutated**, not verifier-gated.
- **Voyager** — Wang et al., 2023. arXiv:2305.16291. A skill library of **validated** executable
  artifacts — adjacent to a verifier-aware store.
- **Generative Agents** — Park et al., UIST 2023. arXiv:2304.03442. Append-only NL "memory stream" with
  recency/importance retrieval — between transcript and typed store; no commit lifecycle.
- **A Survey on the Memory Mechanism of LLM-based Agents** — Zhang et al., 2024. arXiv:2404.13501. The
  design-space map to situate typed-provenance against transcript-buffer and vector-store paradigms.

## Cluster G — Context degradation (the evidence for structural context hygiene)

Why *excluding* failed paths beats trusting the model to ignore them.

- **Lost in the Middle** — Liu et al., TACL 2024. arXiv:2307.03172. U-shaped accuracy vs. position —
  padding context with dead-ends buries what matters. Load-bearing citation.
- **LLMs Can Be Easily Distracted by Irrelevant Context** — Shi et al., ICML 2023. arXiv:2302.00093. A
  **single** irrelevant clause sharply drops reasoning. The cleanest "exclude dead-ends" evidence.
- **The Power of Noise (RAG)** — Cuconasu et al., SIGIR 2024. arXiv:2401.14887. The most harmful content
  is the **plausible near-miss** — exactly what a discovery agent generates and fails.
- **RULER** — Hsieh et al., COLM 2024. arXiv:2404.06654. Big context windows ≠ robustness to
  distractors. ("Context Rot", Chroma 2025 — *vendor report, not peer-reviewed; framing only.*)

## Cluster H — Provenance / "git for artifacts" (the state model)

- **PROV-DM (W3C)** — Moreau, Missier et al., 2013. The reference vocabulary for **entities,
  activities, agents** — Verity's typed record is a commit-gated specialization of this triad.
- **Ground: A Data Context Service** — Hellerstein et al., CIDR 2017. Closest sibling: a versioned
  graph of artifacts + activities; Verity narrows it to an agent's working set behind a commit gate.
- **MLflow** — Zaharia et al., 2018; **ModelDB** — Vartak et al., HILDA 2016. Passive logging stores;
  Verity adds the **verifier-aware commit lifecycle + "no implicit accept"**, turning a log into an
  auditable record *with decisions*.
- **ProvDB** — Miao et al., HILDA 2017. arXiv:1610.04963; **DataHub** — Bhardwaj et al., CIDR 2015
  ("git for data"). (DVC — tool/repo, cite by docs, not as a paper.)

## Cluster I — Long-horizon autonomy / reliability (the "why this matters now")

- **Measuring AI Ability to Complete Long Tasks** — Kwa et al. (METR), 2025. arXiv:2503.14499.
  **Reliability over task length**, not raw capability, is the binding constraint — exactly what a
  gated, provenance-tracked loop extends.
- **ReAct** — Yao et al., ICLR 2023. arXiv:2210.03629. The foundational read→act loop; error
  propagation motivates wrapping each step in a gate.
- **AgentBench** / **WebArena** / **GAIA** — Liu et al. 2308.03688 / Zhou et al. 2307.13854 / Mialon et
  al. 2311.12983 (all ICLR 2024). Stark long-horizon human-vs-agent gaps on **verifiable** tasks — the
  regime where a gated commit lifecycle pays off.
- *(Flagged, verify before citing:)* "Where LLM Agents Fail and How They Can Learn From Failures"
  (arXiv:2509.25370) — error cascades; "CaRT: Teaching Agents to Know When They Know Enough"
  (arXiv:2510.08517) — the stop/halt problem (relates to `--stop-on-accept`). Both 2025 arXiv-only,
  author rosters not individually re-verified.

---

## Citation hygiene
- **Non-papers** (cite by repo/docs, not as peer-reviewed): DVC; Chroma "Context Rot" report.
- **Double-check before a references list**: the 2025 arXiv-only items (AlphaEvolve 2506.13131, AIDE
  2502.13138, Weaver 2506.18203, Mind-the-Gap 2412.02674, and the two flagged reliability papers).
- For context-degradation claims, lean on **Liu (TACL)** and **RULER (COLM)** as the load-bearing
  peer-reviewed cites; use vendor/term sources for framing only.
