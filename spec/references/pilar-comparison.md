<!--
Copyright (c) 2026 Kynetyk Holdings LLC. All rights reserved.

No part of this file may be reproduced, distributed, or transmitted in any 
form or by any means without the prior written permission of the owner.
-->

# Comparison: pilar as a Proven Soft-Gate Instance

**Purpose.** A grounding analysis for the [Self-Revising Discovery Harness spec](../self-revising-discovery-harness.md). pilar is *not* an implementation of this spec — it is a Claude Code plugin for medical writers building Scientific Communication Platforms (the implementation of the [`scp-plugin`](../../../functional_tools/marketing_comms/scp-plugin/) spec; see also its generalization, [Clara](../../../functional_tools/strategy_analysis/strategic-analyst-plugin/)). But it is the closest *working, in-use* relative of the harness on the soft-gate side, and it supplies one principle the spec was missing outright. This document records the parallels, the lessons, and the honest limits.

**Source.** Plugin at `/Users/joshuaziel/Documents/coding/claude-plugins/plugins/pilar` (commands/, agents/, schemas/, scripts/, docs/CONVENTIONS.md, IMPLEMENTATION_ROADMAP.md). The user reports it works well in practice.

---

## 1. Where pilar sits

scienceclaw (see [`scienceclaw-evaluation.md`](scienceclaw-evaluation.md)) and pilar are the two poles of this spec:

- **scienceclaw** — strong typed provenance, **no gate**: every successful tool call becomes durable state. Validates the *store* half; cautionary tale for the *gate* half.
- **pilar** — a strong, **enforced, production gate** with an independence contract and append-only audit discipline, over a *fixed* deliverable schema. Validates the *gate / lifecycle* half; proof that the discipline is buildable and usable.

Between them they vindicate both halves of the spec from opposite directions. pilar specifically is a hand-built, domain-specialized instance of **§14 (the soft-gate consulting domain)**: `Source → Clue → Insight → Recommendation` appears as `manifest-source → reference-statement → scientific-statement → pillar`; the cheap grounding/relevance gates are the Fact-Checker; the human-in-the-loop worth-keeping gate is the Strategic Reviewer.

## 2. Architectural mapping (instructive, not exact)

| Spec element (harness §) | pilar mechanism | Evidence |
|---|---|---|
| Typed provenance, not transcript (§2.1) | Git markdown artifacts with YAML frontmatter; engagement repo *is* the state | `schemas/`, `docs/CONVENTIONS.md` |
| Stable IDs, append-only (§5.1–5.2) | `P-NN`, `SS-NN`, `RS-NN`, `GAP-NNN`, `ASP-NNN`, `cd-NNN`; preserved (never reused) even on rewind | `docs/CONVENTIONS.md` § "Append-only renumbering" |
| Operation edges / provenance (§4) | `sources: [ref-id]` on reference statements; composite refs `P-04.SS-01.RS-02`; `linked_to` / `linked_statement` | `schemas/pillar.md`, registers |
| Commit path (§7) | Every command *proposes* a commit; user approves; no auto-commits. Enforced by command logic + CI scripts, not a runtime kernel | `commands/*.md`, `IMPLEMENTATION_ROADMAP.md` |
| Mandatory gate (§5.7, §7) | QC sequence Editor → Fact-Checker → (consolidated) Strategic Reviewer; high-severity findings block handoff | `commands/run-qc.md`, `commands/handoff.md` |
| Proposer ≠ gate (Principle 6) | Three **independent reviewer subagents** with a statically-enforced **independence contract** | `agents/`, `scripts/context-audit.py` |
| Status lifecycle (§6) | Pillar `draft→narrative-approved→statements-approved→complete`; RS `draft/approved/gap/aspirational`; sprint `pending→confirmed/revisions-requested/deferred/rewind` | `schemas/*.md` |
| Regenerate, never append (§9–§10) | Consolidated draft is *deterministically regenerated* from canonical pillars; never edited in place | `scripts/consolidate.py`, Phase-8 decision |
| Bounded context from disk each session (§9–§10) | `CLAUDE.md` auto-load reads roadmap + latest sprint summary + active plan; commands re-read from disk, not transcript | `commands/init.md` |
| Schema migration (§15) | `REF-NNN` → property-based ref-id migration, recorded with plan + breaking version bump | Decisions Log 2026-05-05 |

## 3. Lessons for the spec

Ranked by novelty to the spec.

### 3.1 The independence contract — the gate must be *denied context*, not merely a different agent
The standout. Each pilar reviewer receives *only* the inputs it needs: the Fact-Checker sees the artifact and its cited sources, never the briefing, drafting rationale, or other pillars; the Strategic Reviewer sees the briefing, roadmap, and consolidated draft, never the rationale or source files. This is **statically enforced** by `scripts/context-audit.py` (a CI gate that asserts each subagent prompt contains only allowlisted variables, no forbidden tokens, and only permitted tools). Our Principle 6 says "proposer ≠ gate" and stops. pilar shows the stronger, correct rule: *a gate that can see the drafting rationale will rationalize the draft*. Starving the verifier of context is what makes a Breaker adversarial rather than confirmatory. → **Add to §10 and §8.3.**

### 3.2 Deterministic scripts are how invariants get enforced when the store is files
Our §5 says invariants "become property tests." pilar shows it concretely: `validate-schemas.py` (IDs append-only, unique, well-formed — CI gate), `detect-gaps.py` (orphan reference statements = a structural "no implicit accept" / grounding check), `context-audit.py` (the independence contract). This is a pragmatic middle path between scienceclaw's convention-only discipline and a true runtime kernel: the agent *can* technically write a file, but a deterministic gate refuses to let an invariant-violating state be committed. → **Cite at §5 and §7 as `[PILAR]`.**

### 3.3 Regenerate the derived view; never edit it in place
pilar's consolidated draft is deterministically regenerated from the canonical pillars; findings are fixed at the *source* pillars and re-consolidated, never edited in the draft. That is exactly §10 "regenerate, never append; the store is canonical, the projection is disposable" — applied to the *deliverable*, not just the prompt context. Independent confirmation the principle generalizes. → **Cite at §9–§10.**

### 3.4 Richer "held" states for a soft domain, with dedicated registers
`gap` and `aspirational` statuses, plus the evidence-gaps and aspirational-statements registers, are a working realization of "status is richer than accept/reject" and of *logging what is not yet supported* — our `tentative` and rejected-log in the soft-gate idiom. → **Informs §6 and §14.**

### 3.5 Structural gate vs. semantic gate, split cleanly
pilar separates the deterministic structural check (`detect-gaps.py`: does every claim resolve to a real source?) from the semantic judgment (Fact-Checker: overstatement, misattribution, source-strength mismatch). This is our cheap-gates-then-hard-gates lifecycle (§6) and reliability ladder (§11) — deterministic rung earns the structural pass; the LLM/human rung makes the semantic call. → **Cite at §11.**

### 3.6 Migration done for real
pilar performed a schema migration (`REF-NNN` → property-based ref-ids) with a recorded migration plan and a breaking-version bump, and even recorded an implementation-driven spec amendment. A working instance of §15's "recorded schema transition." → **Cite at §15.**

## 4. Honest limits — where pilar is *not* this system

- **A fixed-schema instantiation, not a domain-agnostic kernel.** pilar hardcodes one deliverable shape; it has no schema/tool/gate *registries*. (Clara is pilar's generalization and is structurally closer to "kernel + extension points.") pilar validates a *domain*, not the kernel.
- **Enforcement is command-convention + CI, not a single kernel-owned write surface.** Same structural caveat as scienceclaw — the model can write a file — but materially mitigated by the deterministic validators.
- **The human is the loop.** pilar does not autonomously run propose→gate→commit; the writer drives and approves at every checkpoint. It validates the *discipline*, not the *autonomy* the agent loop (§10) assumes.
- **The gate is soft and advisory-plus-blocking, not a numeric functional.** Appropriate to its domain (§14); not the objective-gate case (§12).

## 5. One-line takeaway

pilar is the strongest evidence that the gate-and-audit half of the spec is sound, buildable, and pleasant in use — and it contributes one principle the spec was missing: the **context-starved, statically-enforced independent verifier**.
