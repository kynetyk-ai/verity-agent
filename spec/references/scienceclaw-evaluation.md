<!--
Copyright (c) 2026 Kynetyk Holdings LLC. All rights reserved.

No part of this file may be reproduced, distributed, or transmitted in any 
form or by any means without the prior written permission of the owner.
-->

# Evaluation: ScienceClaw as the Reference Implementation

**Purpose.** This is a grounding analysis for the [Self-Revising Discovery Harness spec](../self-revising-discovery-harness.md). It evaluates the implementation the source paper is written about — **ScienceClaw / CategoryScienceClaw** (`github.com/lamm-mit/scienceclaw`) — against both the paper's own categorical framework and against our spec, and extracts what the spec should learn from it.

**Sources read.** Wang & Buehler, *Self-Revising Discovery Systems for Science* (arXiv:2606.01444), pp. 1–30 (full body + Methods/Definitions; pp. 28–36 are references). ScienceClaw repo at a local checkout: `ARCHITECTURE.md`, `artifacts/artifact.py`, `artifacts/reactor.py`, `artifacts/mutator.py`, `artifacts/discovery_rubric.py`. File/line citations below refer to that checkout (commit `f4a6286`, "add paperclip skill").

---

## 1. ScienceClaw is the implementation, not an example of it

The paper's footnotes and §2.6–2.7 make ScienceClaw the named subject: `github.com/lamm-mit/scienceclaw`, the `categoryscienceclaw-mechanics` branch, and the `infinite` discourse platform. The system on disk is the three-layer architecture the paper describes:

- **ScienceClaw** — the *execution substrate*: a typed skill registry (300+ skills), immutable content-hashed artifacts with parent lineage, shared "open needs," plannerless pressure-based coordination, and workflow mutation of the active artifact graph.
- **Infinite** — the *discourse substrate*: posts, comments, votes, reputation, read by the paper as verifier signals.
- **CategoryScienceClaw** — the claimed *categorical / proof-carrying layer* over those two.

The Builder/Breaker protein-mechanics case with the MDL gate — the paper's quantitative result — lives in a **separate** repo (`github.com/lamm-mit/BreakingTheWorld`) and the mechanics branch, *not* in the mainline ScienceClaw substrate. This distinction matters for everything below.

## 2. The paper says exactly where the implementation stops short of the theory

The authors are candid, and the code confirms each admission verbatim:

- "The unary shadow of this hypergraph generates a free provenance category … the full multi-parent synthesis structure is more accurately read as a typed multicategory … **This is weaker than a software-enforced schema category, and should not be overclaimed.**" (§2.7)
- "The ArtifactReactor **does not yet solve a formal Kan-extension or lifting problem**; it performs the implemented engineering analogue, using pressure scores and schema overlap." (§2.7)
- The publication map "is **not yet a certified functor in software**, but it has the functorial shape required for one." (§2.7)
- §3 (Conclusions): the next step "is therefore **not to replace ScienceClaw, but to lift structures already present** … Making this explicit would change the operational status of the platform. The system would no longer merely *display* provenance; it could **verify that provenance diagrams commute, that every public claim has an admissible artifact path, that retractions and supersessions preserve old evidence, and that new artifact types are introduced through recorded schema transitions.**"

That final paragraph is, almost verbatim, the thesis of our spec. **The harness spec is the "next step" the paper names** — converting displayed discipline into enforced discipline. The paper also gives us the audit contract directly: §2.3 lists "stable artifact identifiers, typed tool or skill signatures, explicit parent lineage, append-only or explicit supersession semantics, status records for failed or retried calls, and no silent merge or deletion of accepted artifacts" as the conditions under which the endofunctor model applies — i.e., our §5 invariants are the paper's own audit contract.

## 3. What the substrate has, and what is *enforced* vs. *by convention*

| Spec component (harness §) | In ScienceClaw? | Enforced? | Evidence |
|---|---|---|---|
| Typed provenance store (§4) | **Yes, strong** | Append-only JSONL + sha256 `content_hash`; per-agent `store.jsonl` + shared `global_index.jsonl` | `artifact.py:582–626` |
| Stable IDs (§5.1) | Yes | uuid4 | `artifact.py:550` |
| Operation edges / DAG (§4.1) | Yes (`parent_artifact_ids`) | By convention | `artifact.py:525`; depth walk `:918` |
| Multi-parent synthesis (§8.2) | Yes | Naive `payload.update` merge, newest key wins | `reactor.py:_react_multi:788` |
| Schema registry / meta-schema (§8.1) | **Partial** — `SKILL_DOMAIN_MAP` declares skill→*output* types; **inputs are not typed**, scraped from `--help` regex | Not typed | `artifact.py:36`; `reactor.py:_skill_input_params:305` |
| Open needs = typed holes (§8.4) | Yes (`needs`, pressure-ranked) | Heuristic match, not a lifting problem | `reactor.py:react_to_needs:1178` |
| **Status lifecycle (§6)** | **Absent** — no `proposed/tentative/accepted/rejected/superseded`; only a `result_quality` tag set at creation | — | `artifact.py:526` |
| **Gate / commit path (§7)** | **Absent in the substrate** — `create_and_save` writes unconditionally on skill success ("success" = process returned a dict) | — | `reactor.py:1021–1036`; `_transform`, `react_to_needs` |
| Supersession (§5.2, §7) | **Absent** — artifacts only accrete; nothing is ever retired | — | no `superseded` field anywhere |
| Schema migration (§15) | **Absent** — `schema_version` hardcoded `"1.0"`; only `from_dict` setdefaults | — | `artifact.py:554, 572` |
| Immutability (§5.2) | **By convention only** — `Artifact` is a plain (non-frozen) dataclass; docstring *claims* immutable | Not enforced | `artifact.py:509–514` |
| Capability gate | Yes — an agent may only claim types within its `preferred_tools` | Enforced (`ArtifactDomainError`) — but this is *authorization*, not *quality* | `artifact.py:989` |
| Quality rubric (§11) | Yes, **but explicitly non-gating** — `DiscoveryEvaluation` is a "Soft rubric (never gates posting)"; heuristic key-name scan → 0–100 score + tier | Not binding | `discovery_rubric.py:104–105` |
| Mutation / cleanup | Yes (`fork/prune/graft/merge_conflict`) | Heuristic, post-hoc; `merge_conflict` resolves by **"newer timestamp wins"** | `mutator.py:391–445` |

## 4. The central finding

**ScienceClaw implements the typed-provenance half of the vision strongly and the verifier-aware-lifecycle half barely at all.** Every successful skill invocation becomes a durable, content-hashed, lineage-bearing artifact, with **no commit gate deciding whether it was worth keeping**. The one component resembling a quality judgment, `discovery_rubric.py`, is documented to *never gate*. The MDL/AIC gates from the paper's results sit in a different repo.

This makes ScienceClaw an almost-exact instance of the system our spec's §2 (Principle 4) warns about:

> "A generic harness with stub gates is a beautifully-audited way to accumulate garbage."

It is the strongest possible external validation of the spec's central bet: the hard, non-negotiable, build-first part is exactly the part that a sophisticated, 300-skill, multi-agent production system left as *displayed, not enforced*.

## 5. Lessons for the spec

### Confirmations — keep / borrow
- **The two-tier store is real and works.** Per-producer `store.jsonl` (full payloads) + a payload-free `global_index.jsonl` (fast cross-agent scan) is a concrete, proven realization of the manifest / bounded-context split (§9–§10). Cite it as the reference pattern.
- **content-hash + append-only JSONL** is a legitimate zero-infra backend, validating the "git-on-disk JSON is a valid v0" stance (§4).
- **Open needs + pressure scoring** are working instantiations of typed holes and a plannerless retrieval/planner policy (§8.4). Good prior art to name.

### Sharpenings — where the gaps prove our invariants must be airtight
1. **"No implicit accept" is load-bearing (§2 Principle 4, §5.7, §7).** ScienceClaw is the live proof of what happens without it. The failure is not theoretical.
2. **Immutability must be enforced, not documented (§5.2).** A mutable dataclass labeled "immutable" silently erodes the guarantee. The spec should require enforcement (frozen types / writes only through the commit path).
3. **A non-binding rubric is decoration (§11).** `discovery_rubric.py` "never gates." A gate's verdict must *bind* the lifecycle; heuristic key-name scoring sits low on the reliability ladder and must be labeled weak.
4. **"Newer wins" is a silent-merge anti-pattern (§5.6).** `mutator._merge_conflict` resolves contradictions by timestamp — the exact hazard the invariant forbids.
5. **Operation signatures need both sides (§8.1–§8.2).** Declaring output types but inferring inputs from `--help` regex degenerates compatibility into string key-overlap. Signatures must be typed on input *and* output.
6. **Migration is the hardest, last piece — unbuilt even here (§15).** `schema_version` is frozen at "1.0". Confirms deferring-but-seaming migration is realistic, and that it is genuinely the distinguishing "self-revising" capability.

## 6. One-line takeaway

The harness spec should position itself, explicitly, as the enforced successor the paper's own §3 calls for: take ScienceClaw's typed-provenance substrate as validated prior art, and supply the one thing it leaves to convention — a kernel-owned commit path with a mandatory gate and an enforced lifecycle.
