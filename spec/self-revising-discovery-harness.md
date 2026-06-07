<!--
Copyright (c) 2026 Kynetyk Holdings LLC. All rights reserved.

No part of this file may be reproduced, distributed, or transmitted in any 
form or by any means without the prior written permission of the owner.
-->

# Self-Revising Discovery Harness

## Natural Language Specification

**Status:** Draft v0.2 — adds the physical/service architecture (a deployable service decomposition, an advisory verifier, and the `refine` verdict) over the v0.1 logical kernel; pending review.
**Audience:** Coding agent (primary), human reviewers (secondary).
**Scope:** Architecture-level, in **two views**. The *logical view* defines the components, contracts, invariants, and the rationale behind them for a domain-agnostic kernel and its extension points. The *physical view* (§3.3–§3.6) decomposes that kernel into deployable, containerized services — a control plane, a sandbox, and a verifier — with the API contracts between them. It commits to the decisions that shape an implementation and leaves choices that do not materially affect the architecture to the implementer.

**Conceptual origin:** Wang & Buehler, *Self-Revising Discovery Systems for Science: A Categorical Framework for Agentic Artificial Intelligence* (arXiv:2606.01444, 2026), saved at `references/2606.01444v1.pdf`. The categorical framework is the *specification lens*, not a runtime object: it tells us what must be true of the system's state and operations. We do not implement category theory. We implement disciplined record-keeping plus a verifier-aware lifecycle, and use the framework only to check that the disciplines are complete and consistent. Where this spec invokes a categorical idea (a regime transition, an empty-but-valid type, the obstruction to migrating isolated evidence), it does so to name a requirement, never to mandate a representation.

---

## 1. Vision

The **Self-Revising Discovery Harness** is a domain-agnostic **kernel** that maintains an agent's persistent state as a *typed provenance record* — artifacts, the operations that produced them, and the gate decisions made about them — rather than as a chat transcript or an embedding store. Around that kernel sit a small number of **typed extension points** (schema, tools, gates, retrieval) that configure it for a specific domain. The kernel enforces an audit contract and a commit lifecycle. The domain supplies the artifact schema, the tools, and — the irreducible per-domain work — the **gates** that decide what is worth keeping.

The mental shorthand is **"git + a type system + a verifier-aware lifecycle for agent artifacts."** Git, because state is an append-only, attributable, never-silently-rewritten history. A type system, because artifacts and the operations between them have declared types and the kernel reasons over their signatures. A verifier-aware lifecycle, because nothing enters durable state without passing a declared gate, and the *basis* on which it was kept is recorded alongside it.

The harness exists because the two common ways of giving an agent memory both fail the same test — *can you trust, and audit, what the system believes?* A transcript loses everything past the context window and cannot adjudicate a contradiction between two things it once said. An embedding store conflates "relevant" with "true" or "accepted," so a confidently-wrong passage retrieves as readily as a verified one. A typed provenance record fixes both: it persists independently of context size, it records why each artifact is believed, and it is auditable after the fact.

This is an **integration**, not a research project. Workflow orchestrators, experiment trackers, and agent loops all exist. The differentiated combination is three things held together: **typed provenance as the primary state**, **the gate as a first-class lifecycle stage** rather than a side-effect of prompting, and **schema migration as a supported operation** rather than a rewrite. The reusable asset is the kernel; the per-domain moat is the gates.

## 2. Guiding Principles

These principles are the load-bearing design conclusions. Each is a deliberate choice with a cost, not a default, and the rest of the spec is their elaboration. An implementation that violates one of these has not implemented this spec.

1. **State is typed provenance, not transcript or vector.** The durable record is artifacts, the operations that produced them, and the gate decisions about them. This is the source of truth. Everything an implementer might reach for instead — the conversation, a vector index — is at most a derived, disposable projection of this record.

2. **Two tiers: a durable store and a volatile context.** The store persists and is authoritative. The context window is a *bounded, regenerated projection* of the store — rebuilt each turn from scratch, never appended to across turns. Committing an artifact is the act that flushes it from volatile scratch into durable state. A consequence, made testable later, is that context size must not grow with store size.

3. **The harness enforces the discipline; the model only proposes.** The agent never writes to the store directly. It emits *proposals*. A kernel-owned commit path runs the gate, records the decision, and sets status. A hallucinated proposal therefore becomes a `rejected` row with a rationale — a recorded non-event — never corrupted state.

4. **The gate is the moat.** Everything else in this system generalizes across domains; the gate does not. A gate's trustworthiness lives in the evaluation harness behind it, and the quality of the whole system is capped by the quality of its gates. It follows that there is **no implicit accept**: the kernel must refuse to commit an artifact of a type that has not declared an explicit gate. A generic harness with stub gates is a beautifully-audited way to accumulate garbage, and the spec is designed to make that failure impossible rather than merely discouraged.

5. **"Worth keeping" is two questions, not one.** *Validity* — is the artifact sound? — and *selection* — does it earn its place over the incumbent, net of its cost? Where a numeric scoring functional exists (the paper's MDL/AIC gates are the model), commit to it up front and the selection decision becomes mechanical. Where one does not, descend the reliability ladder (§11) deliberately and record the reliability cost; do not pretend a judgment call is a measurement.

6. **The proposer is never its own gate.** The model that produced an artifact must not be the authority that judges it. Separation is achieved with a deterministic checker, a *different* model, or an adversarial "Breaker" brief whose job is to refute. This Builder/Breaker structure is a kernel-enforced property of the commit path, not a convention the domain is trusted to follow. In the physical architecture (§3) this separation hardens into a **network boundary**: the verifier is a distinct service that the proposer cannot reach, and it is handed only the artifact and the declared store-slice it needs to rule.

7. **Status is richer than accept/reject.** The lifecycle is `proposed → {tentative | rejected | revised}` and then `tentative → {accepted | rejected | revised | superseded}`. `tentative` means an artifact cleared the cheap gates (for example, grounded and relevant) but not the hard ones (novelty, robustness, actionability): it is held but not yet trusted. `revised` records the third answer a gate can give — *recoverable*: the artifact is mostly sound but carries a localized, named defect, so the right move is neither accept nor discard but a tracked revision (§6, §7). Recording these states captures the *basis* for keeping (or fixing) something, not merely the fact of it, and lets context assembly prefer verified artifacts over provisional ones.

8. **Share the meta-schema; keep the actual types per-domain.** The kernel needs only that types have names and operations have typed signatures. It must not be generalized into one universal cross-domain ontology — that recreates the semantic-web tax for no benefit. The meta-schema is shared; the concrete types live in each domain's schema.

9. **Logical kernel, physical services.** The kernel is one logical contract realized as cooperating services (§3.3). Exactly one service — the **control plane** — may mutate durable state; it is deliberately *unintelligent*, doing mechanical orchestration and nothing more. The **agent** proposes from an isolated **sandbox** and never writes. The **verifier** judges *advisorily* — it returns a verdict; it does not commit. Intelligence lives in the agent and in the gates, never in the control plane. This separation is what lets the trustworthy machinery be hardened once while the unpredictable components are quarantined behind contracts.

## 3. Architecture: Two Views

The architecture is described in **two complementary views**, and both are normative. The **logical view** (§3.1–§3.2) divides the system by *responsibility*: a domain-invariant kernel and four per-domain extension points. The **physical view** (§3.3–§3.6) divides the same system by *deployment unit*: the cooperating services a coding agent actually builds, containers, and wires together. The logical view says what must be true; the physical view says where each truth is enforced and across which API boundary. Read them together — every service in the physical view is an assembly of logical responsibilities, and the mapping between them (§3.3) is explicit so neither view drifts from the other.

**Logical view.** The system divides cleanly into a **kernel** that is written once and never varies by domain, and four **extension points** that carry all the per-domain content. The dividing line is the whole point of the design: it is what lets the expensive, trustworthy machinery be built and verified once, while domains plug in through narrow typed seams.

```
┌──────────────────────────────────────────────────────────────────┐
│                          EXTENSION POINTS                          │
│  (per-domain; configured, registered, plugged in)                  │
│                                                                    │
│   Schema registry     Tool registry     Gate registry   Retrieval │
│   types + op sigs      typed plugins     bindings over   / planner │
│   + versioning         (fn + I/O types)  gate plugins    policy    │
└───────────┬───────────────┬─────────────────┬──────────────┬──────┘
            │               │                 │              │
            ▼               ▼                 ▼              ▼
┌──────────────────────────────────────────────────────────────────┐
│                              KERNEL                                │
│  (domain-invariant; written once)                                  │
│                                                                    │
│   Agent loop:   read ─▶ propose ─▶ gate ─▶ commit                  │
│                                                                    │
│   Commit path ──── enforces ───▶ Audit-contract invariants         │
│   Lifecycle state machine        Provenance & audit queries        │
│   Context assembly (bounded, regenerated each turn)                │
│                                                                    │
│   ┌────────────────────────────────────────────────────────────┐  │
│   │  Store (behind a backend-agnostic interface)               │  │
│   │  artifacts · operations · decisions · schema_versions      │  │
│   │  default reference impl: SQLite; git-on-disk JSON valid v0 │  │
│   └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.1 Logical view: the kernel (domain-invariant)

The kernel owns, and is solely responsible for:

- **The store** — artifacts, operations, decisions, and schema versions, behind an interface (§4).
- **The commit path** — the single privileged route by which a proposal becomes durable state: propose → run the type's gate pipeline → record the decision → set status → supersede any artifact it replaces (§7).
- **The audit-contract invariants** — the properties that must hold over the store at all times (§5).
- **The lifecycle state machine** — the legal status transitions and who may trigger them (§6).
- **Provenance and audit queries** — `get_provenance(id)`, "why do we believe X," and the rejected- and superseded-logs (§4.2).
- **Context assembly** — regenerating a bounded prompt context from the store each turn (§9).
- **The agent loop** — read → propose → gate → commit (§10).

### 3.2 Logical view: the extension points (per-domain)

| Extension point | Form | How much real work |
|---|---|---|
| **Schema registry** | Declarative: artifact types and typed operation signatures, plus schema versioning and migration. | Mostly data, but authoring a *good* schema is genuine modeling effort, and it churns. |
| **Tool registry** | Typed plugins: a function plus declared input/output types that reference the schema. | Per-domain but mechanical. |
| **Gate registry** | A declarative *binding* (type → ordered gate pipeline + thresholds + which gates require a human) layered over bespoke gate *plugins* of the form `(artifact, store-context) → {verdict, rationale, score?}`. | **The irreducible code.** Bindings are config; gate implementations are domain logic and are where the value lives. |
| **Retrieval / planner policy** | A pluggable strategy: which artifacts to preload, how to rank them, how a goal decomposes. | Pluggable, but a good policy carries domain knowledge. |

**Design rule (the meta-schema boundary).** The kernel is allowed to know only the meta-schema: that types have names and operations have typed input/output signatures. It must never be taught the *meaning* of a domain's types. Do not build a universal ontology that all domains map into. Share the meta-schema; keep the concrete types inside each domain's schema registry.

### 3.3 Physical view: the services

The logical kernel is realized as **three cooperating services** for the MVP, with a fourth split left as a documented seam. The decomposition follows Principle 9: exactly one service mutates state, the unpredictable components are quarantined behind contracts, and the control plane stays mechanical.

```
        ┌──────────────────────────────────────────────────────────────┐
        │                        CONTROL PLANE                         │
        │            sole mutator · unintelligent · async              │
        │  store · commit orchestration · lifecycle · context assembly │
        │  audit/provenance API · the four registries · task config    │
        │  orchestration policy (run-again / stop)                     │
        └──────┬──────────────────────────────────────────┬────────────┘
            ▲  │                                        ▲  │
   proposal │  │ assembled context             verdict │  │ shaped proposal
   + meta   │  │ · shape-error · refine        {accept │  │ + declared
            │  ▼ feedback                      |reject │  ▼ store-slice
            │                                  |refine} │
   ┌────────┴───────────────────┐      ┌────────┴───────────────────┐
   │           SANDBOX          │      │      VERIFIER SERVICE      │
   │  agent runtime + workspace │      │  advisory · queue-fronted  │
   │  ephemeral, rebuilt each   │      │  SDK of gate primitives    │
   │  cycle · proposes, never   │      │  sees only the allowlisted │
   │  writes durable state      │      │  store-slice               │
   └────────────────────────────┘      └────────────────────────────┘
            │
            │ tools · read-only data sources · contextual files (skills, SOPs, gold examples)
            ▼

   The sandbox and the verifier never talk directly — the control plane mediates every exchange.
```

**Mapping (logical responsibility → owning service).** Every logical responsibility from §3.1–§3.2 lands in exactly one service:

| Logical responsibility | Owning service | Note |
|---|---|---|
| Store, commit path, audit invariants, lifecycle, provenance/audit queries (§4–§7) | **Control plane** | The sole mutator. |
| Context assembly (§9) | **Control plane** | Assembled and served to the sandbox each cycle. |
| Schema / tool / retrieval registries (§8.1, §8.2, §8.4) | **Control plane** | Configuration the control plane reads. |
| Gate registry — **bindings** (§8.3) | **Control plane** | Resolves the pipeline, enforces "no implicit accept," and slices store-context to each gate's declared allowlist. |
| Gate registry — **gate plugins** (§8.3) | **Verifier service** | The primitives that actually judge. |
| Agent loop (§10) + LLM client | **Sandbox** | Proposes; never writes. |
| Tool execution (§8.2) + data-source access | **Sandbox** | Inside the ephemeral workspace. |
| Orchestration policy — run-again / stop | **Control plane** | New in this view; owns loop control (§3.4). |

The two-layer gate registry of §8.3 splits across the boundary on purpose: the **binding** (which gates, in what order, with which thresholds, which declared inputs) is control-plane configuration — it is where "no implicit accept" is enforced and where the store-slice is cut — while the **plugin** (the code that renders a verdict) is a verifier primitive. The control plane never judges; the verifier never decides *whether a type is allowed to be judged at all*.

**The seams (designed-in, not built for MVP).**
- **Runtime ↔ Workspace split (three → four services).** The sandbox fuses the agent runtime (the loop + LLM client) with the workspace (tool execution). They are kept as a clean internal seam so the workspace can later become its own service — useful when the runtime and the execution sandbox need independent scaling, or when one workspace image must serve several loop frameworks. The MVP fuses them because the loop makes many tool calls and a network hop per call is needless latency, and because workspace-regeneration and chat-flush are one ephemerality event (§3.5).
- **Parallel agents (one → many sandboxes).** Nothing in the contracts assumes a single sandbox. Multiple sandboxes can run against one control plane — *diversity* (different approaches to the same task) or *race* (the same approach, best-wins) — because status, provenance, and decisions are already per-artifact and attributable (§16). The MVP runs one sandbox against one task.

### 3.4 The control plane

The control plane is the spec's logical kernel minus the agent loop, plus the configuration surface and the loop-control policy. It is the **only service permitted to mutate durable state**, and it is deliberately **unintelligent**: it runs the mechanical commit protocol (§7), assembles context (§9), and applies a declarative orchestration policy — it hosts no LLM and makes no judgment calls. All intelligence is pushed outward, to the agent and the gates (Principle 9).

It is **async by design** and built for a single user at MVP while **designed for multi-tenancy and auth** (§16): the API is structured around a task identity from the outset so that tenant scoping and authorization slot in without reshaping it.

**The configuration surface (per task).** A task is configured, not coded. The control plane accepts and version-stamps:
- **Task instructions** — the discovery goal and any task-specific constraints. This is the per-task, user-authored *layer* of the composed system prompt (below); the user writes **intent, not paths**, because the kernel orientation layer already documents where everything lives. (The *domain* instruction layer is part of the domain definition, §8, not the per-task config.)
- **Proposal-shape spec** — what an acceptable proposal must be *composed of* (its required parts and types), **not** the quality of its contents. This is what the shape-error channel (§7) checks. A proposal may be text, structured objects, or object attachments (a script, a data file, or a serialized model — §3.5, §4).
- **Data sources** — the task's inputs, which vary widely (an SQLite database, a git repository, an organized corpus of publications). These are served to the sandbox **read-only**.
- **Additional contextual information** — Agent Skill files, SOP markdown, gold-standard examples, and the schema / tool / gate-binding / retrieval registrations of §8.

**The API surface** (names are the implementer's; the contract is):
- **Configure-by-task** — register or update a task's configuration surface above.
- **Proposal intake** — receive a shaped proposal plus its metadata (a summary of *how and why* the agent produced it) from the sandbox; respond with either a correctable **shape-error** or acknowledgement that the proposal entered the commit protocol. A proposal may include **object attachments** (a script, a data file, a binary artifact) that the agent has written to the workspace **outbox** (the workspace contract, below). At intake the control plane **harvests the outbox, content-addresses each object, and writes it to the object store (§4.2) — and it does this *before* the workspace is regenerated** (§3.5), because the ephemeral workspace is about to be destroyed. Harvest-before-teardown is a hard ordering constraint: the objects are gone if the sandbox is recycled first.
- **Verifier dispatch (internal)** — emit a shaped proposal plus its **declared store-slice** to the verifier and receive the verdict. This is the only path to the verifier; the sandbox has none.
- **Cycle control** — serve the assembled context to the sandbox; after a commit, apply the orchestration policy to decide **run-again** (issue a fresh cycle with regenerated context) or **stop**.
- **Extraction** — read final outputs, the context files (the accepted artifacts), full provenance/audit lineage, and the rejected / superseded / revised logs.

**The orchestration policy (loop control).** Continuation is the control plane's call, not the verifier's (a verifier judges *one artifact*; it does not plan the run). The policy is a declarative function of the accumulated verdicts, a budget, and explicit stop-conditions → `{run-again, stop}`. It also enforces the **refine cycle cap** (§6, §7) so a never-satisfiable artifact cannot loop forever.

**The workspace contract and the composed prompt.** The control plane owns two things that orient every agent identically, so a task author never has to predict where things live or how to return them.

*An invariant workspace contract.* The control plane provisions the sandbox to a **fixed directory skeleton** — fixed *roles and paths*, into which the per-task contents are mounted. The exact names are the implementer's; the contract is roles like these:

```
  data/          (read-only)            the task's data sources (§3.4 config)
  context/       (read-only)            skills, SOPs, gold-standard examples
  tools/         (read-only)            the task's registered tools (§8.2)
  spec/          (read-only)            the proposal-shape spec / output schema (§3.4)
  scratch/       (writable, ephemeral)  the agent's working space
  outbox/        (writable, ephemeral)  where the agent writes its proposal + object attachments
```

The *structure* is invariant; only the *contents* vary by task. Two consequences are load-bearing. First, the **read-only / writable split makes gold-data isolation (§3.5) physical**: the agent literally cannot mutate a gold data source, and only the writable regions exist to be discarded when the sandbox is regenerated. Second, the **outbox is the deterministic harvest source**: object harvest (§3.4 intake, §3.5) is simply "read the outbox," not "find the path the agent happened to name."

*A composed system prompt.* The control plane **mechanically assembles** the agent's system prompt — pure string assembly, no judgment, consistent with its unintelligence — from three layers:
1. **Kernel orientation** (invariant across all domains) — the read → propose → gate → commit loop, the propose-only contract, the workspace layout above, how to emit a proposal and write object attachments to the outbox, where to find the output schema, and how shape-errors and refine feedback come back.
2. **Domain instructions** (per-domain; part of the domain definition, §8) — e.g. "produce a self-contained script that reads `data/` and writes the model and description to `outbox/`."
3. **Task instructions** (per-task; the user's config above) — the specific goal, dataset, and constraints.

The kernel-orientation layer and the workspace-layout description are **stable**, so they belong in the context's stable prefix and preserve prompt caching (§9). The orientation preamble and the workspace contract are **kernel artifacts, versioned** like the schema (§8.1), so any change to "how the agent is oriented" is deliberate and recorded rather than ambient.

### 3.5 The sandbox (agent runtime + workspace)

The sandbox is where the agent loop (§10) runs and where every tool executes. It is the home of all the work and all the unpredictability, and it is **ephemeral by construction**.

- **Rebuilt each cycle.** After a proposal is resolved, the workspace is regenerated *and* the agent's chat history is flushed — one event. This guarantees the agent re-reads current context from the control plane (§9) rather than drifting on stale in-context state, and that any mutation of a gold-standard data source inside the workspace is discarded rather than propagated to the next cycle. Durable memory lives only in the store; the sandbox holds nothing across cycles.
- **Proposes; never writes.** The sandbox emits a proposal — a new artifact plus the operation that would produce it — together with **metadata** (the how/why summary) to the control plane. It cannot reach durable state; the commit path is the only writer (Principle 3, §7).
- **Carries objects out by reference, before teardown.** A proposal may include **object attachments** that live in the workspace filesystem — for the v1 domain, the submitted code (§12). The agent writes them to the **outbox** (§3.4); the control plane harvests the outbox at intake. Because the workspace is regenerated each cycle, the cycle order is fixed and load-bearing: **propose → the control plane harvests the outbox and runs the commit protocol → regenerate the sandbox** (§9). Objects must be pulled before regeneration, never after.
- **Receives corrections, not just rejections.** Two return channels flow back from the control plane: a **shape-error** (the proposal is malformed — fix the formatting and resubmit; cheap, not a recorded decision) and **refine feedback** (the proposal is mostly sound but carries named defects — produce a tracked revision; §6, §7).
- **Never contacts the verifier.** Every exchange with the verifier is mediated by the control plane. This is what makes the independence contract a network fact (§10).
- **Provisioned to the invariant workspace contract (§3.4).** The control plane builds the workspace to the fixed skeleton: read-only `data/`, `context/`, `tools/`, and `spec/`; writable-ephemeral `scratch/` and `outbox/`. Read-only mounts mean the gold copy is immutable regardless of what the agent does; the writable regions are exactly what regeneration discards. The agent writes its proposal and object attachments to the `outbox/`, the control plane's harvest source.

The agent runtime should be built from **open-source agent-loop tooling** (e.g., a LangChain DeepAgent, pi-mono, or openclaw) driving a **local or cloud LLM chosen per task** (a powerful cloud model for one task, a local model for another — a configuration choice, not an architectural one). The spec stays tooling-agnostic; it requires only that the loop honor the propose-only contract and the per-cycle ephemerality above.

### 3.6 The verifier service

The verifier is the physical home of the gate plugins (§8.3) and the reliability ladder (§11). It is **advisory**: it consumes a shaped proposal — plus the declared store-slice and any **object attachments** the control plane harvested (§3.4) — and returns a verdict — `{verdict ∈ accept | reject | refine, rationale, defects?, score?}` — and the control plane, not the verifier, acts on it. The verifier never writes to the store and never decides whether to continue the run. Object attachments are how a gate gets the material it must *execute* rather than merely read: in the v1 domain the code-runner receives the submitted code and the gate's own datasets (the agent's data plus the reserved verification set), runs the pipeline, and scores the resulting model (§12). The verifier is handed what it needs; it never reaches into the sandbox.

- **Queue-fronted and async.** It takes a queue of properly-shaped proposals via its API and emits properly-shaped verdicts; it does not need to know anything about the agent loop, the workspace, or the store beyond the slice it is handed.
- **An SDK of primitives.** The verifier is best built as a small SDK of composable gate primitives — an **auto-code runner**, a **model tester**, an **LLM-judge**, an **agentic grader**, and a **human-in-the-loop** prompt — each realizing a rung of the reliability ladder (§11). A domain composes its gate pipeline from these; a binding (held in the control plane, §3.3) names which primitives run in what order. In the simplest case a single primitive — even Claude Code answering over a CLI — satisfies the contract.
- **Independence is a network boundary.** The verifier is a different process from the proposer and receives **only** the artifact and the store-slice its binding declares — never the proposer's drafting rationale, the goal it chased, or sibling artifacts it competed against (§8.3, §10). It is the least standardizable component, which is exactly why it sits behind the narrowest contract: a queue in, a verdict out.

### 3.7 Prior art and reference substrate: ScienceClaw

The source paper is written about a real implementation — **ScienceClaw / CategoryScienceClaw** (Wang & Buehler; `github.com/lamm-mit/scienceclaw`) — and that system is the most useful prior art for this spec. A full evaluation is in [`references/scienceclaw-evaluation.md`](references/scienceclaw-evaluation.md); the load-bearing conclusion is summarized here because it shapes the spec.

ScienceClaw implements the **typed-provenance half** of this vision strongly: immutable, content-hashed artifacts with parent lineage, in a two-tier store (per-producer full-payload logs plus a payload-free shared index), with "open needs" as typed holes and pressure scoring as a plannerless retrieval policy. These are validated patterns and this spec borrows them (see §4, §8.4, §9–§10).

It implements the **verifier-aware-lifecycle half barely at all.** There is no status lifecycle and no commit gate: every successful skill invocation is written to durable state unconditionally, and the one quality rubric in the substrate is documented to *never gate*. The paper is explicit that this is the gap — the realized structure is "weaker than a software-enforced schema category, and should not be overclaimed," the coordination layer "does not yet solve a formal Kan-extension or lifting problem," and the publication map "is not yet a certified functor in software." The paper's own §3 names the next step: not to replace ScienceClaw, but to *lift* its displayed discipline into enforced discipline, so the system "could verify that … every public claim has an admissible artifact path, that retractions and supersessions preserve old evidence, and that new artifact types are introduced through recorded schema transitions."

**This spec is that enforced successor.** ScienceClaw is therefore both the proof that the typed-provenance substrate is buildable and the cautionary instance of Principle 4 — a sophisticated, 300-skill, multi-agent system that left the gate to convention and so cannot adjudicate what it believes. Where the gaps in that substrate sharpen a specific requirement here, this spec cites it inline as **[SC]**.

**[SC]** ScienceClaw is also prior art for the *physical* decomposition (§3.3): the system on disk is already three layers — an execution substrate (the skill registry and artifact store), a discourse substrate ("infinite," read as verifier signals), and the categorical layer over them. That a real implementation separated execution from verification-signal along roughly these lines is corroboration that the control-plane / sandbox / verifier split is a natural cut, not an invented one.

### 3.8 Reference point: pilar

A second, complementary reference point is **pilar** — a Claude Code plugin for medical writers that is *not* an implementation of this spec, but is a proven, in-use soft-gate system at the opposite pole from ScienceClaw: where ScienceClaw has strong provenance and no gate, pilar has a strong, enforced gate (independent context-restricted reviewers + deterministic commit-time validators) over a fixed deliverable schema. It is the closest working relative of the soft-gate domain (§14) and contributes one principle outright — the context-starved, statically-enforced verifier (§10). A full comparison is in [`references/pilar-comparison.md`](references/pilar-comparison.md); where it sharpens or confirms a requirement, this spec cites it inline as **[PILAR]**.

### 3.9 Deployment and portability

The kernel and any domain instantiation must be **containerizable with no host-specific assumptions**. Concretely: the system builds into container images; all configuration arrives through the environment (no values baked into the image, no reliance on a particular host path or user); and all durable state lives behind the Store interface (§4.2) in a mounted volume or an external store, never on a container's ephemeral filesystem. This is cheap to honor because the store is already backend-agnostic — the file-based v0 backends run as a single control-plane container with a mounted volume, and the Postgres path for concurrent or multi-agent writers (§4) is a second service in a Compose file rather than a rewrite. The spec does not prescribe a base image, an orchestrator, or a Compose topology; it requires only that nothing in the implementation defeats this portability.

**One container per service.** The physical decomposition (§3.3) is the deployment unit: the control plane, the sandbox, and the verifier each build into their own image and communicate only over their declared APIs. Two properties follow and must be preserved. First, the **sandbox is the only place untrusted code runs** — agent-driven tool execution and proposal generation are confined there, so the control plane (the privileged mutator) and the gold-standard data never share a process with the least predictable component. Second, the **control plane reduces to a gold image**: stateless except for the mounted store, it is the natural unit to harden, snapshot, and later replicate for multi-tenancy.

**Async and multi-tenancy are designed-in, not retrofitted.** The control-plane API is async and keyed by task identity from the first commit, so authorization and tenant scoping (§16) slot in without reshaping the surface, and the future move from one sandbox to many (§3.3) is a scaling change rather than a redesign.

**[SC]** The reference substrate (§3.7) ships an installer, a systemd service unit, and documented local/cloud/cluster deployment models — prior art that this requirement formalizes as a constraint rather than leaving to packaging convention.

## 4. The Store: Data Model and Contract

The store is the durable source of truth. It is reached only through a **backend-agnostic interface** so the backend can be swapped without disturbing the kernel. The interface is normative; the backend is a choice.

**Backends.** The default reference implementation is **SQLite** — a single-file relational store whose tables map directly onto the data model below, with zero infrastructure. **git-on-disk JSON** (one file per artifact/operation/decision, committed) is a defensible zero-infra alternative that leans into the "git for agent artifacts" framing and makes the audit log literally the commit history; it is a valid v0. **Postgres** is the path when concurrent or multi-agent writers arrive. The kernel must depend only on the interface in §4.2, never on backend specifics. **[SC]** The append-only-JSONL-plus-content-hash backend is validated prior art: the reference substrate (§3.7) realizes exactly this — per-producer full-payload logs plus a payload-free shared index for fast cross-agent scanning — which is also the concrete model for the manifest split in §9–§10.

### 4.1 The data model

The conceptual schema, expressed relationally (a backend may realize it differently so long as the interface and invariants hold):

```
artifacts(id PK, type, payload, status, superseded_by, revised_by, created_by, created_at)
    status  ∈ {proposed, tentative, accepted, rejected, superseded, revised}
    payload := inline value | {blob_ref, content_hash}   -- object/binary payloads held by reference

operations(op_id PK, op_name, parents[], output_id, status, created_at)
    status ∈ {success, failed, retried}      -- the typed provenance edges

decisions(artifact_id, gate, verdict, rationale, defects?, score?, created_at)
    verdict ∈ {accept, reject, refine}

schema_versions(version, types[], op_signatures[], created_at)
```

- **artifacts** are the nouns: a typed payload with a lifecycle status, a creator, and — once replaced — a pointer to the artifact that replaced it (`superseded_by` when an accepted artifact is beaten; `revised_by` when a `refine` verdict spawns a tracked revision, §6–§7). A payload may be an **inline value or a reference** (`blob_ref` + `content_hash`) to a content-addressed object in the object store (§4.2), so a proposal that carries code or a binary artifact fits the relational/JSON backends: the store holds typed metadata plus the hash and pointer, and the bytes live in the object store. **Object bytes follow a retention policy:** the typed metadata, the gate decision, and the `content_hash` are *always* retained — the record of what was tried is never lost (§5.5) — while the bytes are persisted for objects belonging to artifacts that reach durable state and are garbage-collectable once referenced only by terminal `rejected`/`superseded` artifacts. ("Persist them if useful": keep the record always; keep the bytes when they earn their storage.)
- **operations** are the typed provenance edges: a named operation with zero or more `parents` (artifact ids) and one `output_id`, recording whether it succeeded, failed, or was retried. These edges, taken together, are the provenance DAG. A **`revises`** operation (parent = the flagged artifact, output = its revision) is how a refine is recorded as provenance-bearing work rather than an in-place edit.
- **decisions** are the gate verdicts: for an artifact, which `gate` ruled, its `verdict` (`accept`, `reject`, or `refine`), the `rationale`, an optional numeric `score` where the gate produced one, and — on a `refine` — an optional `defects` list naming the *localized* failures the revision must address. An artifact accumulates one decision row per gate that ruled on it.
- **schema_versions** record the registered schema over time: the set of types and operation signatures in force, each version stamped.

### 4.2 The Store interface

The kernel interacts with the store through a small, typed interface. The exact method names are the implementer's, but the contract is:

- **Reads.** Fetch an artifact by id; query artifacts by type and/or status; fetch the operations into or out of an artifact; fetch the decisions on an artifact; read the current and historical schema versions.
- **propose(artifact)** — write a new artifact in status `proposed`, with a stable id, attributed to its creator, linked to the operation that produced it. This does not accept anything; it records an intention.
- **commit(...)** — *not* a raw write. The commit path (§7) is the only privileged mutation route; it is what runs the gate, records decisions, and sets status. The store exposes the primitive status-and-decision writes that the commit path composes, but those primitives must never be reachable from domain code or from the agent.
- **supersede(old_id, new_id)** — mark `old_id` as `superseded` and set its `superseded_by` to `new_id`, atomically with the acceptance of `new_id`.
- **revise(old_id, new_id)** — mark `old_id` as `revised` and set its `revised_by` to `new_id`, atomically with recording the revision that a `refine` verdict requested (§6, §7). The original is retained, never edited in place.
- **put_object / get_object** — write and read a binary object **by content hash**, behind the same backend-agnostic interface (a directory of hash-named files for the file backends; an object store or large-object column otherwise). Artifact payloads reference objects via `blob_ref` + `content_hash` (§4.1) instead of inlining bytes, so a large or binary payload — the v1 submission's code, and future models or datasets — lives in the object store while the typed metadata lives in the artifact row.
- **Provenance/audit queries** — `get_provenance(id)` returns the transitive operation-and-artifact lineage behind an artifact; "why do we believe X" returns the decisions and the lineage together; the rejected- and superseded-logs return those histories for audit and for trial-count accounting (§12).

All writes are append-oriented (§5): the interface offers no operation that overwrites an accepted artifact's payload or hard-deletes an accepted artifact.

## 5. Audit-Contract Invariants

These are the properties that must hold over the store at all times. They are the audit contract. Each is stated so that it can become a **property test or an acceptance test** — an implementation is expected to ship these as executable checks, not as prose aspirations. This list is the paper's own audit contract (§2.3 of the source) made enforceable; the reference substrate (§3.7) holds several of these by *convention* rather than by enforcement, and the `[SC]` notes below mark where that distinction bites.

**[PILAR]** These checks need not wait on a runtime kernel to be real. A proven soft-gate system enforces an equivalent contract with **deterministic validator scripts run on every commit** — asserting ID format, uniqueness, and append-only assignment; flagging any claim whose cited source does not resolve; and confirming each verifier's input isolation (see [`references/pilar-comparison.md`](references/pilar-comparison.md)). When the store is files and the agent can technically write to them, a deterministic commit-time gate that refuses to record an invariant-violating state is the pragmatic enforcement mechanism — stronger than convention, and available before any of the kernel is built.

1. **Stable, unique, never-reused ids.** Every artifact and operation has a stable identifier — a UUID or a content hash — that is never reused for a different entity.
2. **Append-only payloads.** The payload of an accepted artifact is never overwritten. A change is a *new* artifact plus the old one marked `superseded` with `superseded_by` set. **[SC]** Immutability must be *enforced* (frozen types; writes only through the commit path), not merely documented: the reference substrate ships a mutable artifact record whose docstring claims immutability, and a convention like that erodes silently.
3. **No hard delete of accepted artifacts.** Removal is soft, via status only. Accepted history is never destroyed.
4. **Every non-root accepted or tentative artifact has provenance.** Such an artifact has at least one operation edge whose `parents` are recorded. Root artifacts (raw inputs) are the only exception and are typed as such.
5. **Failures are recorded, not erased.** Operations that failed or were retried remain in the `operations` table with the appropriate status. The history of what was tried is part of the record.
6. **No silent merge.** Two distinct accepted artifacts are never merged into one without an explicit, recorded operation. Identity is not quietly collapsed. **[SC]** A "newer value wins on key conflict" merge — the reference substrate's conflict-resolution rule — is exactly the anti-pattern this forbids: it collapses contradicting artifacts by timestamp rather than by a recorded, gated decision.
7. **No implicit accept.** Committing an artifact of a type that has no declared gate is an error, not a default-accept. This is the enforcement of Principle 4 and is non-negotiable. **[SC]** This is the invariant the reference substrate most conspicuously lacks: there, every successful tool call is written to durable state unconditionally, which is precisely how a well-audited store fills with ungated content.
8. **Reproducible gate verdicts.** A gate run on the same artifact and the same store-context yields a stable verdict. Stochastic gates achieve this by fixing seeds or by averaging over a declared number of runs; either way the verdict is reproducible and the method is recorded.

## 6. The Lifecycle State Machine

Artifact status is an explicit, testable state machine. It exists so that the *basis* on which an artifact is held is legible from its status alone, and so context assembly can prefer trusted artifacts.

```
                 cheap gates pass            hard gates pass
   ┌──────────┐  (grounded, relevant)  ┌───────────┐  (novel, robust,   ┌──────────┐
   │ proposed │ ─────────────────────▶ │ tentative │ ─ actionable, …) ─▶│ accepted │
   └────┬─────┘                        └─────┬─────┘                    └────┬─────┘
        │ any gate rejects                   │ a hard gate rejects           │ replaced by
        ▼                                    ▼                               ▼ a better artifact
   ┌──────────┐                         ┌──────────┐                    ┌────────────┐
   │ rejected │                         │ rejected │                    │ superseded │
   └──────────┘                         └──────────┘                    └────────────┘

   refine (recoverable: a localized, named defect) — from proposed or tentative:
   ┌──────────────────┐  gate verdict = refine   ┌─────────┐  revises op  ┌───────────┐
   │ proposed /       │ ───────────────────────▶ │ revised │ ───────────▶ │ proposed  │
   │ tentative        │  defects[] recorded      │ (revised_by → ───────▶ │ (revision)│
   └──────────────────┘                          └─────────┘              └───────────┘
```

| From | To | Trigger | Who |
|---|---|---|---|
| (none) | `proposed` | Agent emits a proposal via `propose` | Agent (proposer) |
| `proposed` | `tentative` | Cleared the cheap/early gates in the type's pipeline but the pipeline is not exhausted | Commit path |
| `proposed` | `rejected` | Any gate in the pipeline rejects | Commit path |
| `proposed` | `revised` | A gate returns `refine` (recoverable, with `defects`); a tracked revision is requested | Commit path |
| `tentative` | `accepted` | Cleared the remaining (hard) gates in the pipeline | Commit path |
| `tentative` | `rejected` | A hard gate rejects | Commit path |
| `tentative` | `revised` | A gate returns `refine` (recoverable, with `defects`) | Commit path |
| `tentative` | `superseded` | Replaced by a better artifact before it was ever accepted | Commit path |
| `accepted` | `superseded` | Replaced by a better artifact of the same type | Commit path |

Notes that bind the machine:

- **Only the commit path triggers transitions** out of `proposed`. The agent can create a `proposed` row; it cannot move one.
- **`tentative` is a real, valuable state.** It records "true and relevant, but not yet shown to be worth keeping." A type whose gate pipeline is entirely cheap may move `proposed → accepted` in one commit; a type with hard gates passes through `tentative`. The split between cheap and hard gates is declared in the gate binding (§8.3).
- **`rejected`, `superseded`, and `revised` are terminal and retained.** They are never deleted (§5.3) and remain queryable (the rejected- and revised-logs feed trial-count accounting in §12).
- **`revised` is the recoverable verdict made legible.** It records "mostly sound, but carrying a named defect — fixed by a tracked revision," and it points to that revision via `revised_by`, exactly as `superseded` points via `superseded_by`. The revision enters as a fresh `proposed` artifact produced by a `revises` operation (§4.1) and runs the gate pipeline again from the top; nothing is edited in place. A `refine` verdict is **bounded by the orchestration policy's refine cycle cap** (§3.4, §7) so an un-satisfiable artifact cannot loop forever — once the cap is hit, the lineage terminates in `rejected`.
- **Supersession and revision are atomic with their cause.** An artifact becomes `superseded` only as the same operation accepts its replacement (`superseded_by` set, §5.2); it becomes `revised` only as the revision is recorded (`revised_by` set).

## 7. The Commit Path

The commit path is the kernel's single privileged route from a proposal to durable state. It is the mechanical heart of the audit contract: every guarantee in §5 and §6 is enforced here, so that neither the agent nor domain code can bypass it.

In the physical architecture (§3) the commit path is a **control-plane-orchestrated protocol** that spans services — but the orchestration changes nothing about who may mutate state. The control plane runs every step below and performs every write; the verifier is **advisory** (it returns verdicts, step 3) and the sandbox only proposes. The protocol is distributed; the authority is not.

Given a `proposed` artifact of type `T`:

0. **Validate shape (pre-gate).** Check the proposal against `T`'s proposal-shape spec (§3.4): is it *composed of* the required parts and types? If not, return a correctable **shape-error** to the sandbox and **record nothing** — a malformed proposal is a formatting failure to be fixed and resubmitted, not a gated decision, so it never enters the rejected-log or the trial count (§12). This is the cheap, fast correction loop; the quality question begins only once the shape is valid.
1. **Resolve the gate pipeline.** Look up the gate binding for `T` in the gate registry. **If `T` has no declared gate, fail** — this is the "no implicit accept" invariant (§5.7), surfaced as a hard error with a clear message, not a silent accept.
2. **Enforce proposer ≠ gate.** In the service topology this is satisfied structurally — the verifier is a different process the proposer cannot reach (§3.6, Principle 6) — and the control plane additionally asserts that no gate's identity equals the producer's. A gate that is an LLM-judge must be a different model instance or carry an adversarial brief; a deterministic gate trivially satisfies this.
3. **Run the pipeline in order, advisorily.** For each gate, the control plane dispatches the artifact plus that gate's **declared store-slice** (and nothing else, §8.3, §10) to the verifier service and receives `{verdict, rationale, defects?, score?}`. The control plane records a `decisions` row for every gate that rules. On the first `reject`, set the artifact to `rejected` and stop. On a `refine`, go to step 5b.
4. **Advance status by pipeline position.** When the cheap/early gates have passed but hard gates remain, set `tentative`. When the whole pipeline has passed, set `accepted`.
5. **Resolve replacement, if any.**
   - **(a) Supersede.** If this artifact replaces an existing accepted artifact of the same type (the proposal or its selection gate names the incumbent), mark the incumbent `superseded` with `superseded_by`, atomically with this acceptance, preserving lineage.
   - **(b) Refine.** On a `refine` verdict, set the artifact `revised`, attach the gate's `defects` to the decision row, and return that defect list to the sandbox through the cycle's feedback channel (§3.5). When the agent re-proposes, the revision is recorded via a `revises` operation with `revised_by` set (§4.1) and re-enters the pipeline at step 0 — bounded by the orchestration policy's **refine cycle cap** (§3.4); on exhausting the cap the lineage terminates in `rejected`.
6. **Record provenance.** Ensure the operation edge(s) that produced the artifact are present with their `parents` (§4.2), so the accepted/tentative artifact is never provenance-less.

The commit path is deterministic given its inputs and the gate verdicts, and it is reproducible (§5.8). It is the only code permitted to set status or write decisions; the verifier advises but never writes.

## 8. The Extension Points

The four extension points are the entire per-domain surface. Each is a typed contract; the kernel manipulates them only through these contracts.

### 8.1 Schema registry

The schema registry declares, for a domain, the **artifact types** (each a name plus a payload shape) and the **operation signatures** (each a name plus typed inputs → typed output, referencing the artifact types). It is versioned: the registered schema at any time is a `schema_versions` row. Authoring a schema is declarative, but authoring a *good* schema is real modeling work and the schema churns as the domain is understood — which is why versioning and, eventually, migration (§15) are first-class. The kernel reads only the meta-schema (names and signatures); it never interprets payload semantics.

### 8.2 Tool registry

A **tool** is a typed plugin: a function plus a declared input/output signature whose types reference the schema. A tool consumes artifacts of declared types and produces an artifact of a declared type, and its invocation is recorded as an `operations` row (with `parents` = the input artifact ids, `output_id` = the produced artifact, and a success/failed/retried status). Tools are per-domain but mechanical; they carry no acceptance authority — a tool produces a *proposal*, and only the commit path can accept it.

**[SC]** The signature must be typed on *both* sides — inputs and output. The reference substrate (§3.7) declares only output types and infers inputs by scraping a tool's `--help` text, which reduces composability to string key-overlap between payloads; declared input types are what make operation composition checkable rather than coincidental.

### 8.3 Gate registry

The gate registry has two layers, and the separation matters:

- **Gate plugins** (code, the irreducible work) — each a function `(artifact, store-context) → {verdict, rationale, score?}`. A plugin may be a deterministic check, a heuristic, an LLM-judge, an adversarial Breaker, or a human-in-the-loop prompt. The `store-context` argument is what lets a gate reason against existing state — incumbents to beat, prior frames, the rejected-log.
- **Gate bindings** (config) — for each artifact type, an **ordered pipeline** of gates, the **thresholds** they apply, the split between cheap/early gates (clearing them earns `tentative`) and hard gates (clearing them earns `accepted`), and a **requires-human** flag on any gate that must not auto-resolve. Composing gates into a per-type pipeline is configuration; what each gate *decides* is domain logic.
- **Declared inputs (the independence contract)** — each gate declares the *exact* set of store inputs it may see, and the binding withholds everything else. A gate is given the artifact and the slice of store-context it needs to rule, and nothing more — never the drafting rationale, the proposer's reasoning, or sibling artifacts that would let it rationalize rather than judge. This declaration is itself checkable (see §10).

The gate registry is where the "no implicit accept" rule is satisfied or violated: a type with no binding cannot be committed.

### 8.4 Retrieval / planner policy

A pluggable strategy of the form `(goal, store) → context bundle`: which artifacts to preload for a step, how to rank them, and how a goal decomposes into sub-goals. The kernel provides a default (recency- and status-aware retrieval that prefers `accepted` over `tentative`), and a domain may replace it with a policy that carries domain knowledge. The policy feeds context assembly (§9) but does not change the bounded-context guarantee.

## 9. Context Assembly

Context assembly is how the volatile context window is regenerated from the durable store each turn (Principle 2). It is a **control-plane responsibility** (§3.4): the control plane assembles the bounded context and serves it to the sandbox at the start of each cycle — the sandbox never accumulates its own durable context. It carries the spec's one hard performance guarantee.

**The split.** Each turn's context is assembled from two parts:

- A **stable prefix** — the composed system prompt (§3.4), the tool definitions, and a slow-changing manifest (a compact, ranked index of what the store holds, not the artifacts themselves). The composed prompt's **kernel-orientation and workspace-layout layers are invariant**, and its domain/task instruction layers change rarely, so the prefix changes rarely — which is deliberate: it preserves prompt caching.
- A **volatile tail** — the artifacts retrieved for *this* step (via the retrieval policy, §8.4), the current goal, and a short rolling scratch. The tail is rebuilt every turn.

**The manifest** is the bridge. It is a bounded, ranked summary of the store — enough for the agent to know what exists and to ask for it, never the full contents. The retrieval policy reads the manifest and the store to populate the tail.

**Two rules make this a discipline rather than a habit:**

- **Regenerate, never append.** Context is rebuilt from the store each turn. The previous turn's context is not carried forward and grown; that is what a transcript does, and it is the failure mode this design rejects. In the service topology this is enforced at the strongest possible boundary: between cycles the **sandbox is rebuilt** — its workspace regenerated and its chat history flushed together (§3.5) — so there is no carried-forward context to grow, and any in-workspace mutation of a gold-standard data source is discarded rather than propagated.
- **Flush on commit.** The act of committing an artifact is what moves it from volatile scratch into durable state. Until committed, work lives in the tail and is disposable; once committed, it is retrievable from the store and need not be held in context.

**[PILAR]** This "the store is canonical, the projection is regenerated" rule is not unique to the prompt context — a proven soft-gate system applies it to the *deliverable* too: its consolidated document is deterministically regenerated from the canonical source artifacts on every assembly and is never edited in place; findings are fixed at the source and the view is rebuilt (see [`references/pilar-comparison.md`](references/pilar-comparison.md)). The same discipline that keeps context bounded keeps every derived view trustworthy.

**The bounded-context guarantee (a test, not a hope).** The assembled context size must not grow with the size of the store. A long run that accumulates thousands of artifacts must assemble a context no larger than a short run did. This is an acceptance criterion (§13) and must be demonstrated by running the agent against a deliberately large store and measuring the assembled context.

## 10. The Agent Loop and Builder/Breaker Separation

The agent operates a single, repeating loop, and the loop's shape is itself a discipline:

```
read  ─▶  propose  ─▶  gate  ─▶  commit
 ▲                                   │
 └───────────────────────────────────┘
```

1. **Read.** The agent receives the assembled context (§9) served by the control plane: the manifest, the retrieved artifacts, the goal.
2. **Propose.** From the sandbox (§3.5), the agent emits a proposal — a new artifact plus the operation that would produce it — together with **metadata** summarizing how and why it got there. It proposes; it does not write, and it never contacts the verifier.
3. **Gate.** The control plane validates the proposal's shape, then runs the type's gate pipeline against the advisory verifier (§7).
4. **Commit.** The control plane records the decision and sets status. A malformed proposal returns a correctable **shape-error** (nothing recorded); a `reject` becomes a `rejected` row with a rationale; a `refine` returns the named defects for a tracked revision (§6, §7); an `accept` becomes durable state.

**Builder/Breaker is enforced, not encouraged (Principle 6).** The agent that *builds* a proposal is never the authority that *judges* it. The commit path checks this (§7, step 2). In practice a gate is one of: a deterministic checker (no model judgment at all); a *different* model instance acting as judge; or a "Breaker" — a model given the adversarial brief of finding the disconfirming evidence or the alternative explanation that should sink the artifact. The Breaker pattern is the soft-domain workhorse (§14) and exists precisely because a confirmatory self-review is worthless.

**The independence contract: a gate must be denied context, not merely be a different agent. [PILAR]** Separation of identity is necessary but not sufficient — a verifier that can see the proposer's reasoning, the goal it was chasing, or the sibling artifacts it is competing against will rationalize the proposal rather than judge it. So each gate is given only the inputs it declared (§8.3): the artifact under test and the minimal store-slice it needs to rule, and nothing else. This input restriction is **itself a checkable property**, in the same family as the audit-contract invariants (§5): a static check confirms that each gate's resolved inputs are a subset of its declared allowlist, so an over-broad gate fails before it ever runs. The discipline is drawn from a working soft-gate system whose independent reviewers each see only their permitted inputs, enforced by exactly such a static audit (see [`references/pilar-comparison.md`](references/pilar-comparison.md)). In the physical architecture the restriction is *also a network boundary*: the verifier is a separate service reachable only by the control plane, which transmits exactly the declared store-slice and nothing else (§3.6) — so an over-broad gate cannot even receive what it was not granted.

## 11. The Reliability Ladder for Gates

A gate answers two questions — *validity* and *selection* (Principle 5) — and the **selection** question is the hard one. Where a numeric scoring functional exists, the design commits to it up front and the decision is mechanical: an artifact earns its place if and only if it improves the score net of its added cost. The paper's MDL/AIC gates are exactly this — "does it improve the model net of added complexity." This is objective, which is why the v1 reference domain (§12) is built on such a gate.

Where no scoring functional exists, the implementer descends a **reliability ladder**, and the discipline is to descend it *consciously* and *record the cost*, never to disguise a weaker rung as a stronger one:

1. **Numeric scoring functional** — a closed-form score the artifact must improve. Most trustworthy; mechanical.
2. **Deterministic check** — a pass/fail predicate computed from the artifact and store. Trustworthy where it applies.
3. **Heuristic** — a coded rule of thumb. Cheap, fallible, honest about being a heuristic.
4. **LLM-judge** — a model verdict with a rationale. The **weakest gate that may ship as a default, and never unlabeled**: a default gate at this rung must be explicitly marked as weak so no one mistakes a judged accept for a measured one.
5. **Human** — a person rules (the `requires-human` flag, §8.3). The legitimate home of the hard call in fuzzy domains.
6. **Community** — adjudication by many parties over time. The most expensive; reserved for claims nothing cheaper can settle.

**Default gate policy.** The kernel ships no gate that auto-accepts. The weakest *default* it will provide is a labeled-weak LLM-judge; anything stronger is domain-supplied. The cheap gates (validity hygiene — grounded, relevant, well-typed) should be made *free* so they always run; the expensive selection gate is where human judgment legitimately re-enters, and the harness's job there is to structure the evidence so the human's call is cheaper and to log every rejection (§12).

**Where the ladder lives.** Physically, each rung is a **verifier primitive** (§3.6): the numeric/deterministic rungs are an auto-code runner or a model tester, the LLM-judge and agentic-grader rungs are model primitives, and the human rung is the HITL prompt. A binding composes a type's pipeline from these primitives (§3.3). The **`refine`** verdict is available at any rung that can *localize* a defect — a deterministic check that names the failing field, or a judge instructed to return per-part findings — and it earns its place precisely on **expensive or composite artifacts**, where accept would launder a known error into durable state and reject would throw away sound, costly work (a five-part analysis with one bad part; a trained model sound but for one leaking feature). There, "fix this part" is the right third answer (§7).

**[SC]** A gate's verdict must *bind* the lifecycle. The reference substrate (§3.7) ships a quality rubric that scores an artifact on a 0–100 scale and then, by its own documentation, "never gates" — a score that does not move an artifact's status is decoration, not a gate. A rung on this ladder counts only if its verdict is what the commit path acts on.

**[PILAR]** Split the structural check from the semantic one. A proven soft-gate system runs a *deterministic* structural gate (does every claim resolve to a real source? — answerable by code) ahead of a *semantic* gate (is the claim overstated, misattributed, or weakly supported? — an LLM-judge or human call). This is the cheap-gates-then-hard-gates lifecycle (§6) in practice: the deterministic rung earns the `tentative` (grounded, well-formed) pass for free and always runs; the higher rung makes the worth-keeping call. Keeping them distinct stops a cheap, certain check from being conflated with an expensive, fallible judgment.

## 12. Reference Domain — Feature Engineering (v1, the validation target)

The first domain to instantiate is a **feature-engineering agent**, chosen deliberately *because its gate is objective*. It proves the harness end-to-end without entangling the system bring-up with soft-gate ambiguity. This is the v1 validation target.

**The task, concretely.** The agent is given a **dataset** — a CSV in which one named column is the target `y` and the rest are candidate features — optionally with notes on what the columns mean, and is asked to **clean the data and design 1–5 new features that improve a boosted random-forest model.** Its deliverable each cycle is a **self-contained script** that cleans the data, engineers the features, and trains the model, plus a **JSON-schema'd description** of the features it added and why — both written to the `outbox/` (§3.4), from which the control plane harvests the code object. It does *not* return a trained model: the model is produced by the gate, from the script, on data the agent never sees (below). The task is deliberately recognizable — a Kaggle-shaped problem — so the bring-up exercises the kernel against work a practitioner would recognize, not an abstract transform calculus.

**Schema.**
- `DatasetVersion` — the pinned dataset (the CSV, the named target column, fixed folds), plus — **held by the gate and never exposed to the agent** — a **larger reserved verification dataset**. Root artifact.
- `Submission` — the agent's proposal for a cycle and the **gated** artifact: an **object-bearing** payload that references the **code** (a `blob_ref` + `content_hash` over the self-contained script, §4.1) together with the structured description (the 1–5 engineered features, each with a name, definition, and rationale) conforming to the proposal-shape spec (§3.4).
- `Feature` — **harvested by the harness** from the submission's description, one typed artifact per declared feature, linked to its `Submission` by a `harvest` operation (§4.1). Harvesting keeps the provenance DAG real — each feature is an inspectable node — and lets a `refine` verdict point at a specific feature, without asking the agent for any extra ceremony beyond the description it already writes.

**Tools.** The workspace's read-only access to the dataset, a script-execution capability, and the standard ML libraries — *not* a closed transform library. Free-form agent code is the point; the gate, not a constrained tool surface, is what keeps the result honest.

**Gate (objective, the MDL/AIC analog).** The gate is split across two artifact types:
- **`Submission` — cheap/structural (earns `tentative`).** The submission is well-shaped (code present, description conforms) *and the code runs*: the verifier executes it on the agent's dataset without error and produces a model. Deterministic rung.
- **`Submission` — hard/selection (earns `accepted`).** The model the verifier **trains from the submitted code** improves performance over the incumbent **net of added complexity**, scored on the **reserved verification dataset**. This is the MDL/AIC-analog selection decision, made mechanical.
- **`Feature` — grounding (cheap, deterministic).** Each harvested feature must be genuinely *defined by the submitted code* (not a described-but-absent feature). This trivial structural check is the harvested type's declared gate, so "no implicit accept" (§5.7) holds for it too.
- **`refine`.** When the code runs and most features help but one or more declared features are harmful, leaky, or error, the gate returns `refine` with `defects` naming the offending `Feature` ids (§7) — the recoverable case, since a five-feature submission with one bad feature is worth fixing, not discarding.

The protocol is:

- **Leakage-safe by who-holds-what.** The agent never sees the reserved dataset and never produces the judged model. The verifier trains from the submitted code on the agent's data and scores on the reserved data, so a "feature" that peeks at the target inflates the agent's own cross-validation but **fails to generalize and is caught on the reserved set**. No closed transform space is needed, and there is no delivered model to trust.
- **Seed-fixed and reproducible** (§5.8), so the same submission yields the same verdict.
- **Trial-count-aware.** High query volume against one validation signal fits it by selection — adaptive overfitting, the genuinely agent-specific risk. The mitigation is built into the store: **the `rejected` log is the trial count.** The selection gate reads the rejected-log via store-context (§8.3) and **deflates significance by the number of submissions already evaluated**, so the hundredth submission must clear a higher bar than the first.

**Open seam (refine and the trial count).** The `refine` verdict (§6, §7) interacts with this accounting, and the interaction is deliberately *flagged, not silently decided* here. A refine that sends a submission back to be re-coded and re-evaluated still consumes a query against the same validation signal, so it must count toward the trial total or it becomes a loophole around the significance deflation. The recommended default is to **count each refine attempt in the trial total** (a `revised`/refine counter read alongside the rejected-log), with the orchestration policy's refine cycle cap (§3.4) keeping any one lineage's contribution bounded; the v1 implementer should confirm this against the concrete statistics before locking it in.

This domain exercises every kernel mechanism — typed provenance, the commit path, supersession, the bounded context, and the object-handoff lifecycle (§3.4–§3.6) — against a gate whose correctness is not in question, which is exactly what a bring-up needs.

## 13. Acceptance Criteria (v1)

The v1 build is complete when the feature-engineering domain demonstrates all of the following, each as a runnable check:

1. **Append-only history.** Across a run, no accepted artifact's payload is ever overwritten; changes appear as new artifacts plus supersession (§5.2).
2. **A submission rejected and retained.** A submission that fails the objective gate becomes a `rejected` row with a rationale and is queryable afterward (it is not erased).
3. **A submission superseded with intact lineage.** A better submission replaces an incumbent; the incumbent is `superseded` with `superseded_by` set, and `get_provenance` still returns the full lineage of both — including their harvested features (§4.2).
4. **"Why was this submission accepted" is answerable from provenance.** For any accepted submission, the system returns the operations that produced it (including the feature harvest) and the gate decisions (with scores and rationales) that accepted it — entirely from the store, with no recourse to a transcript.
5. **Bounded context across a long run.** Run the agent against a deliberately large store and show the assembled context size does not grow with store size (§9).
6. **Refusal to commit a gateless type.** Register a type with no gate binding, attempt to commit an artifact of that type, and confirm the commit path errors rather than accepting (§5.7, §7).
7. **Malformed proposal is correctable, not recorded.** Submit a proposal that violates its shape spec and confirm the control plane returns a correctable shape-error with **no** `decisions` row and no trial-count increment; a corrected resubmission then proceeds normally (§3.4, §7 step 0).
8. **Refine produces a tracked revision with intact lineage.** Drive a gate to a `refine` verdict on a composite artifact; confirm the artifact becomes `revised` with `revised_by` set and a `defects` list recorded, the revision enters via a `revises` operation, and `get_provenance` returns the full chain across the revision (§6, §7).
9. **Privileged-mutator boundary holds.** Confirm that durable state changes only via the control plane: the sandbox has no write path to the store and no path to the verifier, and a direct write attempt from sandbox code fails (§3.3, §3.5, Principle 3/9).
10. **Object attachment round-trips and is executed, not trusted.** Submit a proposal carrying a code object: confirm the control plane **harvests it from the sandbox before the workspace is regenerated**, content-addresses it (`blob_ref` + `content_hash`), and hands it to the verifier, which **runs the code** to train and score a model (no model is stored) — and that the object is retrievable by reference afterward (§3.4–§3.6, §4.1–§4.2).
11. **Leakage is caught on the reserved set.** Submit code whose engineered feature leaks the target; confirm its optimistic agent-side score does not carry, because the judged model is the one the verifier trained and scored on the **reserved verification dataset** the agent never saw (§12).
12. **Sandbox is ephemeral between cycles.** Mutate a gold-standard data source inside the workspace, complete a cycle, and confirm the next cycle's regenerated workspace shows the unmutated gold copy and a flushed chat history — no leakage across cycles (§3.5, §9).

## 14. Second Reference Domain — Consulting / Soft Gate (post-v1)

The second domain is a **consulting / insight agent**, and it exists to exercise the parts of the lifecycle the objective domain does not: the `tentative` state, the adversarial Breaker gate, the client-prior-as-artifacts pattern, and human-in-the-loop promotion. It is **post-v1** and is specified here so the kernel leaves room for it, not so it is built first.

**Schema.** `Source → Clue → Insight → Recommendation`, where `synthesize` is a multi-parent operation (an `Insight` drawn from several `Clue`s).

**Cheap auto-gates (earn `tentative`, never `accepted`).** **Grounding** — every cited clue is real and says what the insight claims it says. **Relevance** — the insight bears on the client brief. Clearing both establishes "true and relevant" and moves the artifact to `tentative`. This is hygiene, made free so it always runs.

**The worth-keeping gate (earns `accepted`).** A composite of **novelty** versus the client's prior frame, **actionability** ("so what — does it change a decision?"), and **robustness under adversarial search**. The dangerous failure mode here is the cherry-picked, narrative-driven insight that is merely "somewhat supported," so the support gate must be **adversarial**: a **Breaker** that actively searches for disconfirming clues and alternative explanations, not a confirmatory re-read. This gate typically carries the `requires-human` flag.

**Client-prior-as-artifacts.** To make *novelty* computable, the client's existing frame is represented as artifacts in the store. An insight's novelty is then measured against concrete prior-frame artifacts rather than a vibe. The highest-value output is the **reframe** — a shift in the client's frame, which is a regime transition (§15) wearing a strategy hat.

**The generalization this domain teaches.** Cheap gates establish *true + relevant* (hygiene); the worth-keeping gate is selection/novelty/robustness and is where human judgment legitimately re-enters. The harness's contribution is to make the cheap gates free, to structure the evidence so the human's hard call is cheaper, and to log the rejections.

## 15. Regime Transition / Schema Migration (deferred, seamed)

Day to day, the schema is fixed. A **regime transition** is what happens when a genuinely new type, operation, or verifier is needed — the system's schema itself revises. This is the capability that distinguishes a *self-revising* discovery system from a static-schema agent, and the spec treats it as **deferred but seamed**: v1 need not implement true migration, but it must not foreclose it.

The shape of the eventual capability:

- A new schema is registered as a new `schema_versions` row (§4).
- **"Transport old evidence into the new schema"** is a re-typing/migration step that runs as recorded operations over existing artifacts — migration is itself provenance-bearing work, not an off-the-books rewrite.
- A genuinely **isolated new type starts empty** — the categorical obstruction (the Kan obstruction in the source framework). The migration tooling must handle this gracefully: empty is valid; nothing auto-populates the new type; new evidence must arrive through normal proposal and gating. A migration that silently invents members of an isolated type is a bug.

**v1 scope.** v1 may support **append-only type registration** — adding new types and operation signatures in a new schema version — and defer true migration of existing artifacts. That is acceptable to start. The requirement on v1 is to leave a **clean seam**: the schema is already versioned, operations already carry provenance, and migration is already expressible as operations, so the deferred capability slots in without reworking the kernel.

**[SC]** Deferring this is realistic, not lazy: the reference substrate (§3.7) pins its schema version at a constant and never migrates, and the source paper lists "learning the base schema category" and "multicategorical discovery" as open problems. Regime transition is the genuinely distinguishing *self-revising* capability and the hardest piece — which is exactly why v1 seams it rather than faking it.

**[PILAR]** When migration does land, there is a working pattern to copy: a proven soft-gate system performed a real schema change (an opaque sequential identifier replaced by a property-derived one) by recording a migration plan, bumping a breaking version, and updating its validator — and it even records *implementation-driven spec amendments* in a decisions log (see [`references/pilar-comparison.md`](references/pilar-comparison.md)). A recorded, versioned, validator-backed migration is the concrete shape of this section's "recorded schema transition."

## 16. Non-Goals and Out of Scope

- **A universal cross-domain ontology.** The kernel shares only the meta-schema. Building one ontology that all domains map into is an explicit non-goal (Principle 8); it recreates the semantic-web tax.
- **A literal category-theory runtime.** The categorical framework is the specification lens (see preamble). No functor, colimit, or Kan extension is a runtime object. Naming a requirement categorically does not license implementing it categorically.
- **A multi-agent discourse layer.** Multiple agents debating and negotiating artifacts is a future seam, not v1. The store and commit path are designed not to preclude it — status, provenance, and decisions are all per-artifact and attributable — but it is out of scope here.
- **Parallel agents against one task.** v1 runs a single sandbox against a single task. The contracts already permit *N* sandboxes against one control plane — *diversity* (different approaches to the same task) or *race* (the same approach, best-wins) — because status, provenance, and decisions are per-artifact and attributable (§3.3). Designed-for, not built now.
- **Multi-tenancy and authentication.** v1 is single-user. The control-plane API is async and keyed by task identity from the first commit, so tenant scoping and auth slot in without reshaping the surface (§3.4, §3.9). Designed-for, not built now.
- **The four-service split (workspace as its own service).** v1 fuses the agent runtime and the workspace into one sandbox. Promoting the workspace to a fourth service is a clean internal seam (§3.3), deferred until independent scaling or a workspace image shared across loop frameworks requires it.
- **A pressure-scoring planner.** A planner that ranks goals by some notion of discovery pressure is a future seam. v1 uses the default retrieval/planner policy (§8.4).
- **True schema migration in v1.** Deferred and seamed (§15), not built initially.

These are named as seams, not dismissals: the architecture is expected to accommodate each without rework when its time comes.

## 17. How a Coding Agent Should Consume This Spec

Build in this order; each step is testable before the next, which is the point of the ordering.

1. **Kernel contracts first (§4–§7).** Implement the Store interface and the data model behind it; the audit-contract invariants as property tests; the lifecycle state machine as an explicit, testable transition table; and the commit path. Stop and validate the invariants and the state machine in isolation before adding any domain.
2. **Extension-point interfaces (§8).** Implement the schema, tool, and gate registries and the retrieval/planner policy as typed seams, with a trivial fake domain to prove the contracts — including a type with *no* gate, to prove "no implicit accept" rejects it.
3. **Context assembly (§9).** Implement the stable-prefix/volatile-tail split, regenerate-per-turn, and flush-on-commit, and write the bounded-context test.
4. **Service scaffolding (§3.3–§3.6).** Split the now-proven kernel into the three services and wire their APIs against the fake domain. Prove the seams before any real domain: the shape-error correction loop (§7 step 0), the advisory-verifier dispatch with a declared store-slice (§3.6), the `refine → revised → revises` round-trip (§6–§7), and the privileged-mutator boundary (the sandbox can neither write the store nor reach the verifier). Each service builds into its own container (§3.9).
5. **The feature-engineering domain (§12).** Wire the real schema, the workspace tools and object harvest, the harvested-`Feature` grounding gate, and the objective `Submission` gate (run the submitted code, score the trained model on the reserved verification dataset, seed-fixed, trial-count deflation). This is the first real end-to-end run.
6. **The v1 acceptance criteria (§13).** Demonstrate all twelve as runnable checks. v1 is not done until they pass.
7. **Only then** the soft-gate consulting domain (§14) and, when its time comes, true schema migration (§15).

Where this spec leaves a default rather than a mandate (the storage backend, the default retrieval policy, the weakest default gate), the default is a starting point; raise any reversal back to the human reviewer before locking it in.

## 18. Glossary

- **Artifact.** A typed payload with a lifecycle status — the durable noun of the system. (§4, §6)
- **Operation.** A typed, recorded edge from parent artifacts to an output artifact; the unit of provenance. (§4, §8.2)
- **Decision.** A recorded gate verdict on an artifact — gate, verdict, rationale, optional score. (§4, §7)
- **Gate.** A function `(artifact, store-context) → {verdict, rationale, score?}` that decides whether an artifact is worth keeping; the per-domain moat. (§8.3, §11)
- **Commit path.** The kernel's single privileged route from a proposal to durable state, where all invariants are enforced. (§7)
- **Status / lifecycle.** `proposed → {tentative | rejected | revised}`; `tentative → {accepted | rejected | revised | superseded}`; `accepted → superseded`. (§6)
- **`tentative`.** The status of an artifact that cleared the cheap gates (true + relevant) but not the hard ones (worth keeping). (§6, §14)
- **Supersede.** To replace an accepted (or tentative) artifact with a better one, atomically, preserving lineage via `superseded_by`. (§5.2, §7)
- **Provenance.** The transitive operation-and-artifact lineage behind an artifact; the basis for "why do we believe X." (§4.2)
- **No implicit accept.** The rule that committing a type with no declared gate is an error. (§2, §5.7)
- **Builder/Breaker.** The enforced separation of the agent that proposes an artifact from the gate that judges it; the Breaker is an adversarial judge. (§10, §11)
- **Reliability ladder.** The ordered descent from a numeric scoring functional down to community adjudication, used when no objective gate exists. (§11)
- **Reframe.** In the consulting domain, a shift in the client's prior frame — the highest-value output and a regime transition. (§14)
- **Regime transition / schema migration.** The revision of the schema itself — new types/operations/verifiers — and the provenance-bearing migration of evidence into it. (§15)
- **Meta-schema.** The only thing the kernel knows about types: that they have names and operations have typed signatures. (§3.2, §8.1)
- **Bounded-context guarantee.** The requirement that assembled context size not grow with store size. (§9)
- **Control plane.** The sole-mutator service: owns the store, commit orchestration, lifecycle, context assembly, the registries, task config, and the orchestration policy; deliberately unintelligent. (§3.4)
- **Sandbox.** The ephemeral service where the agent loop runs and tools execute; proposes but never writes, rebuilt (workspace + chat) each cycle. (§3.5)
- **Workspace contract.** The invariant directory skeleton (fixed roles/paths: read-only `data`/`context`/`tools`/`spec`, writable-ephemeral `scratch`/`outbox`) the control plane provisions into every sandbox; structure invariant, contents per-task. (§3.4)
- **Outbox.** The writable workspace region where the agent writes its proposal and object attachments; the control plane's deterministic harvest source. (§3.4, §3.5)
- **Composed system prompt.** The agent's system prompt, mechanically assembled by the control plane from three layers — kernel orientation (invariant), domain instructions, and per-task instructions; the orientation + layout layers are stable and live in the prompt-cache prefix. (§3.4, §9)
- **Verifier service.** The advisory service hosting the gate primitives; takes a shaped proposal plus a declared store-slice and returns a verdict; never writes. (§3.6)
- **Shape-error.** A correctable, unrecorded response to a malformed proposal — wrong composition or types, not wrong content; checked before any gate. (§3.4, §7)
- **Refine.** The third gate verdict — *recoverable*: the artifact is mostly sound but carries a localized, named defect, fixed by a tracked revision. (§7, §11)
- **`revised`.** The status of an artifact a `refine` verdict sent back for revision, pointing to its revision via `revised_by`. (§6)
- **Orchestration policy.** The control plane's declarative run-again / stop rule (and the refine cycle cap); loop control lives here, not in the verifier. (§3.4)
- **Object/blob payload.** An artifact payload held by reference (`blob_ref` + `content_hash`) so code or a binary artifact fits the store, with the bytes in the object store. (§4.1, §4.2)
- **Object attachment / harvest.** An object a proposal carries (e.g., the v1 submission's code), living in the sandbox workspace; the control plane *harvests* it at intake — before the workspace is regenerated — content-addresses it, and hands it to the verifier. (§3.4–§3.5)
- **Submission.** In the feature-engineering domain, the gated artifact for a cycle: the agent's code (clean → engineer → train) plus a JSON description of its features; the model is trained by the gate, not submitted. (§12)
- **Harvested `Feature`.** A typed artifact the harness derives from a submission's description (one per declared feature), linked by a `harvest` operation; carries a cheap grounding gate. (§12)
