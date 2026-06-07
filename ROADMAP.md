# Verity Roadmap

The living path from an empty repo to **MVP**. This is the canonical answer to *"where are we, and
what's next?"* — `README.md` and `CLAUDE.md` point here.

**MVP = spec v1.** The feature-engineering domain (spec §12) runs end-to-end and **all twelve §13
acceptance criteria pass**. That is the finish line for this roadmap (Phase 4).

The shape of the path: build the **control plane first** — the hard part, the sole mutator that owns
every invariant — and prove it works against *bespoke, throwaway* agent/workspace and verifier
stand-ins. The control plane is deliberately ignorant of how the agent and verifier work internally,
so simple stubs are enough to validate it. Each later phase then **replaces a stub with the real
service**.

---

## How to use this roadmap (hygiene)

- **The roadmap reflects reality, not intentions.** When a PR changes the plan's reality — a phase
  starts, a milestone lands, scope shifts — it **updates this file in the same PR**. A roadmap that
  has drifted from the code is worse than none.
- **Status legend:** ⬜ not started · 🚧 in progress · ✅ done.
- **One source of truth.** If you want to know what to build next, read the lowest ⬜/🚧 item here.

## Tracking off-roadmap items (use GitHub issues)

The roadmap tracks the **planned arc**. Everything else — anything that surfaces mid-stream but
doesn't need immediate resolution and **must not be lost** — goes into a **GitHub issue**, never a
buried `TODO` in the code or a line in someone's head. Deferred decisions, known limitations, a spec
ambiguity, a future refactor, a "we should revisit this": **open an issue.**

- Suggested labels: `deferred`, `tech-debt`, `spec-question`, `enhancement`.
- Link issues from the phase they bear on, so the arc and the backlog stay connected.
- The "Explicitly deferred past MVP" list below should each become an issue/epic when it comes alive.

## Cross-cutting principles (always on)

These hold in every phase (see `CLAUDE.md` → *Coding habits*):

- **uv** for everything; **structured logging from day one**; **tests alongside the code**.
- **Branch + PR** for all changes; **never a PR on buggy or embarrassing code.**
- **Container-per-service readiness** — control plane, sandbox, and verifier each build into their
  own image; nothing defeats that portability.
- **Spec-traceability** — implementation decisions cite the spec section (`§N`) they realize; the
  vendored `spec/` tree is authoritative.
- **Invariants are property tests**, not prose aspirations (spec §5).

---

## Phases

### Phase 0 — Bootstrap ✅

Working day-one infrastructure so everything after this is built on solid ground.

- uv project, `src/` package layout, structured logging, green test harness, lint/format/type config,
  a `justfile` for the common commands.
- **Exit:** lint + type + test green on the skeleton; minimal CI runs them on every push.

*(Delivered by the initial scaffold; CI added in `feat/kernel-contracts` — a small GitHub Actions
workflow calling `uv run` directly, no `just` in CI. CI is intentionally not a focus this early.)*

### Phase 1 — Control plane + integration doubles 🚧

A configurable control plane, proven end-to-end against bespoke stand-ins for the agent/workspace and
the verifier (test tooling in `tools/harness/`, **not** product).

- **1.1 Kernel contracts ✅** — Store interface + data model (artifacts / operations / decisions /
  schema_versions + object store), audit-contract invariants as property tests, the lifecycle state
  machine, the commit path. Landed in `feat/kernel-contracts`: SQLite reference backend behind a
  `Store`/`CommitSink` split (privileged writes unreachable from the read+propose surface); the §6
  transition table; §5 invariants as Hypothesis property tests; the §7 commit protocol (shape-error,
  no-implicit-accept, proposer≠gate, cheap→tentative→accepted, reject, refine, supersede). *(§4–§7)*
- **1.2 Extension-point interfaces ✅** — schema / gate registries (+ bindings) and the
  retrieval/planner policy; the invariant workspace contract + composed 3-layer prompt; per-task
  config; a trivial fake domain including a **gateless type** to prove "no implicit accept." Landed:
  `registries.py` (typed schema + gate registries; the gate registry *is* the commit path's
  binding resolver, so a gateless type resolves to `None` → `NoImplicitAccept`). **The §8.2 tool
  registry was intentionally collapsed** — a tool's harness-agnostic content is just its typed
  signature (= an `OperationSignature` in the schema registry), and its executable form is
  harness-specific, so it belongs to the Phase-3 sandbox adapter (parallel to `WorkspaceLayout`),
  not the control plane. Divergence from §8.2 to sync back to the spec. Also: `ports.py`
  (`SandboxPort`/`VerifierPort` behind a config-keyed `ProviderRegistry`, async + no-op lifecycle
  hooks — the multi-harness/multi-verifier seam; the verifier request carries no rationale field by
  construction); `workspace.py` (invariant `WorkspaceContract` + pluggable `WorkspaceLayout` + a
  default spec-role layout + outbox harvest); `config.py` (`TaskConfig` + deterministic 3-layer
  prompt); `domains/fake.py`. *(§8, §3.4)*
- **1.3 Context assembly ✅** — stable-prefix / volatile-tail split, regenerate-per-turn, the
  manifest, and the bounded-context test. Landed in `context.py`: a `ContextAssembler` that
  regenerates the two-part context from the store each turn (stable prefix = system prompt +
  capped, payload-free, ranked manifest; volatile tail = retrieved artifacts + goal + scratch).
  The **bounded-context guarantee (§13.5)** is structural — manifest/tail caps + per-item
  truncation — and proven by a test: 40× the artifacts assembles to the same size. *(§9)*
- **1.4 Service boundary + async API ✅** — configure-by-task; proposal intake + shape validation +
  object harvest; verifier dispatch; cycle control; extraction. Landed in `api.py`: an async
  `ControlPlane` (the sole mutator) that stamps the schema version + resolves/provisions services
  on configure; at intake validates shape first (malformed → records **nothing**), harvests outbox
  objects before teardown (content-addressed), segregates the agent rationale onto a side channel,
  then proposes and runs the §7 commit path; dispatches to the async verifier over the
  rationale-free declared slice (incumbents + rejected-log) by running the sync commit in a worker
  thread; an `OrchestrationPolicy` (max-cycles, refine cap, stop-on-accept) drives the
  serve→collect→commit→regenerate loop; extraction reads accepted artifacts, provenance, and the
  rejected/superseded/revised logs. *(§3.4)*
- **1.5 Integration doubles** (`tools/harness/`, non-product) — a **stub agent/workspace** (reads the
  served context, emits scripted proposals, writes object attachments to the outbox, handles
  shape-error and refine feedback) and a **stub verifier** (returns scripted verdicts, consumes the
  object attachments it's handed).
- **Exit:** the control-plane subset of the §13 criteria demonstrated via the doubles —
  propose→gate→commit; shape-error returns correctable with **no** decision row; refine →
  `revised`/`revised_by` → `revises` with intact lineage; object **harvest-before-teardown**;
  no-implicit-accept refusal; the privileged-mutator boundary holds; bounded context across a large
  store; "why do we believe X" answerable from provenance.

### Phase 2 — Verifier service + gate-primitive SDK ⬜

Replace the stub verifier with the real, advisory verifier.

- Queue-fronted async API; an SDK of gate primitives (auto-code-runner, model-tester, llm-judge,
  agentic-grader, human-in-the-loop); the independence/network boundary; declared store-slice
  handling. *(§3.6, §8.3, §11)*
- **Exit:** the real verifier renders reproducible verdicts on (still-simple) artifacts; the stub
  verifier is retired from the happy path.

### Phase 3 — Sandbox service (agent runtime + workspace) ⬜

Replace the stub agent with the real sandbox.

- Open-source agent loop + LLM client; ephemeral workspace provisioned to the workspace contract;
  the outbox; per-cycle workspace regeneration + chat flush; read-only data/context mounts; never
  contacts the verifier. *(§3.5, §10)*
- **Harness-bound tools.** This is where the executable form of a tool lands — the sandbox adapter
  binds each domain operation (typed in the schema registry, §8.1) to its framework's native tool
  model (the control plane carries no tool registry; see Phase 1.2). Parallel to `WorkspaceLayout`.
- **Exit:** a real agent drives read→propose→gate→commit against the control plane; ephemerality and
  gold-data isolation hold across cycles.

### Phase 4 — Feature-engineering domain (§12) → MVP ⬜

The first real discovery run, and the MVP.

- Real schema (`DatasetVersion` / `Submission` / harvested `Feature`); workspace tools + code-object
  harvest; the objective gate (the verifier runs the submitted code and scores the trained model on
  the **reserved verification dataset**, net of complexity, trial-count-deflated); the grounding gate
  on harvested features; `refine` on a bad feature. *(§12)*
- **Exit:** **all twelve §13 acceptance criteria pass → MVP reached.**

---

## Explicitly deferred past MVP

Designed-for, not built before MVP (spec §16 seams). Each becomes a tracked issue/epic when its time
comes:

- Multi-tenancy and authentication on the control plane.
- Parallel agents against one task (diversity / race — N sandboxes → 1 control plane).
- The four-service split (workspace promoted out of the sandbox).
- True schema migration / regime transition (spec §15).
- The soft-gate consulting domain (spec §14).
