# Verity Roadmap

The living path from an empty repo to **MVP**. This is the canonical answer to *"where are we, and
what's next?"* — `README.md` and `CLAUDE.md` point here.

**MVP = spec v1.** The feature-engineering domain (spec §12) runs end-to-end and **all twelve §13
acceptance criteria pass**. That is the finish line for this roadmap (Phase 4).

> **Status: MVP reached ✅** — Phases 0–4 are done. All twelve §13 criteria pass as runnable checks
> (`tests/test_feature_engineering_acceptance.py`); a `@live` run drives real Claude proposals scored
> in a container on the stellar dataset. What remains is the post-MVP backlog below.

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

### Phase 1 — Control plane + integration doubles ✅

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
- **1.5 Integration doubles ✅** (`tools/harness/`, non-product) — a **stub agent/workspace** (reads
  the served context, emits scripted proposals, writes object attachments to a real
  `DefaultLayout` outbox, handles shape-error and refine feedback, regenerates to discard the
  writable workspace) and a **stub verifier** (returns scripted verdicts, consumes the object
  attachments it's handed). Both implement their ports and hold no path to the store or each other.
- **Exit ✅:** the control-plane subset of the §13 criteria demonstrated via the doubles
  (`tests/test_exit_criteria.py`) — propose→gate→commit; shape-error returns correctable with **no**
  decision row / trial-count entry; refine → `revised`/`revised_by` → `revises` with intact
  lineage; object **harvest-before-teardown** round-trip; no-implicit-accept refusal; the
  privileged-mutator boundary holds; bounded context across a large store; "why do we believe X"
  answerable from provenance.

### Phase 2 — Verifier service + gate-primitive SDK ✅

Replace the stub verifier with the real, advisory verifier (spec §3.6): an **SDK of composable gate
primitives** behind the existing `VerifierPort` seam. The verifier holds the gate **plugins** (keyed
by gate name); the control plane keeps only the **binding** — pipeline position (`is_hard`) and the
declared-inputs allowlist — so §8.3's split (bindings in the control plane, plugins in the verifier)
is realized, not just designed. Built in sub-phases, each a tested, gate-green commit:

- **2.1 SDK primitives + the verifier service ✅** — a `GatePrimitive` contract and the rungs of the
  reliability ladder (§11) that need no container: **deterministic-check** (rung 2, free → earns
  `tentative`), **numeric-scorer** (rung 1, improve-score-net-of-cost), **llm-judge** (rung 4,
  *labeled-weak*, behind a `ModelClient` seam with a deterministic fake so the suite stays offline),
  plus *human-in-the-loop* (returns no verdict → rests `tentative`) and *model-tester* as seams. An
  `SdkVerifier` implementing `VerifierPort`, keyed by gate name, registered as a provider. *(§3.6,
  §8.3, §11)*
- **2.2 Container-isolated auto-code-runner ✅** — a `CodeRunner` seam with a real
  **`ContainerCodeRunner`** (`docker run` with a hostile-input posture: `--network=none`, read-only
  mounts, writable `tmpfs`, non-root, memory/cpu/pids limits, dropped caps, hard timeout; result read
  from a captured output file — a concrete data-plane in/out, feeds #3) and a deterministic
  `FakeCodeRunner` for the unit suite. The `auto-code-runner` primitive composes over it. One real
  integration test is marked `@pytest.mark.docker` and auto-skips when Docker is absent, so the
  minimal CI needs no Docker-in-CI. *(§3.6, §11, §12)*
- **2.3a Contract extraction ✅** — pulled the cross-service vocabulary into a new
  **`verity.contracts`** package (the value model + the service ports/provider registry), so no
  service depends on another's internals: the verifier now imports `verity.contracts` and has
  **zero** `control_plane` imports. `control_plane.store`/`commit` re-export the value types they
  operate on (a legitimate façade); `control_plane.ports` was a pure passthrough after the move and
  was **deleted**, its callers repointed to `verity.contracts`. Control-plane-internal types
  (`Decision`/`Provenance`/`Store`/`CommitSink`, the commit machinery, the registries) stay in
  `control_plane` — the cut is "what crosses a wire" vs "what the control plane persists". This is
  the structural enabler for the multi-verifier / multi-harness vision (a second verifier is an
  added adapter against a fixed contract); a wire serialization schema + a networked transport
  adapter remain later work. *(§3.3)*
- **2.3 Independence boundary, made checkable ✅** — each gate declares its store inputs
  (`declared_inputs: frozenset[StoreInput]` on `GateSpec`, default **empty** → the gate sees only
  the artifact under test); the new `control_plane/independence.py` assembles exactly those via
  per-input providers and asserts the resolved slice's sources ⊆ the declared allowlist *before*
  dispatch (`IndependenceViolation` otherwise), so an over-broad gate fails before it runs (§10).
  This complements the type-level no-rationale guarantee from `verity.contracts` (slice is
  `Artifact`-only): that closes the rationale channel, this closes the over-broad-context channel.
  The per-gate slice replaces the old one-size-fits-all `_build_slice`; the fake domain's selection
  gate declares `{INCUMBENTS, REJECTED_LOG}`. Property-tested (Hypothesis). *(§8.3, §10)*
- **2.4 Real artifacts through the loop ✅** — a minimal **code-execution domain**
  (`domains/code.py`: a `Submission` type gated by a cheap `parses` check (real `ast.parse`) then a
  hard `runs-clean` execution gate over the auto-code-runner) plus reusable *genuine* sample scripts
  (`tools/harness/sample_code.py`: `CLEAN` / `RAISES` / `SYNTAX_ERROR`). The stub agent emits real
  Python; the `SdkVerifier` parses and executes it, so verdicts are **earned**, not scripted.
  `test_code_loop.py` drives it end-to-end: offline, a syntax error is localized to a `refine`
  before any run and a clean submission commits through the fake runner; under Docker, a clean
  script *executes* in a container → accepted and a raising script → rejected — accept/refine/reject
  all on real evaluation.
- **2.5 Integration + exit ✅** — the deterministic fake domain gained real `SdkVerifier` plugins
  (`build_fake_verifier`, marker semantics as `deterministic_check`s), and `test_exit_criteria`
  (the control-plane §13 demonstrations) was migrated onto the **real verifier** — the stub verifier
  now survives only as a port-protocol double (`isinstance(StubVerifier(), VerifierPort)`).
  `test_reproducibility.py` pins the exit bar: identical inputs → identical verdicts across
  independent verifier instances, and the same code submission reaches the same committed outcome
  through two independent stacks.
- **Exit ✅:** the real verifier renders **reproducible** verdicts on (still-simple) artifacts; the
  stub verifier is retired from the happy path.
- **Correction — opaque verifier (ADR 0001).** After Phase 2 landed, the verifier seam was reworked
  to match the product vision: the verifier is an **opaque, user-selected package** that returns a
  **verdict bundle** (status + decisions) from one handoff — the control plane no longer sequences
  gates or parses the proposal. The gate pipeline + cheap/hard staging moved into the verifier; the
  control plane keeps coverage (no-implicit-accept), the per-type declared slice (§10), proposer ≠
  gate, and lifecycle validation. Also: the proposal-shape check is a presence-only pre-gate filter,
  and harvested objects are recorded as an artifact **sidecar** (payload untouched). This amends
  spec §3.3/§3.6/§7/§8.3/§10 and adds the `proposed → accepted` edge (§6); see
  [docs/adr/0001](docs/adr/0001-opaque-verifier-and-control-plane-boundary.md). Folds in #8.

*Decisions taken entering Phase 2:* container isolation is built **now** (not deferred) for the code
runner; the LLM client is `anthropic` behind a `ModelClient` seam (suite uses a deterministic fake);
the runner shells out to the `docker` CLI rather than taking a Python Docker SDK dependency. Hardening
beyond Phase 2's needs (rootless/gVisor/seccomp) and the networked standing-service data plane (#3)
are tracked as issues, not blockers.

### Phase 3 — Sandbox service (agent runtime + workspace) ✅

Replace the stub agent with the real sandbox. Framework: **Deep Agents** (provider-agnostic — Claude
now, cheap open models later; ADR 0002). Built as two sprints behind one `SandboxPort`.

- **Sprint 1 — in-process Deep Agents sandbox ✅**
  - `verity.sandbox`: a framework-neutral `AgentSandbox` (`SandboxPort`) over the `DefaultLayout`
    workspace + a `SandboxDriver` seam; the Deep Agents driver is the one module importing the
    framework. The agent is a **general-purpose coding agent in YOLO mode** (Deep Agents' native
    coding tools incl. shell `execute`, no permission prompts); steer comes via the control plane.
    *(§3.5)*
  - **Harness-bound tools.** The sandbox adapter binds each domain `OperationSignature` (§8.1) to a
    propose tool; the agent writes a **proposal descriptor** to the outbox and the trusted host
    harvests + mints the typed `Artifact`+`Operation` (so it can't forge ids/lineage). Parallel to
    `WorkspaceLayout`. **Additional sandbox configs are additive** (a new model/driver/framework is a
    registration line or one new driver — ADR 0002).
  - Offline tests (fake driver + a fake-model Deep Agents run) drive `read→propose→gate→commit` on the
    fake + code domains; a live Claude smoke commits an accepted `Note` end to end.
- **Sprint 2 — container isolation + cross-cycle exit ✅**
  - `DeepAgentsContainerDriver` runs the loop in a fresh per-cycle container (hostile-input posture:
    non-root, all caps dropped, `no-new-privileges`, read-only root + tmpfs, mem/CPU/PID limits,
    timeout-kill) — but with **outbound network** for the model API and the workspace mounted writable,
    with read-only roles re-mounted ro on top (**physical gold-data isolation**) and `data_sources`
    mounted ro into `data/`. The container entrypoint reuses the shared agent builders, so a host that
    only orchestrates containers needs no Deep Agents install (`Dockerfile.sandbox` carries it).
  - A docker+live integration test commits a `Submission` end to end: the agent **writes and runs
    code inside the container**, then proposes; the verifier gates it `ACCEPTED`. A cross-cycle test
    proves feedback-driven correction + ephemerality (a shape-error records nothing, threads back, and
    the next cycle commits on a freshly regenerated workspace).
- **Exit ✅:** a real agent drives read→propose→gate→commit against the control plane; ephemerality and
  gold-data isolation hold across cycles, in-process and under container isolation.

### Phase 4 — Feature-engineering domain (§12) → MVP ✅

The first real discovery run, and the MVP — the §12 domain on the real Kaggle stellar dataset
(`feature-engineering-test/`), run for multiple proposal rounds against the live control plane. Two
settled decisions shape it: the submitted **script trains end-to-end** and the verifier scores it on
a **reserved hold-out** split from `train.csv` (leakage caught on the reserved set, §13.11); and the
submission declares a **package list** the runner **pip-installs at run time** (network on; pinned
versions for reproducibility). Built in sub-phases, each a tested, gate-green PR:

- **4.1 Domain skeleton + data + object provisioning ✅** — `domains/feature_engineering.py`: the §12
  schema (`DatasetVersion` root / `Submission` gated `{INCUMBENTS, REJECTED_LOG}` / `Feature` gated)
  + `submit`/`revises`/`harvest` op signatures + the presence-only shape spec + domain instructions
  (the script I/O contract). A **durable-object provisioning policy** (`ObjectProvisioningPolicy`,
  modes `ALL`/`ALL_ACCEPTED`/`LAST_ACCEPTED`) on `TaskConfig`, resolved by the control plane in
  control-plane-native terms (status/recency, **not** verifier semantics) and materialized into a
  **writable** role via `ServedContext.workspace_objects` — **re-provisioned every cycle** (agent
  edits never persist; read-only is reserved for gold data). A deterministic stdlib stratified split
  (`tools/harness/dataset.py`). Offline tests + the per-mode policy + a two-cycle run that provisions
  the prior accepted script. *(§9, §12)*
- **4.2 The two `Submission` gates ✅** — `build_feature_engineering_verifier`: both gates over one
  cached run of the submitted script (train on agent data, predict the reserved hold-out). The cheap
  **runs-clean** gate (→ `tentative`) requires a clean exit + a well-formed prediction for every
  reserved row; the hard **selection** gate (→ `accepted`) scores **balanced accuracy** net of a
  per-feature complexity penalty, **deflated by the rejected-log** (the trial count), and must beat
  the incumbent (status from the slice, scores from the verifier's own measurement ledger — so
  independence holds). The `CodeRunner` gained a `requirements`/`network`/`env` path: pip-install the
  declared deps into an `exec` tmpfs over outbound network (the no-deps default stays hardened). A
  `@docker` test installs pandas and scores a real script end to end. *(§11, §12)*
- **4.3 Harvested `Feature`s + grounding + refine ✅** — a domain `harvester` hook on `TaskConfig`
  (`HarvestedChild`) + the control-plane harvest step: on acceptance it mints one `Feature` per
  declared feature via a `harvest` operation (parent → child), carrying the parent's code sidecar, and
  commits each through its own cheap **grounding** gate (genuinely defined by the code, else refused —
  no implicit accept for `Feature` too, §13.6). A static **features-defined** check refines a
  submission whose code omits a declared feature, naming it in the defects → a tracked revision with
  intact lineage (§13.8). Ids/lineage are minted on the trusted side; the domain only declares
  content. *(§12, §6, §7)*
- **4.4 Live multi-round run + the twelve §13 criteria ✅** — `test_feature_engineering_acceptance`
  runs **all twelve §13 criteria** as runnable checks on the feature-engineering domain through the
  real control plane, verifier, and sandbox (offline, deterministic runner). A `@live` multi-round
  demonstration (`test_feature_engineering_live`) drives real Claude proposals scored in a container
  on the actual stellar dataset, improving on the provisioned incumbent — auto-skips without a key /
  Docker / the dataset.
- **Exit ✅:** **all twelve §13 acceptance criteria pass → MVP reached.**

---

## Explicitly deferred past MVP

Designed-for, not built before MVP (spec §16 seams). Each becomes a tracked issue/epic when its time
comes:

- Multi-tenancy and authentication on the control plane.
- Parallel agents against one task (diversity / race — N sandboxes → 1 control plane).
- The four-service split (workspace promoted out of the sandbox).
- True schema migration / regime transition (spec §15).
- The soft-gate consulting domain (spec §14).
