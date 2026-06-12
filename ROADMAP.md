# Verity Roadmap

The living path from an empty repo to **MVP**. This is the canonical answer to *"where are we, and
what's next?"* — `README.md` and `CLAUDE.md` point here.

**MVP = spec v1.** The feature-engineering domain (spec §12) runs end-to-end and **all twelve §13
acceptance criteria pass** — the MVP finish line (Phase 4). The arc then continues into the
**Post-MVP roadmap** (Phases 5–7) below.

> **Status: post-MVP, deep into the Post-MVP roadmap.** Phases 0–4 reached MVP (all twelve §13 criteria
> pass as runnable checks in `tests/test_feature_engineering_acceptance.py`; a `@live` run drives real
> Claude proposals scored in a container on the stellar dataset). Beyond MVP: **Phase 5 ✅**
> (reliability & observability hardening); **Phase 6 🚧** (model breadth — code-complete, local open
> model live-validated 2026-06-08, hosted-OpenAI live pending a key); **Phase 7 🚧** (service split &
> full containerization — the seams are live, and the full-containerization **done-line is reached:
> a generic control plane runs in a container and launches ephemeral worker containers, running FE via
> its API — 7.5 ✅**). **Phase 8 🚧** (the long-lived, configurable control-plane service — a standing
> daemon configured + run via a CLI/HTTP, no image rebuild), settled by
> [ADR 0004](docs/adr/0004-long-lived-configurable-control-plane.md) and **activating issue #3**: the
> transport-agnostic **`ControlService` core (8.1 ✅)**, the **standing daemon + `verity` CLI over a
> Unix socket (8.2 ✅)**, the **container wiring + isolation proof (8.3 ✅)**, and the
> **external HTTP/REST control API (8.4 ✅)** have all landed — so **Phase 8 v1 is complete**: one
> image serves a local Unix-socket daemon *and* an authenticated (bearer-token) network API with an
> over-the-wire byte data plane, configured + run with no rebuild, durable across restart, with the
> exchange + store volumes provably off every worker. Remaining beyond v1 (deferred engine tracks):
> hosted-model live validation, the multi-tenancy/concurrency engine (#58), crash recovery (#57), the
> full #3 networked data plane, and a `K8sBackend`.
> The **control-plane API reference** (with a worked FE example) is
> [`docs/api-surface.md`](docs/api-surface.md).

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
- **Container-per-service readiness** — the architecture keeps each service independently imageable
  (async ports, no shared mutable state). Today the **sandbox** ships its own image; the control
  plane and verifier run in-process and are imaged when the service split lands (Post-MVP Phase 7,
  issues #3 / #10).
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
- **Enhancement — `fe-kaggle`: the real Kaggle leaderboard as the final-test gate ✅.** An additive
  task type (`domains/feature_engineering_kaggle.py` + `composition/fe_kaggle.py`, registered in the
  catalog) that keeps the FE domain but swaps the verifier for a **two-tier ladder**: a cheap
  local-hold-out proxy filters every cycle, then a hard gate regenerates the submission on the full
  train + the real `test.csv`, **submits to a live Kaggle competition** (`KaggleScorer` seam +
  `RealKaggleScorer`, the `kaggle` extra), and accepts only what beats our best **public-leaderboard**
  score. Trusted-submitter (creds never reach a worker); the ~5/day cap is read from the API and the
  gate blocks until budget frees; degrade-don't-crash on API failure. New `verity create
  --request-file` + `--test-data` (two inputs); the committed, runnable task package
  (`feature-engineering-test/{PROTOCOL.md,task.json,run.sh}`, local-agent default). Offline-tested on a
  `FakeKaggleScorer`; a `@kaggle @live` test submits for real (auto-skips without creds).

---

## Post-MVP roadmap

MVP (spec v1) is reached; this is the path beyond, **sequenced by dependency**. Reliability and
observability come first so the later phases are measurable and debuggable; the open-model payoff
follows; the full service split is last and largest. Each sub-item becomes a tracked GitHub issue as
it activates (roadmap hygiene).

### Phase 5 — Reliability & observability ✅

Harden the loop and make it measurable before scaling models or splitting services.

- **5.1 Error-handling & reliability hardening ✅** — the loop now degrades-don't-crash on *both*
  sides. A `GateUnavailable` (transient verifier/runner infrastructure, or a dispatch hung past a
  backstop) is a recorded, fed-back failed cycle, symmetric with the sandbox side (#21); the
  consecutive-failure breaker counts it too. Transient model + Docker calls retry with bounded
  backoff (a shared `retry_async`), and a missing/broken daemon is a typed error, not a raw crash.
  The commit path is **atomic** across its decision rows + terminal status. **#12** stamps the
  workspace-contract / orientation version into the store (with an orientation digest) like the
  schema version (§3.4). **#13** enforces JSON-object proposal payloads at intake (+ an optional
  per-operation required-keys schema). A per-cycle **step budget** (`StepBudgetMiddleware`) nudges
  then hard-stops a non-converging agent. "A failed step is recorded, not fatal" is a property test.
  Durable failure-provenance (recording failed cycles in the store, not just the in-memory
  `RunReport`) is split out as **#32**.
  - *Deferred — LLM-judge reproducibility (**#11**, §5.8).* The `llm_judge` primitive + `ModelClient`
    port exist and are unit-tested (against `FakeModelClient`), but the judge sits in **no live gate
    stack** and `AnthropicModelClient` is integration-only — so determinism / reproducibility
    discipline has nothing to bite on yet; its utility is theoretical. Build it out **when** we expand
    the verifier primitives *or* stand up an LLM-judge in a real domain's gate stack (a natural fit for
    Phase 6's cheap / local models). Until then the primitive stays frozen as a tested seam.
- **5.2 Agent-harness hardening ✅** — **soft-deadline signal** shipped: a `DeadlineMiddleware`
  pre-model-step hook injects a one-shot "≈Xs of Ys remaining — finalize and submit now" once past
  0.8 of the budget, so a long cycle yields a rushed-but-real proposal instead of being killed (the
  hard sandbox timeout stays the backstop; the control plane supplies the budget via
  `CycleInput.deadline_s`, the harness injects it). **Context compaction** confirmed already on —
  `create_deep_agent` auto-injects `SummarizationMiddleware`. Tool-call / step budgets and **#13**
  JSON-object payload enforcement were folded into 5.1 (both delivered there).
- **5.3 Metrics foundation — the store-derived `RunReport` ✅** — score-agnostic JSON projection of the
  store (the audit record): per-cycle outcomes (accept / reject / revise / refine / sandbox-fail),
  supersessions, trial counts, object-store growth, and cycle / gate / sandbox latencies (**5.3a**),
  plus **agent-loop telemetry** (tokens / model-steps / tool-calls / model) carried back from the
  sandbox through a reserved outbox file and summed per run (**5.3b**). All fields nullable — the
  consumer parses, the control plane does not assume a numeric score. Live emission + a dashboard come
  later (7.3).
- **5.4 Sandbox tools & extensibility ✅** — the `ToolBinding` seam (**#6**): a
  `SANDBOX_TOOL_REGISTRY` + `resolve_tools()` that binds non-propose, **executable** tools into the
  agent's harness by name (the control plane declares *what* a task gets via `sandbox_tools`; the
  adapter binds *how*; names — not callables — cross the container boundary). First concrete tool:
  **`read_pdf`**, so the agent can read PDFs in its read-only roles.

### Phase 6 — Model breadth: open, local & cheaper hosted 🚧 *(the thesis payoff)*

"Good proposals from cheap models" is the point of the whole approach — make it real across local /
open weights *and* cheaper hosted APIs. **Code landed and offline-green; live validation is the
remaining paired step** (stand up a local server / supply an OpenAI key — see below).

The seam is a typed, provider-agnostic **`ModelSpec`** (`provider`, `model`, `base_url`,
`api_key_env`, `extra`) + a lazy **`resolve_model`**: no `base_url` → the bare `provider:model` string
(Anthropic unchanged); a `base_url` → a `ChatOpenAI` pointed at any OpenAI-compatible endpoint. It
crosses the container boundary as env vars (Anthropic byte-identical to before), the driver forwards
the spec's key by name and adds `--add-host=host.docker.internal:host-gateway` only for host-local
endpoints, and both drivers resolve a spec at their own edge.

- **6.1 Local / self-hosted model ✅ (code + live-validated)** — `local_spec(...)` + a
  `deepagents-local` config over a **generic OpenAI-compatible endpoint** reached via
  `host.docker.internal`; **not** coupled to any one server (vLLM / vLLM-mlx, Ollama, llama.cpp
  interchangeable — see `docs/local-models.md`). Leans on 5.2 compaction for small local context
  windows. **Live-validated 2026-06-08** (Ollama, see below).
- **6.2 Cheaper hosted providers ✅ (code), live deferred** — `openai_spec(...)` + a `deepagents-openai`
  config behind the same seam; `OPENAI_API_KEY` forwarded automatically, no host gateway. The `@live`
  smoke is written and **auto-skips until a key exists** (no live OpenAI call was made — deferred per
  the no-key constraint). Other hosted providers swap in by config.
- **6.3 Eval / benchmarking harness ✅** — `src/verity/eval/` runs the **same task across model arms**
  and projects **quality + cost + latency** from the 5.3 `RunReport`: a nullable `cost_usd` + a small
  overridable pricing table (cost lives in the eval module, the control plane stays dollar-free), and a
  `Comparison` (outcomes, accepted-count, tokens, cost, best-score, latencies). `tools/benchmark_models.py`
  is the live cross-model entrypoint. The instrument that tells us whether the thesis holds.

*Live validation:* **(1) local — done (2026-06-08).** `qwen3.6:27b-coding-mxfp8` (Ollama on an M3
Ultra) drove the full read→propose→gate→commit loop to an accepted commit. Two benchmarks
(`tools/benchmark_models.py`, local vs frontier sonnet on identical data):
> - **Trivial code task:** parity — both accepted, local **$0.00** vs **$0.23**, ~1.5× latency.
> - **Scored FE domain (§12), 3 refine cycles:** the gap is real — frontier reached **0.910** balanced
>   accuracy (2 accepted), the local model landed **0 accepted** (all 3 cycles stuck at the
>   `features-defined` *refine* gate; scripts ran clean but never cleared the bar), at ~3× latency and
>   more tokens. So *good proposals from cheap models* held for easy work and **not** for this hard
>   task/model. *Observation (parked, not now):* the local arm got **stuck on one gate's refine loop**,
>   not failing randomly — likely the **prompt + the `refine`/`revise` feedback being too terse for a
>   limited-reasoning model** to act on, rather than a hard ceiling. A prompt/feedback-tuning question,
>   not a blocker.

**(2) hosted OpenAI — pending a key.** Both paths are manual (Docker + image + a real model), outside
the CI gate.

> **Revisit for hardening — Docker Model Runner (DMR).** We first tried DMR's `vllm-metal` backend as
> the *cleaner, Docker-managed* way to host the local model, but it **failed `EngineCore`
> initialization on Qwen3.6** (architecture too new for that backend build) and we fell back to
> Ollama. The seam already supports DMR (any `*.docker.internal` gateway). Worth revisiting once DMR's
> `vllm-metal` tracks newer model architectures — it is the tidier host-model story on macOS. Tracked
> as **#40** under the woven model-server hardening track.

### Phase 7 — Service split & full containerization 🚧

The biggest, most deferrable: make each service independently deployable. The **seams are done and
live-validated** (7.1–7.3 + the multi-tenancy seams): the verifier runs as a standing HTTP service the
control plane drives over the wire, with **no `ControlPlane` change** (the async ports paid off); the
transport seam is CI-provable via an in-process loopback and proven against the built image.

**The full-containerization architecture is now settled by
[ADR 0003](docs/adr/0003-control-plane-and-ephemeral-worker-provisioning.md):** a long-lived control
plane that runs sandbox/verifier work as **ephemeral, per-cycle, isolated workers** behind a
backend-agnostic provisioning seam (local Docker now, k8s later). Two decisions matter for the
roadmap: the per-cycle path is **batch-job workers** (`launch(input) → wait → harvest → destroy`), not
standing servers — which **supersedes the deferred "sandbox-as-server"** (we will not build it); and
the **control plane stays agnostic** — provisioning lives in a neutral `provisioning/` package behind
the ports, never on the CP's surface. **7.4 is the live edge** and ends at the full-containerization
done-line: the FE run driven entirely by control-plane configuration, no ad-hoc wiring.

- **7.1 Service entrypoints + images ✅ (verifier)** — `verifier/__main__.py` serves the
  advisory verifier over HTTP (uvicorn); `Dockerfile.verifier` builds it (core + `service` extra, no
  deepagents). The built image serves `/health` + `/provision` live. *Re-homed:* the **control-plane**
  entrypoint/image is absorbed into **7.4** (the worker-driven CP); the outward data-plane API (**#3**)
  is the Multi-tenancy & run-control track. *Superseded by ADR 0003:* the sandbox-as-server (the
  per-cycle path is batch-job workers — 7.4).
- **7.2 Networked transport ✅ (verifier path)** — a JSON **wire codec** for every boundary type
  (`contracts/wire.py`, object bytes inline-base64, rationale segregation structural) + a **transport
  seam** (`transport/`): `Transport` + an error-classifying envelope (boundary errors cross as
  themselves; `GateUnavailable`/`SandboxError` recoverable), `Remote*` clients / `*Server` hosts, a
  **loopback** transport (CI) and an **HTTP/FastAPI** transport. The **networked verifier (#10)** is
  live; degrade-don't-crash survives the hop (a killed verifier mid-run → recorded gate failure, run
  continues). *Reused:* the transport + `wire.py` are the seam for a **future launched-verifier-worker**
  (ADR g — not the FE batch path). *Re-homed:* the standing control-plane data plane (**#3**) →
  Multi-tenancy & run-control track. *Deferred (designed-for):* the **four-service split** (workspace
  out of the sandbox, §16).
- **7.3 Live telemetry + panel ✅ (emission)** — `telemetry.py`: a `MetricsSink` + `export_run_report`
  (outcome / gate-decision counters, per-cycle latency histograms, run-total gauges) over the 5.3
  RunReport, with a `Null` default (opt-in) and a lazy **Prometheus** sink (`telemetry` extra) + a
  structlog→metrics processor. *Deferred (ADR k — the telemetry push sink):* the Grafana dashboard +
  OpenTelemetry sink + wiring a scrape; under the worker model, workers ship labelled structured logs
  (`{tenant, run, cycle, role}`) and a fleet sink aggregates them.
- **Multi-tenancy & run-control seams ✅** — `JobQueue` port + in-process default, a `RunRecord` store
  keyed by `(tenant_id, run_id)` over the RunReport, and `tenant_id` threaded through `TaskConfig` /
  the report (single default tenant). Seams-first; the engine (real queue + workers, store upgrade,
  tenant isolation, **#27**/**#9**) is the deferred adapter swap.

- **7.4 Worker provisioning → full containerization, FE via config (ADR 0003) ✅** — the live edge.
  Realize ADR 0003: sandbox/verifier per-cycle work runs as **ephemeral, isolated, batch-job workers**
  behind a neutral `WorkerBackend` seam, ending at the **done-line: the FE run driven entirely by
  control-plane configuration, no ad-hoc wiring** (today it is hand-wired in
  `tools/benchmark_models.py:_build_fe_task`). Built fresh off `develop`. **Genericity invariant (held
  throughout):** the control plane stays agnostic — it talks only to `SandboxPort`/`VerifierPort` via
  the registry; provisioning lives in a **neutral `provisioning/` package** the CP never imports, and
  is **registration/deployment config, never `TaskConfig`**; FE-specifics are confined to a
  domain-layer registration builder. Each sprint a green, focused commit (suite green at every step):
  - **a. Provisioning contract ✅** — `provisioning/backend.py`: the `WorkerBackend` Protocol
    (`launch/status/wait/logs/stop/destroy/list/reap` + optional `recover` + `run_to_completion` = the
    batch path) + `WorkerSpec` (labels `{harness,tenant,job,run,cycle,role,config}`, image, command,
    env, mounts, `input_files`, limits, `runtime="runc"`, network, timeout). Imports nothing
    service/domain. Unit-tested; unconsumed.
  - **b. `DockerBackend` ✅** (riskiest — security argv) — `provisioning/docker.py`: ONE parameterized
    hostile-posture `docker run` builder driven by `WorkerSpec`, replacing the two duplicated builders;
    `--label` per key; `list`/`reap` by label; structured launch/destroy audit logs; subprocess (no
    docker SDK); `runtime` knob default `runc` (gVisor `runsc` opt-in, ADR i). **Byte-level argv-pin
    tests for both postures** (sandbox network-on / code-runner `--network=none`) before deleting the
    old builders; a `@docker` smoke.
  - **c. Backend-backed sandbox driver ✅** — `BackendSandboxDriver` (in `sandbox/`) over the existing
    `CycleInput` + `container_entry` (reused unchanged); `DeepAgentsContainerDriver` becomes a thin
    wrapper.
  - **d. Backend-backed code-runner ✅** (the untrusted-code-isolation resolution) — `BackendCodeRunner`
    (in `verifier/`): untrusted submitted code runs in a backend-launched, **labelled, isolated
    worker**, never an in-process subprocess. The FE verifier's gate **logic stays trusted + held
    across the run** (its in-memory incumbent ledger forbids per-cycle disposal); `reserved_labels`
    (the answer key) never enters any worker.
  - *(Test substrate: stubs-or-integration, no in-process middle.)* Offline tests use **`FakeBackend`**
    (a proper stub — scripts the worker boundary, runs nothing, real gate logic in-process);
    integration uses **`DockerBackend`** (`@docker`/`@live`). An "`InProcessBackend`" was considered and
    **dropped** — a half-real backend is neither a deterministic stub nor a faithful substrate, and it
    would drag sandbox/verifier knowledge into the neutral `provisioning/` layer. Backends are
    *substrates* (Docker now, k8s later); it is addable later behind the same port if Docker-free
    real-logic execution is ever needed.
  - **f. Declarative wiring at the registration layer ✅** (kills the ad-hoc wiring) — the
    `verity.composition` root: a `ProvisioningConfig` (backend + per-role substrate shape) + an
    FE composition builder `build_fe_control_plane` that replaces `_build_fe_task`. `TaskConfig` and
    `ControlPlane` unchanged; provisioning stays off `TaskConfig`. (The old
    `DeepAgentsContainerDriver`/`ContainerCodeRunner` classes remain for the `code` benchmark domain;
    full retirement is deferred to a cleanup pass.)
  - **g. FE-via-config acceptance ✅** (the done-line) — `tests/test_fe_via_config_acceptance.py`: FE
    built purely from declarative config + a backend selection; asserts an accepted `Submission`, the
    no-cross-cycle-bleed property (§3.5), untrusted code in a `{role:code-runner}`-labelled worker, and
    `reserved_labels` absent from every worker. Green on `FakeBackend` (offline — config-driven flow +
    real gate logic, scripted execution) and `DockerBackend` (`@live` for the real model; a recording
    backend wrapper runs the same isolation assertions on the live path).
  - **h. Placed seams ✅** — a neutral `RunContext` (`tenant_id`/`run_id`/`cycle`) the control plane
    binds to any service advertising `SupportsRunContext`; the backend-backed sandbox driver +
    code-runner stamp it onto every worker's `{harness,tenant,run,cycle,role,config}` labels (the
    verifier forwards the bind to its runner). A `RunRecord` is emitted at run end (keyed by
    `(tenant_id, run_id)`, pointing into the store). A label-reaper + the `verity-reaper` console
    entrypoint reap orphaned workers by selector (ADR e), out of band — the control plane never
    touches a backend.
  - *Deferred (placed seams, per ADR 0003 — see the tracks below):* the reconciliation-loop + per-tenant
    concurrency engine (ADR c/h) = the Multi-tenancy engine + "parallel agents against one task"; the
    `K8sBackend` (a second impl of the same port); the **launched-verifier-worker** for FE (precondition:
    move the incumbent ledger into the store + re-read incumbents inside the commit `transaction()`,
    ADR j); rootless + `docker-socket-proxy` posture and a Ryuk-style GC death-switch (deployment config,
    ADR f); gVisor/microVM (`WorkerSpec.runtime`, ADR i).

- **7.5 Fully containerized control plane — generic CP + FE via config (proven live) ✅** — the real
  done-line: a **task-agnostic** control plane runs *in a container* and launches the sandbox +
  code-runner **worker containers** as siblings on the host daemon (controlled `/var/run/docker.sock`,
  ADR 0003 §f — not docker-in-docker), running the §12 FE test by **configuring the CP via its API**,
  no ad-hoc wiring. Sub-parts:
  - **Decouple the CP from FE.** `build_fe_control_plane` (which built a `ControlPlane` *inside* an
    FE-specific function) is deleted. The CP is constructed generic; a task is applied through the CP's
    API (`register_sandbox`/`register_verifier` + `configure`) by `composition.fe.configure_fe_task` /
    `configure_code_task`. Enforced **mechanically**: an AST guard asserts `verity.control_plane`
    imports nothing from `verity.domains`, and one generic CP runs both the FE and `code` tasks.
  - **Sibling-mount fix.** `DockerBackend(staging_root=…)` / `VERITY_WORKER_STAGING` — staging dirs go
    under a host↔CP-container shared path so worker bind mounts resolve on the host daemon.
  - **CP image + entrypoint.** `Dockerfile.controlplane` (verity core + the docker CLI, no sandbox
    extra) + `python -m verity.composition.fe_run` (generic CP, FE applied via its API). The
    stratified-split helper moved into the package (`verity.composition.dataset`).
  - **Wiring + live proof.** `infra/compose.fe.yml` + `just fe-containerized`. Live on the local model:
    the CP container launched both worker roles and reached an **accepted Submission** (cycle 1
    shape-error → corrected → cycle 2 accepted, 5 Features grounded), with the 7.4.h worker labels and
    no stragglers.

### Phase 8 — The long-lived, configurable control-plane service (ADR 0004) ✅ *(v1)*

The batch control-plane container (7.5) becomes a **standing daemon** an operator (or Claude Code)
delegates to: configure a task, select sandbox + verifier configs from a **catalog**, provide data,
run, and pull results + durable artifacts — **without rebuilding any image**. "Wrap, don't rebuild":
an additive control/data plane over the *unchanged* kernel and the ADR 0003 worker model. **Activates
issue #3**; settled by [ADR 0004](docs/adr/0004-long-lived-configurable-control-plane.md). Each sprint
a green, focused commit.

**Guardrails (held throughout):** genericity (`verity.control_plane` imports no domain); the sandbox
**byte-provenance** isolation invariant (the answer key / wrong-role bytes never reach a worker);
**per-task store isolation** (a store per task instance — no same-typed `INCUMBENTS`/manifest bleed,
#58); verifier **opacity** (no task/verifier locking — self-description + graceful run-time failure);
and `composition/fe_run.py` + the library API stay valid.

- **8.1 `ControlService` core + task catalog + per-task durable stores ✅** — a transport-agnostic
  async facade over a **per-task `ControlPlane` + `SqliteStore` multiplexer** (a store per task
  instance — no cross-task bleed, no kernel change; it mirrors `fe_run`'s one-CP-per-run shape); a
  `TaskCatalog` + a JSON `TaskRequest`; refactor `configure_fe_task` / `configure_code_task`
  (`composition/`) into catalog entries consuming a `TaskRequest` (old signatures preserved);
  **persist the `TaskRequest`** so tasks are durable, rehydratable runnable entities; wire the
  `JobQueue` + daemon-level `RunRecord` seams (`contracts/jobqueue.py`, `control_plane/run_record.py`).
  **Carve-out (the riskiest new logic):** the split + answer-key (`reserved_labels`) derivation moves
  into the builder against client-supplied data — split-correctness tests + a positive
  **byte-provenance** test (and that two same-typed task instances never see each other's
  `INCUMBENTS`). Offline on `FakeBackend`. **Landed:** the new `verity.service` package
  (`ControlService` + `TaskIndex`) over `verity.composition`'s `TaskCatalog` / `TaskRequest`
  (`build_fe_task` / `build_code_task` consume a request; the old `configure_*_task` signatures are
  preserved); run identity is unified by injecting the daemon `RunRecordStore` + a per-run id cell
  into each per-task CP (kernel untouched, the AST genericity guard stays green). Tests:
  `test_dataset_split_carveout`, `test_fe_builder_byte_provenance` (answer key absent from every
  worker), `test_per_task_store_isolation` (no `INCUMBENTS` bleed), `test_control_service`
  (create → run → results + restart-then-run rehydration + catalog dispatch of the `code` type).
- **8.2 The daemon + internal IPC + the `verity` CLI ✅** — `verity serve` (the multiplexer +
  `DockerBackend`; a **background-task executor** + a global run lock; `status`/`results` served
  concurrently with an in-flight run) over a **Unix-domain socket**; the `verity` console-script
  **client** over that socket (`catalog` / `ingest` / `task create` / `run` / `status` / `results` /
  `export`); registry **self-description** (each sandbox/verifier publishes its contract) rendered by
  `verity catalog`; an incompatible task/verifier pairing **fails gracefully + informatively** via the
  existing degrade-don't-crash path (tested). **Landed:** background execution lives in the
  `ControlService` core (`submit_run`/`await_run` + an `asyncio.Lock`; `run()` stays a blocking
  wrapper, 8.1 tests unchanged); the control surface is **HTTP/FastAPI served over the Unix socket**
  (uvicorn `uds=` + an httpx `uds=` client — *no bespoke protocol*; the socket is the no-auth-yet v1
  trust boundary, and the *same* `build_app` is what 8.4 rebinds to a TCP port + auth). Self-description
  is `TaskTypeDescription` (built from the same domain schema that renders the system prompt) on the
  catalog (`describe`/`describe_all`). New `verity.service` modules: `http.py` (`build_app`),
  `daemon.py` (`verity serve`), `client.py`, `cli.py`; the `verity` console script. Tests:
  `test_control_service_concurrency` (background + concurrent reads + graceful abort-with-reason),
  `test_catalog_describe`, `test_service_http` (full ingest→create→run→results→export over ASGI +
  a real-`uds` round-trip; fastapi/httpx `importorskip`, so the lean gate skips cleanly).
- **8.3 Container wiring + live proof (the done-line) ✅** — the control-plane image now also serves
  the standing daemon, with the **exchange** + **persistent-store** volumes wired and proven isolated.
  **Landed:** `Dockerfile.controlplane` carries the `service` extra (fastapi/uvicorn/httpx) so it can
  run `verity serve`; a new **`infra/compose.daemon.yml`** brings up a long-lived `verity-cp` container
  (entrypoint `verity serve`) mounting the docker socket, the shared staging dir, the **exchange**
  (`VERITY_EXCHANGE`, client↔CP only) and a **persistent store** (`VERITY_STORE_ROOT`, durable
  stores + task definitions); `just cp-serve` / `cp` / `cp-down` drive its lifecycle. The
  **neither-reaches-a-worker** guarantee is `tests/test_daemon_volume_isolation.py` (offline, on a
  real on-disk `ControlService`): a `WorkerSpec` has no host-bind-mount field at all, and the test
  further asserts neither host path leaks (env / passthrough / command / input keys) and the answer
  key never appears. The **live proof** is `tests/test_daemon_live.py` (`@docker @live`, auto-skip):
  one running container lists **both `fe` and `code` task types**, runs an `fe` task
  (ingest→create→run→results→export to the exchange) and a `code` task with **no rebuild between
  them**, then **survives a restart** and re-runs the pre-restart task (durable definitions).
  *Deviation from ADR 0004 sprint 3:* per the chosen layout, `compose.fe.yml` is left **unchanged**
  (the one-shot `fe_run` batch path), so the image's default `ENTRYPOINT` stays `fe_run` and the
  **daemon compose overrides it** to `verity serve` — the daemon is fully reachable, just selected by
  `compose.daemon.yml` rather than baked as the image default.
- **8.4 External HTTP/REST control API ✅** — the *same* FastAPI `build_app` (8.2), now bindable to a
  **network TCP port** with **bearer-token auth**, completing Phase 8 v1. **Landed:** `verity serve
  --http HOST:PORT` (or `$VERITY_HTTP`; `python -m verity.service` mirrors `verifier/__main__.py`)
  serves the app over TCP and **refuses to start without `VERITY_API_TOKEN`**; the auth dependency
  gates every route but `GET /health` (401 otherwise) and stashes the principal on the request (the
  identity hook) — the UDS binding stays auth-free (the local trust boundary). Endpoint-parity with the
  CLI is unchanged; the `Client`/CLI gained a TCP target (`--url`/`--token`, `$VERITY_URL`). The
  minimal **over-the-wire byte data plane** was pulled forward so a client with no shared volume is
  usable: raw-bytes `POST /objects` (capped, 413 over) + `GET /artifacts/{run_id}/{path}`. Tests
  (`tests/test_service_http_external.py`, offline + a real-TCP round trip): `/health` open, 401
  without/with-wrong token, 200 with it, the refuse-without-token guard, and a byte upload → create →
  run → artifact-download round trip. **Resolves ADR 0004 open Q3 → bearer token** (mTLS/OAuth and the
  full #3 data plane — removing the exchange volume, streaming — stay deferred).
- *Deferred (placed seams, per ADR 0004):* crash recovery for in-flight runs (**#57**; Temporal a
  hardening candidate); the **multi-tenancy engine** — *cross-tenant* store isolation, the
  single-writer→Postgres store-engine upgrade, object GC, and parallel live runs (the residual beyond
  v1's per-task stores — **#58** / **#27** / **#9**); the **over-the-wire file API** (removes the
  shared exchange volume — #3 proper); the **plugin loader** for runtime domain-code registration; and
  **auth/authz** on the external HTTP surface.

### Woven through Phases 6–7

- **Code-runner hardening / generalization** — a declarative runner config (image, network policy,
  resource + output caps, mount layout, deps strategy), an offline / pinned-wheels install option to
  reclaim reproducibility, and a cleaner validity-rung vs scoring-rung split. Revisits the accepted
  network-on tradeoff.
- **Configuration guide + a Claude Agent Skill for setting up the system** — now that the configuration
  surface has stabilized through Phase 6, make standing up a new task / domain / model low-friction.
  (a) A **configuration guide** in `docs/` covering the whole surface: building a domain (schema +
  gate pipeline + shape validator + harvester), registering a sandbox via `ModelSpec` /
  `local_spec` / `openai_spec`, registering a verifier + code runner, assembling a `TaskConfig`, the
  `OrchestrationPolicy` (cycles / refine cap / stop-on-accept), and mounting data via
  `static_contents` vs `data_sources` (the §13.12 isolation note). (b) A **Claude Agent Skill** that
  scaffolds that configuration interactively — gathers the task/domain/model intent and emits a
  working registration + `TaskConfig` (and a benchmark arm), so a user (or Claude) can wire a new
  domain or model arm without reverse-engineering the test setups. The live-validation work
  (`docs/local-models.md`, `tools/benchmark_models.py`) is the seed material.

### Multi-tenancy & run-control (a track spanning Phases 5 → 7)

The control plane grows into a multi-tenant **job server**. The typed-provenance store stays the
system of record (artifacts / operations / decisions / objects) — this adds the *operational* layer
*around* it, not a second copy of it. Built **seams-first, engine-later**, so multi-tenancy is an
added adapter, not a reshape.

- **Run-control seams (with Phase 5).** A `JobQueue` port (`enqueue` / `claim` / `complete` /
  `status`) with a trivial **in-process default** (run synchronously, as today); a `RunRecord` /
  run-results store keyed by `(tenant_id, run_id)` that holds the **5.3 RunReport** + run status +
  *pointers* to the accepted artifacts (operational metadata referencing the provenance store, **not**
  a copy of it — the RunReport read-side and this store are the same thing); and a `tenant_id`
  threaded through the API / `TaskConfig` while there is still only one tenant.
- **Multi-tenancy engine (with Phase 7).** The real queue + worker model — commits stay **serialized
  per tenant** so the sole-mutator audit guarantee is preserved; the **store-engine upgrade** the
  queue forces (SQLite is single-writer → Postgres, or per-tenant DBs) — the natural moment for
  **object retention / GC** of bytes referenced only by terminal artifacts (**#9**, §5.5); **tenant
  isolation** (namespaced stores; no cross-tenant reads in retrieval / slices / object-provisioning) —
  the actual hard part; and the async **submit → job_id → poll/fetch** API a standing, networked
  control plane wants (the re-homed **#3** data plane). Naturally siblings with the service split
  (7.1–7.4). **This is the home for ADR 0003's deferred concurrency engine:** the level-triggered
  reconciliation loop + per-tenant concurrency slots/fairness (ADR c/h), and the commit-under-concurrency
  correctness (re-read incumbents inside the commit `transaction()`, ADR j). The `WorkerBackend` (7.4)
  is the seam it drives; a **`K8sBackend`** is the second impl of that port for a cluster deployment.
- **Phase 8 takes the first engine increment.** The long-lived control-plane service (ADR 0004) wires
  the `JobQueue`/`RunRecord` seams into a *standing daemon* (serial, single default tenant) and
  activates the #3 control half. It already isolates **per task** (a store per task instance, 8.1);
  the residual **cross-tenant** isolation (namespaced / Postgres stores, no cross-tenant reads) +
  concurrency stay this deferred engine (#58).
- **Not here:** authentication / authorization (still *further out*); parallel agents against one task
  (a related but separate concurrency concern, below).

### Open issues → roadmap home

Where each tracked issue folds in (so the backlog and the plan stay linked):

Currently-open issues (kept in sync with GitHub):

| Issue | Folds into |
| --- | --- |
| **#3** standing control-plane data plane | **Phase 8** (ADR 0004) + the multi-tenancy engine; the over-the-wire file API (removes the exchange volume) |
| **#5** spec sync: collapse §8.2 tool registry | *Further out* — spec housekeeping (already true in code) |
| **#9** object retention / GC | multi-tenancy engine (with the store-engine upgrade) |
| **#10** networked verifier transport | 7.2 — transport is live; residual: capability advertisement |
| **#11** LLM-judge reproducibility (§5.8) | 5.1 — when a judge enters a live gate stack |
| **#27** multi-tenancy epic | the Multi-tenancy & run-control track |
| **#32** durable failure-provenance (record failed cycles in the store) | 5.1 reliability (split-out) |
| **#40** revisit Docker Model Runner (vllm-metal) as the managed local-model host | Phase 6 (woven model-server hardening) |
| **#42** extensible stop/continuation predicate (accumulate-to-K / optimize-until-plateau) | acceptance-modes extensibility on `OrchestrationPolicy` |
| **#51** provision the in-flight *revised* submission on refine | object-provisioning (largely covered by `LAST_REVISED_OR_ACCEPTED`; confirm/close) |
| **#55** re-audit `docs/context-and-data-flow.md` to the current build | docs housekeeping |
| **#57** crash recovery for in-flight runs | Phase 8 deferred (durable run-status; Temporal at hardening) |
| **#58** store-isolation tiers | per-task in **Phase 8.1 ✅**; cross-tenant in the multi-tenancy engine |
| **#64** make the domain concept optional | architecture — a generic default domain + verifier-criteria in the prompt (§8 / §3.4) |
| **#65** per-task agent skills in the sandbox (harness-agnostic) | 5.4 sandbox extensibility (successor to the `read_pdf` tool seam) |
| **#66** extender docs for sandboxes & verifiers | the woven "configuration guide" item (docs slice) |

**Already done (closed):** **#2** (CI gates `main`/`develop`), **#8** (supersede-on-beat, 4.2),
**#6** (harness-bound executable tools — the `read_pdf` seam, 5.4), **#12** (workspace-contract/orientation
version stamp, 5.1), **#13** (JSON-object proposal payloads, 5.1), and the early correctness fixes
**#16/#17/#18/#20/#21** (naming, `refine_cap` enforcement, the dead manifest path, provisioned-object
naming + manifest, and sandbox-cycle-failure-skips-not-aborts).

### Further out / seamed (spec §16)

Designed-for, not yet scheduled; each becomes an issue/epic when its time comes:

- Authentication / authorization on the control plane (pairs with the multi-tenancy track above).
- Parallel agents against one task (diversity / race — N sandboxes → 1 control plane). The 7.4
  worker model + labels (`run`, `cycle`) are the placed seam (ADR 0003); building it = the concurrency
  engine in the Multi-tenancy track above.
- Runtime registration of new **domain code** — a plugin loader / `verity.task_types` entry-point group
  (ADR 0004 (i)); Phase 8 v1 requires an image build for a brand-new domain (only *config* is
  rebuild-free).
- Over-the-wire **file ingestion/egress API** — removes Phase 8's shared exchange volume (ADR 0004 (f)
  seam / the #3 networked data plane).
- Spec sync: collapse the §8.2 tool registry into operation signatures (**#5**) — already true in code.
- Secrets / egress policy once local + networked services land.
- True schema migration / regime transition (spec §15).
- The soft-gate consulting domain (spec §14).
