# Pre-Ablation Audit & Instrumentation Report

**Date:** 2026-06-29
**Scope:** Adversarial audit of the Verity codebase ahead of the `experiments/ablation/` runs,
with particular attention to the **agent sandbox** and the **verifier**. Covers three asks:

1. Bugs, logical flaws, and leaked / mis-placed hidden instructions.
2. Feasibility + recommended approach for dumping the agent's chat context before teardown (and/or
   more systematic step logging) for experimental instrumentation.
3. The exact outcome messages the verifier emits for every outcome.

**Method:** three read-only mapping passes, then six adversarial passes each tasked to *find a
failure* in one high-risk area (harvest/teardown ordering, answer-key isolation, rationale leakage,
hidden instructions, gate logic, code-runner sandboxing/determinism). Every load-bearing finding
below was re-confirmed by reading the cited source. No code was modified.

> **Status update (post-audit):** several findings have since shipped, so the "current state"
> descriptions and `file:line` citations below are a frozen snapshot that predates them. In
> particular: the **step transcript (§5) is now always-on, harvested, and content-addressed**
> (referenced as `transcript_ref`; telemetry also carries a `stop_reason` on a budget/recursion
> limit-end), the **agent-facing prompt copy was restyled to imperative** (so quotes like "your
> script" / "no leaderboard" no longer match), and the **step-budget safety net was recalibrated**
> (fixed model-step budget + a clamped recursion backstop). Treat this report as the historical audit,
> not the current spec; `tools/render_transcript.py` renders a harvested transcript.

---

## 1. Summary & verdict

**The system is sound enough to run the ablation — there are no blockers — but three classes of
issue can distort the ablation's headline metrics or quietly abort a sweep, and should be addressed
or explicitly controlled-for first.** The architecture's core invariants (answer-key isolation,
rationale segregation, harvest-before-regenerate, no-implicit-accept) hold *in the shipped code
paths*; the weaknesses are at the edges those invariants don't structurally cover.

The three things that most directly threaten the experiment's validity:

- **Scoring non-determinism (V2, V1, V6).** Submissions are scored with multithreaded training
  (`n_jobs=-1`) and **full network access during training** (`network=True`), and dependencies are
  resolved at install time. The *same* `submission.py` can score differently run-to-run — noise the
  ablation will misattribute to the experimental condition. This is the single biggest risk to
  clean deltas across the 10 seeds.
- **In-memory score ledger (G3).** "Best prior" lives in a per-process dict in the (now long-lived,
  sibling-container) verifier. A verifier restart mid-run collapses the baseline to `0.0`, silently
  accepting a regression and missing a supersession — breaking the monotone-best / zero-regression
  property that **Claim A** depends on.
- **Narrow exception net (S1–S3).** `run_cycle` catches only `SandboxError` and `GateUnavailable`.
  Several *predictable* failures (a disk-full `OSError` from the object store, a reachable-but-
  misbehaving remote verifier, a child-harvest error) raise neither, escape the cycle loop, and
  **abort the whole run with the in-flight cycle never recorded** — violating the repo's own
  "a failed step is recorded, not fatal" invariant and silently truncating a `RunReport`.

Plus one experiment-design confound to document: the **goal wording differs** between the bottom-rung
spec and the loop spec (X1).

Everything the brief was most worried about — answer-key leakage to the agent, the verifier seeing
the proposer's rationale — is **genuinely closed in the shipped paths** (see §2.4, §2.5). The
residual isolation risk is a *defense-in-depth* gap at the unguarded data-prep CLI, not a live leak.

---

## 2. Findings

Severity key: **blocker** (must fix before running) · **should-fix** (fix or explicitly control for)
· **nice-to-know** · **non-issue / verified safe**. "Bites the ablation?" flags whether the shipped
`fe-holdout` ablation path actually triggers it (vs. a latent kernel gap).

### 2.0 Master table

| ID | Area | Severity | Bites ablation? | One-line |
|----|------|----------|-----------------|----------|
| V2 | Code-runner | should-fix | **Yes** | Multithreaded training (`n_jobs=-1`) → non-reproducible scores = noise attributed to conditions |
| V1 | Code-runner | should-fix | **Yes** | `network=True` during *training*, not just pip → external/time-varying data → score drift (+ exfil vector) |
| G3 | Verifier/gate | should-fix | **Yes (on verifier restart)** | In-memory score ledger; restart → baseline collapses to 0 → silent regression accept, breaks Claim A |
| S1 | CP cycle | should-fix | **Yes** | Raw `OSError`/`StoreError` from `put_object`/`serve_context` escapes `run_cycle` → run aborts, cycle unrecorded |
| S2 | CP/transport | should-fix | **Yes (remote verifier)** | Non-`GateUnavailable` transport errors escape → abort; a *reachable-but-broken* verifier is fatal, an unreachable one degrades |
| X1 | Experiment cfg | should-fix | **Yes** | Goal wording differs: `spec.exp1.json` "Submit your best work" vs `spec.loop.json` "improve upon previous" |
| V3 | Code-runner | should-fix | **Yes** | Unbounded child stdout/stderr captured into verifier process → a runaway submission can OOM/hang the sweep |
| I1 | Isolation | should-fix | Latent (sloppy prep) | "Isolation by construction" is by *convention* at the unguarded CLI prep boundary; no CP/verifier backstop |
| S3 | CP cycle | should-fix | Conditional (harvester) | Child-harvest failure can rewrite an already-ACCEPTED parent cycle as `gate_error` / escape as `StoreError` |
| G1 | Verifier | should-fix | Latent | Empty hard stage / empty pipeline → **silent ACCEPT** with no hard gate ruling; `SdkVerifier` has no guard |
| G2 | Verifier | should-fix | Latent | `supersedes = verdict.supersedes or supersedes` is last-wins and runs in the cheap stage too |
| R1 | Verifier indep. | medium/latent | Latent | `payload` is a 2nd proposer-authored channel that *does* reach gates; no allowlist; `llm_judge` renders `payload!r` |
| V6 | Reproducibility | nice-to-know | Reproducibility | pip resolves transitive deps at install time → seeds run on different days get different numerics |
| F4 | Sandbox | nice-to-know | Rare | Hard wall-clock kill discards a complete proposal already bridged to the host outbox |
| V5 | Code-runner | nice-to-know | Rare | Orphan container if the verifier process dies mid-run (reaping is a fleet-loop concern, not wired here) |
| I2 | Isolation | nice-to-know | Latent (sloppy prep) | `test.csv` passed to the agent unstripped; a labelled `--test` would leak the target |
| G5 | Verifier | nice-to-know | No | Bundle-consistency check is one-directional (won't catch a fabricated empty-decision ACCEPT) |
| G7 | Verifier | nice-to-know | No | `_KaggleGates._run_cache` never cleared (grows per task); memory only |

**Verified safe (non-issues):** harvest-before-regenerate on every *caught* branch and
bridge-before-container-removal (§2.1); answer-key + credential isolation in the shipped code
(§2.4); the rationale *metadata* channel is airtight (§2.5); `accept_policy` is implemented
correctly (§2.3); deterministic, disjoint train/holdout split; resource caps (`--cap-drop=ALL`,
`--read-only`, `--pids-limit`, non-root, no-swap) are a solid hostile posture.

### 2.1 Sandbox (lifecycle, harvest, teardown)

The sandbox lifecycle is **disciplined**: the workspace is `rmtree`'d and re-provisioned every cycle
(`sandbox/core.py:118-127`), the agent only writes a descriptor + objects to the outbox, and durable
harvest precedes the destructive `regenerate()` on every path the control plane actually catches.
Confirmed safe:

- **Harvest-before-regenerate (F6 — safe).** `regenerate()` is called only by `run_cycle`, always
  *after* `submit_proposal` has `put_object`'d every declared object (`api.py:352`, before commit).
  On the `SandboxError` branch nothing was harvested (the driver raised first); on the
  `GateUnavailable` branch objects were already persisted. Verified at `api.py:520-558`.
- **Bridge-before-removal (F5 — safe).** Workers run `docker run --rm`, but the outbox is a **host
  bind-mount** written under the staging dir; `_collect` runs after `_run` returns and before the
  `finally` `rmtree` (`provisioning/docker.py:80-96`). `--rm` removes the container, not the host
  staging bytes. No cleanup-before-copy race.

Residual sandbox issue:

- **F4 (nice-to-know).** On a *hard* wall-clock backstop kill, `backend_driver.py:129-131` bridges
  the outbox and then raises `SandboxError` — which propagates out of `collect_proposal` *before*
  `harvest` (`core.py:166` is before `:174`), so a complete descriptor the agent wrote at minute 5
  before being killed at minute 30 is discarded, then `regenerate()` deletes it. The graceful
  budget-exhaustion path (`container_entry.py:77-88`, exit 0) harvests normally and is the common
  stop, so this only loses data on the rare hard kill. *Fix direction:* on `timed_out`, attempt
  harvest/parse of the bridged outbox first; fall back to `SandboxError` only if no valid descriptor
  is present.

### 2.2 Control-plane cycle — the narrow exception net (S1–S3)

**Root cause:** `run_cycle` (`api.py:520-548`) wraps the sandbox call in `except SandboxError` and
the commit call in `except GateUnavailable` — and nothing else. The cycle loop in `run()`
(`api.py:605-632`) has no per-cycle catch-all. So any predictable boundary error that is *neither*
type escapes both and aborts the run, with the in-flight cycle never passed to `_record_cycle`
(dropped from history and the `RunReport`) and the workspace never regenerated.

- **S1 (should-fix).** `store.put_object` (`store.py:600-606`) does `write_bytes` → raises a **raw
  `OSError`** on disk-full / permission / I/O error, at the harvest-into-store boundary
  (`api.py:352`). `store.get_object` raises `StoreError`. Neither is caught. Worse, `serve_context`
  is called at `api.py:517` **outside** the try block, so `_materialize_objects` (`core.py:154`) and
  object provisioning (`api.py:294`) can raise raw `OSError`/`StoreError` before the guards even
  apply. The CLAUDE.md hardening bar explicitly forbids "a raw `OSError`/`KeyError` escaping to
  abort a run."
- **S2 (should-fix).** `RemoteVerifier.dispatch` maps only `TransportUnavailable → GateUnavailable`
  (`transport/client.py:62-66`). `parse_result` raises a **bare `TransportError`** (the base class)
  on a malformed envelope or unknown server error-kind (`transport/envelope.py:69,75`), and
  `verdict_bundle_from_dict` can raise `KeyError`/`ValueError` on a malformed-but-`ok` body. The
  dispatch bridge catches only `FuturesTimeout` (`api.py:447-456`). Net effect: a verifier that is
  *reachable but misbehaving* (500 with an unmapped kind, truncated JSON) is **fatal**, while an
  unreachable one degrades gracefully — the opposite of the intended posture. Relevant when the
  ablation runs against the sibling-container verifier across 150 cells.
- **S3 (should-fix, harvester-conditional).** By the time `_harvest_children` runs
  (`api.py:366-381`) the parent is already committed `ACCEPTED`. If a child commit raises
  `GateUnavailable`, it propagates *before* `return result`; `run_cycle`'s handler then records the
  cycle as `entered_protocol=False, gate_error=...`, so the durable store says ACCEPTED while the
  `RunReport` records a gate failure, the parent's real `CommitResult` is lost, and the consecutive-
  failure counter is wrongly incremented. A `StoreError` from the child's `get_object` escapes
  entirely (per S1).

**Cross-cutting fix:** add a final `except Exception` in `run_cycle` that logs, regenerates the
sandbox, records the cycle as a failure (so `RunReport` stays complete and the consecutive-failure
cap still bounds runaways), and returns — turning every predictable boundary error into a
recorded-not-fatal cycle. Add per-boundary typed errors so the *cause* is still classified (S1:
`put_object`; S2: catch base `TransportError` in `RemoteVerifier.dispatch`).

### 2.3 Verifier gate logic (G1, G2, G3; accept_policy verified)

- **`accept_policy` is correct (verified, non-issue).** `always` → `score_and_accept`
  (`feature_engineering.py:364-384`): scores, records on the shared ledger, ACCEPTs unconditionally
  with **no `supersedes`** (every attempt retained) — except it rejects a *non-scorable* run, which
  is the intended "fair, non-hobbled" construct, not an implicit gate. `improve_over_best_prior` →
  `selection` (`:342-362`): `if score > bar: ACCEPT(supersedes=best_id) else REJECT`. Tie handling
  is strict `>` everywhere (`selection`, `numeric_scorer` at `primitives.py:125`), so
  `score == baseline ⇒ REJECT` — load-bearing for the "non-improving ⇒ rejected" claim. The knob
  works as the ablation assumes.
- **G3 (should-fix — the real metric distorter).** Both "best prior" reads come from in-process dict
  ledgers: `_scores` (`feature_engineering.py:292,349,386-395`) and the kaggle
  `_proxy_scores`/`_real_scores` (`feature_engineering_kaggle.py:117-118,151-161`), with
  `baseline = 0.0` when empty. Since #73 the verifier is a long-lived sibling container, but the
  ledger is never persisted. A restart mid-run resets it → `baseline = 0.0` → the next submission
  accepts even if **worse** than the true incumbent (silent regression), and `supersedes=None` so
  the real incumbent is never retired (two accepted artifacts, understated baseline). Under
  `improve_over_best_prior` this breaks the ≤1-accepted / monotone-best invariant that Claim A's
  zero-regression result leans on. *Fix direction:* persist the ledger, or recompute incumbents'
  scores from the store-slice on demand so the baseline survives a restart.
- **G1 (should-fix, latent).** `_run_pipeline` (`service.py:121-125`) guards no-decision only via
  `ruled < len(hard)`. If `hard == []` (cheap-only pipeline) or the pipeline is empty, `0 < 0` is
  False and it falls through to **`ACCEPTED`** — having ruled on no hard gate at all (empty pipeline
  → ACCEPTED with zero decisions). This is exactly the "implicit accept" the architecture forbids,
  leaking in at the verifier layer. Not reachable by the four shipped pipelines (`fake`, `code`,
  `fe`, `fe-kaggle` all include a hard gate), but `SdkVerifier` enforces no minimum. *Fix:* treat an
  empty hard stage as a config error (or rest at TENTATIVE); assert `len(hard) >= 1` at construction.
- **G2 (should-fix, latent footgun).** `supersedes = verdict.supersedes or supersedes`
  (`service.py:120`) is **last-non-None-wins** (the comment-implied "first wins" is wrong) and runs
  inside `for stage in (cheap, hard)` — so a *cheap* gate can set supersession, and if two gates
  name incumbents only the last is retired (the earlier intended incumbent silently stays
  `accepted`). Not triggered by shipped gates (no cheap gate sets `supersedes`; each hard gate sets
  one), but unguarded. *Fix:* take `supersedes` only from the terminal accepting hard decision;
  reject a bundle carrying more than one distinct `supersedes`.
- **G5 / G7 (nice-to-know).** `_assert_consistent` (`commit.py:241-251`) is one-directional (checks
  REJECT⇒REJECTED and REFINE⇒REVISED, not the converse, so it wouldn't catch a fabricated
  empty-decision ACCEPT). `_KaggleGates._run_cache` is never `.clear()`-ed (grows per task; memory
  only). Neither affects shipped correctness.

### 2.4 Answer-key / hold-out isolation (sound in code; convention at the boundary)

Traced end to end (`tools/prepare_fe_data.py` + `tools/harness/dataset.py` → `service/cli.py` role
routing → `composition/fe_*` → `sandbox/core.py` → `verifier/holdout_service.py` + FE gates →
feedback). **The shipped code introduces no leak:**

- The split is pure-stdlib, deterministic, and **row-disjoint** (`dataset.py:44-87`): the agent's
  train labels are not the holdout labels; the holdout features drop the target. Fixed across all 10
  seeds (good — the hold-out isn't re-randomized per seed).
- The verifier holds an **independent copy** and re-runs the agent's script on its own
  `agent_train_csv + holdout.csv`, scoring against its own `reserved_labels`
  (`feature_engineering.py:302-313,348`). The agent's `predictions.csv` is never trusted; the
  workspace is regenerated each cycle so edits don't persist.
- `reserved_labels` are **never** placed on a runner input or a worker mount — they live only in the
  (separate) verifier container's memory. Submitted code cannot reach them via filesystem, env, or
  network (the answer key is a private split that exists nowhere to "look up"). Credentials are not
  forwarded (`WorkerSpec.env_passthrough` defaults to `()`).
- No leak via harvest-back (only `SUBMISSION` script/requirements are harvested, never predictions
  or labels), served context, or verdict feedback (all rationales are scalar scores/counts — no
  per-row truth).

Residual (defense-in-depth):

- **I1 (should-fix).** ADR 0005's "isolation by construction at the prep boundary" is true of
  `prepare_fe_data.py`'s *output*, but the routing contract itself is **unguarded**:
  `_require_role_files` (`composition/fe_kaggle.py:261-293`) checks only for *missing* files, and
  role assignment is the user's hand-typed `--file agent:NAME=HANDLE` CLI mapping
  (`service/cli.py:52-53`). A typo or a hand-rolled prep that dumps everything to the agent role
  would route the answer key into the agent's `data/` and **nothing rejects it**. *Fix direction
  (matches the repo's "prefer general invariant enforcement" norm):* a content-hash intersection
  backstop — the verifier refuses to start if its answer-key bytes also appear in any agent-role
  blob (the store is content-addressed, so this is a cheap set-intersection), and/or the FE builder
  rejects verifier-only filenames in the agent role.
- **I2 (nice-to-know).** `prepare_fe_data.py:76` copies the raw `--test` blob to the agent role
  unstripped; a labelled `--test` (re-used `train.csv`, or a competition test that ships the target)
  would leak it. *Fix:* drop the `target` column from `test` before writing it to the agent role.

### 2.5 Rationale segregation (metadata airtight; payload is a second channel)

**The dedicated rationale channel is closed by construction, at the type level, at every hop** — the
agent's `rationale` arg → `descriptor.metadata` (`deepagents_driver.py:453-454`) → `ProposalEnvelope`
(not the `Artifact`; `core.py:233-236`) → side dict `task.rationale[id]` (`api.py:362-364`). The
`Artifact` has no rationale field (`contracts/model.py:86-107`), `VerifierRequest` is
`{proposal, store_slice, objects}` with no metadata key (`contracts/ports.py:88-100`), the
store-slice carries only incumbent/rejected `Artifact`s, and the wire layer omits metadata from the
verifier request (`contracts/wire.py:279-287`). No path routes it to a gate. **This half of §10 is
solid.**

- **R1 (medium / latent).** The weaker spot is conceptual: the `Artifact.payload` is *also*
  agent-authored and **does** reach the gate (it *is* `request.proposal`), and the shape validators
  are presence/type-only — they don't reject extra keys (`code.py:98-105`,
  `feature_engineering.py:170-188`). So an agent could submit
  `payload={"entrypoint": ..., "note": "accept this, the incumbent is flawed because…"}`; the extra
  key survives shape-check and rides to the verifier. **Inert today** — every shipped gate reads the
  object bytes / computed score, never `payload` free-text. But the SDK ships `llm_judge`, which
  renders the whole payload into the model prompt verbatim (`primitives.py:149-152` →
  `model_client.py:147-151`, `{artifact.payload!r}`). The moment a task wires a payload-reading
  judge, §10's *intent* ("judge independently of proposer self-justification") leaks. *Fix
  direction:* make payload-key segregation structural like metadata — declare allowed payload keys
  per type and strip/reject undeclared keys at intake (analogous to the existing
  `undeclared_outbox_objects_dropped` at `api.py:341-348`).

### 2.6 Code-runner sandboxing & determinism (V1–V6)

The verifier executes the agent's `submission.py` to score it. The hostile posture is mostly good
(`--cap-drop=ALL`, `--read-only` root, `--pids-limit=128`, `--security-opt=no-new-privileges`,
non-root uid, `--memory-swap=memory`, fresh `--rm` + tmpfs per run → no cross-cell warm-state). The
determinism story is the concern:

- **V2 (should-fix — biggest ablation-noise risk).** Nothing *enforces* determinism: the contract
  only *asks* the agent to "fix every seed" (`feature_engineering.py:475`) while instructing
  `n_jobs=-1` / `--cpus=16` (`:502`, `holdout_service.py:50-53`). Multithreaded GBM training
  (LightGBM/XGBoost/sklearn) is not bit-reproducible even with a fixed seed (float reduction order
  varies across threads). Each of the 10 replicates is a fresh process, so the in-process
  `_run_cache` doesn't hide it — identical code re-scores from scratch with thread-nondeterministic
  results, inflating within-condition variance. *Fix direction:* for the ablation verifier, pin
  single-threaded determinism (`OMP_NUM_THREADS=1`, `cpus=1`, `PYTHONHASHSEED`) via `_RUN_ENV`, or
  run each submission N times and record score variance so it's modeled, not misattributed.
- **V1 (should-fix).** `network=True` (`feature_engineering.py:309`,
  `feature_engineering_kaggle.py:133`) and the in-container command is one
  `sh -c "pip install … && python submission.py"`, so the **script trains with full outbound
  network**, not just the install phase. A submission can pull time-varying external data (score
  drift) or, for a public-leaderboard set, fetch known labels. Not exploitable for the private-split
  holdout experiment, but it's a noise source for *every* run. *Fix:* two-phase — `pip install` with
  network on, then re-exec the script with `--network=none`; at minimum document that submissions
  run net-connected.
- **V3 (should-fix).** Both runners read full child output into memory via `communicate()` with
  `PIPE` (`code_runner.py:249`, `docker.py:300`); the `--memory` cap bounds the *container*, not the
  pipe into the verifier process. A submission spewing stdout can OOM/hang the verifier host and
  take down the 150-cell sweep. *Fix:* cap captured output (read with a byte budget / redirect to a
  size-limited file and harvest the tail).
- **V6 (nice-to-know).** `pip install --no-cache-dir --target=… -r requirements.txt` resolves
  *transitive* deps to latest-compatible at install time; the agent pins top-level but not the full
  graph, so seeds run on different days can get different numerics. *Fix:* a frozen wheelhouse /
  fully-pinned constraints for the ablation runner image.
- **V5 (nice-to-know).** Timeouts are correctly enforced (`docker kill` then drain;
  `code_runner.py:248-254`), and `--rm` reaps on exit. Only if the verifier process itself dies
  mid-run does a named `--rm` container orphan; reaping is label-based but not wired into the
  synchronous path. Across 150 cells with restarts, orphans could oversubscribe cores. *Fix:* a
  pre-sweep `reap` by `harness=verity` label.

---

## 3. Hidden-instruction inventory

Every agent-facing instruction string was enumerated and classified against the documented
composition: `compose_system_prompt()` (kernel orientation + domain + task layers) + the per-cycle
user message + the documented middleware. **No stray-problematic instruction exists in the live
`fe-holdout` ablation path.** All agent-facing text is either part of that documented composition or
is recorded-only / unused-by-ablation. The full list:

### Legitimately in the composition (system prompt / user message)

| # | File:line | What it is | Bias risk |
|---|-----------|------------|-----------|
| 1 | `control_plane/config.py:114` | `KERNEL_ORIENTATION_TEMPLATE` (layer 1: work independently, self-verify, submit via tool) | None — invariant across conditions |
| 2 | `domains/feature_engineering.py:461` | `FEATURE_ENGINEERING_INSTRUCTIONS` (layer 2: script contract, "balanced accuracy", `ESTIMATED_BALANCED_ACCURACY` rationale line) | None — identical across conditions; metric name is the task definition, not a gameable leak |
| 3 | `composition/fe_holdout.py:96` | `_FE_HOLDOUT_INSTRUCTIONS` (layer 3: "improve upon any prior submission… reserved hold-out you never see; no leaderboard") | None — **constant** across all five conditions |
| 4 | `sandbox/core.py:247` | Per-cycle user-message nudge ("Do the work, then call exactly one operation tool… Reference … `parents`.") | None — constant |
| 5 | `control_plane/context.py:160` | `# Current goal` (the volatile tail; this is where `spec.goal` reaches the agent) | **See X1** |

### Documented middleware nudges (in path, all condition-invariant)

| # | File:line | What it is |
|---|-----------|------------|
| 6 | `deepagents_driver.py:139` | DeadlineMiddleware 80% pivot ("Time almost up… CALL YOUR PROPOSE/SUBMIT TOOL") — persisted HumanMessage |
| 7 | `deepagents_driver.py:173` | DeadlineMiddleware 50–80% pacing note ("[time budget] ~Xs remaining…") — ephemeral, never persisted |
| 8 | `deepagents_driver.py:212` | StepBudgetMiddleware nudge ("Step budget almost spent… finalize and submit…") |
| 9 | `deepagents_driver.py:71` | `_FINALIZE_NUDGE` ("You have NOT submitted a proposal… Do not stop until that tool returns success") — on no-proposal re-invoke |

Budgets are equal across conditions (§1.3 equal-compute control), so these fire on the same
pressures everywhere — no differential bias.

### Tool docstrings / runtime returns (tool text the model reads)

| # | File:line | What it is |
|---|-----------|------------|
| 10 | `deepagents_driver.py:470` | `propose.__doc__` (op-derived contract: "Actually RUN your script… no partial credit… rationale: never shown to the verifier") |
| 11–12 | `deepagents_driver.py:437-463` | propose tool returns ("payload must be a non-empty JSON object", "you declared … but no such file is in outbox/", "proposal recorded…", "tested=false … run it first next time") |

All generic and op-derived; identical across conditions.

### Recorded-only or unused-by-ablation (stray-but-benign)

- `feature_engineering.py:382` — the `score_and_accept` "always-accept: score recorded" rationale is
  an **ACCEPT** verdict, and `_feedback_from` returns `""` for accepts (`api.py`), so it **never
  reaches the model** — the agent is never told which policy it's under. (Recorded in provenance.)
- `sandbox/tools.py:26` `read_pdf.__doc__`, `composition/fe_holdout.py:103-112` catalog
  self-description, `composition/fe_holdout.py:81` `FE_HOLDOUT_GOAL` default,
  `composition/{code,fe_kaggle}.py` instructions — none wired into the `fe-holdout` agent path.

### The one experiment-design item

- **X1 (should-fix — config, not code).** The agent sees `spec.goal` as `# Current goal` (item 5).
  The bottom rung and the loop ship it in **different files with different wording**:
  `spec.exp1.json` ends **"Submit your best work."**; `spec.loop.json` (and `spec.example.json`) end
  **"Your submission should improve upon any previous submissions."** If exp1 is run from
  `spec.exp1.json` and the loop from `spec.loop.json` (the layout the bottom-rung split created), the
  one-shot condition receives different goal wording than the loop conditions. It's defensible
  (exp1 is single-cycle, so "improve on previous" is inapplicable), and layer-3 task instructions
  are identical everywhere — but it's the only place agent-facing instruction *text* varies by
  condition. *Action:* either run all conditions from a single-goal spec, or state the one-clause
  difference explicitly in the writeup.

**No cross-boundary leak:** the proposer rationale (incl. the `ESTIMATED_BALANCED_ACCURACY` line) is
segregated and never reaches a gate (§2.5); the verifier emits no agent-facing instruction text
beyond the gate rationales catalogued in §4; nothing exposes the hold-out labels or the bar value
ahead of a scored reject.

---

## 4. Verifier outcome-message catalog

Every outcome the verifier can produce, the **verbatim** message string, and its source. Verdict
kinds: `accept | reject | refine` (`contracts/model.py:56-61`). Statuses:
`proposed → tentative → accepted`, plus `rejected | superseded | revised` (`:37-45`). The pipeline
short-circuits: any REJECT → REJECTED, any REFINE → REVISED, a hard gate that returns `None`
(human-in-loop) → TENTATIVE, all-pass → ACCEPTED; there is **no implicit/silent accept** in the
shipped pipelines (but see G1 for the empty-hard-stage kernel gap).

### ACCEPT

| Message (verbatim) | Source |
|--------------------|--------|
| `submission parses as Python` | `domains/code.py:125` |
| `submission ran clean` | `verifier/primitives.py:179` |
| `net {net:.4f} beats baseline {baseline:.4f} (margin {margin:.4f})` | `verifier/primitives.py:128` |
| `runs clean and predicts every reserved row` | `domains/feature_engineering.py:340` |
| `improves: balanced-accuracy {score:.4f} vs bar {bar:.4f} (incumbent {baseline:.4f}, {trials} prior trials)` (sets `supersedes=best_id`) | `domains/feature_engineering.py:359-360` |
| `balanced-accuracy {score:.4f} (always-accept: score recorded, accepted)` | `domains/feature_engineering.py:380-382` |
| `runs clean on the hold-out` | `domains/feature_engineering_kaggle.py:184` |
| `local improvement: local proxy balanced-accuracy {net:.4f} vs best accepted {best:.4f} (margin {margin:.4f})` | `domains/feature_engineering_kaggle.py:204` |
| `clears the top-{fraction:.0%} bar: estimated public score {estimate:.4f} >= target {target:.4f}` | `domains/feature_engineering_kaggle.py:235-237` |
| `leaderboard unavailable; competitive bar skipped` | `domains/feature_engineering_kaggle.py:224` |
| `improves on the leaderboard: Kaggle public score {real:.5f} vs best accepted {best_real:.5f}` (sets `supersedes=best_id`) | `domains/feature_engineering_kaggle.py:282-284` |
| `well-formed` / `worth keeping` (fake/test domain) | `domains/fake.py:151,157` |

### REJECT

| Message (verbatim) | Source |
|--------------------|--------|
| `no {ENTRYPOINT!r} attachment to judge` / `...to run` / `...to execute` / `...to submit` | `code.py:120`, `feature_engineering.py:322`, `primitives.py:204`, `feature_engineering_kaggle.py:167,262` |
| `syntax error: {exc}` (with `defects`) | `domains/code.py:124` |
| `submission timed out before completing` | `primitives.py:174`, `feature_engineering.py:324`, `feature_engineering_kaggle.py:169` |
| `submission exited {exit_code}: {tail}` (stderr tail) | `primitives.py:178`, `feature_engineering.py:326-328`, `feature_engineering_kaggle.py:171-173` |
| `no well-formed {PREDICTIONS_OUTPUT} produced` | `feature_engineering.py:332`, `feature_engineering_kaggle.py:177` |
| `predictions cover {n} of {m} reserved rows ({k} missing)` | `feature_engineering.py:335-338` (and `..._kaggle.py:180-182`) |
| `submission produced no scorable predictions` | `feature_engineering.py:347,377` |
| `does not improve: balanced-accuracy {score:.4f} vs bar {bar:.4f} (incumbent {baseline:.4f}, {trials} prior trials)` | `feature_engineering.py:362` |
| `no scorable hold-out predictions` | `feature_engineering_kaggle.py:194,217` |
| `no local improvement: local proxy balanced-accuracy {net:.4f} vs best accepted {best:.4f} (margin {margin:.4f})` | `feature_engineering_kaggle.py:203` |
| `submission did not regenerate a valid file on the real test` | `feature_engineering_kaggle.py:270-271` |
| `no leaderboard improvement: Kaggle public score {real:.5f} vs best accepted {best_real:.5f}` | `feature_engineering_kaggle.py:286` |
| `net {net:.4f} does not beat baseline {baseline:.4f} (margin {margin:.4f})` | `primitives.py:133` |
| `marked reject` / `not worth keeping` (fake/test domain) | `domains/fake.py:147,156` |

### REFINE (→ REVISED; recovers a mostly-sound artifact with a localized defect)

| Message (verbatim) | Source |
|--------------------|--------|
| `below the top-{fraction:.0%} bar: estimated public score {estimate:.4f} vs target {target:.4f} (gap {gap:.4f}); keep improving the held-out score before we spend a submission` (with `defects=(msg,)`) | `feature_engineering_kaggle.py:230-234` |
| `marked defect` (with `defects`) (fake/test domain) | `domains/fake.py:150` |
| LLM-judge verdicts are prefixed `[weak: llm-judge — a model verdict, not a measurement] {rationale}` | `primitives.py:143,155` |

### TENTATIVE (no verdict — rests pending a human; §8.3)

`human_in_the_loop` returns `None` (logs only); the pipeline yields TENTATIVE via `ruled < len(hard)`
(`primitives.py:225-232`, `service.py:121-123`).

### Errors (not verdicts)

| Error / message (verbatim) | Source | Disposition |
|----------------------------|--------|-------------|
| `NoImplicitAccept`: `type {type!r} is not covered by a gate: committing it is an error, not a default-accept (no implicit accept, §5.7)` | `control_plane/commit.py:144` | fail-fast (misuse) |
| `ProposerIsGate`: `the verifier's identity equals the proposer's ({id!r}): the proposer is never the gate (Builder/Breaker rule, §7.2)` | `control_plane/commit.py:155` | fail-fast (misuse) |
| `VerifierError`: `this verifier has no pipeline for type {type!r} (covers: {…}): a covered type with no pipeline is a config error, never a silent pass (§3.6, §5.7)` | `verifier/service.py:83-86` | fail-fast (misuse) |
| `GateUnavailable`: `verifier did not respond within {timeout}s` | `control_plane/api.py:476` | recoverable (degrade) |
| `GateUnavailable`: `verifier unreachable: {exc}` | `transport/client.py:64` | recoverable |
| `GateUnavailable`: `no Kaggle submission budget within {deadline}s (daily cap)` | `feature_engineering_kaggle.py:296-297` | recoverable |
| `GateUnavailable`: `could not run the submitted code: {exc}` | `verifier/code_runner.py:236` | recoverable |

### What the agent actually sees next cycle (control-plane feedback wrappers)

`_feedback_from` (`control_plane/api.py:726-759`) wraps the above before serving them back. **Accepts
produce no feedback** (`""`):

| Wrapper (verbatim) | Source |
|--------------------|--------|
| `sandbox-error: your previous attempt did not produce a valid submission ({sandbox_error}); produce exactly one complete proposal within the budget` | `api.py:735-737` |
| `gate-unavailable: the verifier could not evaluate your last proposal ({gate_error}) — this was not a rejection; submit your proposal again` | `api.py:740-742` |
| `shape-error: {message}` | `api.py:745` |
| `refine: {", ".join(defects)}` | `api.py:748` |
| `rejected: {first REJECT decision's rationale}` (fallback `the submission did not clear a gate`) | `api.py:750,754-759` |

---

## 5. Chat-context instrumentation: feasibility + recommendation

### Current state

The agent's full message history is a LangGraph in-memory message list returned by `agent.invoke()`
(`deepagents_driver.py:282`). It is **discarded on `regenerate()`** every cycle. Two diagnostics
escape:

- **`__telemetry__.json`** — written *always* and **harvested** back to the host/store
  (`run_agent` at `deepagents_driver.py:286-289`; popped in `collect_proposal` at `core.py:180`).
  Captures summed tokens, `model_steps`, `tool_calls`, model name (`_extract_telemetry` `:335-358`).
- **`__transcript__.json`** — written **only when the cycle produced no proposal**
  (`deepagents_driver.py:290-291` guards on `if not (outbox / RESERVED_PROPOSAL_NAME).is_file()`),
  is **not harvested** (`_persist_transcript` `:315-332`; `core.py` never pops it), and captures only
  `{type, content, tool_call *names*}` — no tool-call arguments, no per-step timing.

So the cycles the ablation most wants to study — the *successful* ones — leave no step-level record,
and even the failure transcript omits tool arguments and stays stranded in worker staging.

**Feasibility: high, and the seam already exists.** The telemetry path is the exact pattern to
follow: write a reserved file in the outbox → it rides the existing harvest bridge back to the host
for both the in-process and container drivers (the container path bridges the outbox via
`backend_driver._bridge_outbox`, so a reserved transcript name is carried automatically — no
container-specific work). The ephemeral-regenerate lifecycle is *not* an obstacle because harvest
runs before regenerate on the success path.

### Recommended approach (no code — design only)

**Promote the transcript to a first-class, always-on, harvested artifact, parallel to telemetry.**

1. **Always write it.** In `run_agent` (`deepagents_driver.py:286-291`), drop the no-proposal guard
   so `_persist_transcript` runs on *every* cycle (success and failure alike), writing a reserved
   name (e.g. `RESERVED_TRANSCRIPT_NAME = "__transcript__.json"` — already a reserved-style name).
2. **Capture more per step.** Extend `_persist_transcript`'s rendering (`:322-328`) from
   `{type, content, tool_call names}` to include **tool-call arguments**, tool **results**
   (the `ToolMessage` content), and a per-message index/role. If wall-clock per step is wanted,
   stamp messages as they arrive (a lightweight middleware, mirroring the existing Deadline/Step
   middlewares, is the natural place). Redact nothing structural, but see the caveat below.
3. **Harvest it.** In `collect_proposal` (`core.py:179-180`), `pop` the reserved transcript name
   alongside the descriptor and telemetry so it does **not** become a domain object attachment, then
   route it like telemetry: content-address it via `store.put_object` and carry an `ObjectRef`.
4. **Thread it into the report.** Add a `transcript_ref` (an `ObjectRef` / on-disk path) to the
   `ProposalEnvelope` → `CycleInput` (`api.py:560-582`) → `CycleReport` (`run_report.py`), exactly
   where `agent_telemetry`, `served_context_chars`, and `provisioned_object_bytes` already live. The
   analyzer (`experiments/ablation/analyze.py`) can then resolve the ref per cycle for step-level
   analysis without bloating `summary.json`.

This is the smallest change that gives every cycle a complete, durable, analyzable transcript, reuses
the existing harvest/store/RunReport machinery, and works identically for the in-process and
container drivers.

**Caveats to fold into the design:**

- **Volume.** Full transcripts (with tool results, which include CSV dumps and stderr) are large
  ×150 cells × N cycles. Store them as content-addressed objects (the store already dedups by hash)
  and keep only the `ObjectRef` in the report, not the inline text. Consider a size cap / tail on
  individual tool results (this also mitigates V3's unbounded-output concern at the capture layer).
- **Secrets/PII.** The agent transcript can echo data rows and, in principle, environment strings.
  For the holdout experiment the agent never has labels or credentials (§2.4), so the exposure is
  limited to the public feature data — acceptable for an internal experiment, but worth a one-line
  redaction pass on any `env`/token-shaped content before persisting if transcripts are ever shared.
- **Determinism of the record.** The transcript is an *observation*, so it doesn't affect scoring —
  but capturing it must stay best-effort (never fail the cycle), exactly as `_persist_transcript`
  already is (`:329-332`).

**Smaller fallback (if full dumps are too heavy):** keep the always-on per-cycle write but record
only a **structured step log** — an ordered list of `{step_index, role, tool_name, tool_args_digest,
content_len, ms}` — dropping large `content`/results. This still reconstructs the agent's *trajectory*
(what it did, in what order, how long) for instrumentation while keeping each record small, and is a
strict subset of the recommended renderer.

---

## 6. Pre-flight checklist (ordered)

Before launching the 150-cell sweep, in rough priority:

1. **Pin scoring determinism (V2, V1, V6).** Force single-threaded, seeded, network-isolated
   training in the ablation verifier (`OMP_NUM_THREADS=1`, `cpus=1`, `PYTHONHASHSEED`, two-phase
   pip-then-`--network=none`, frozen deps) — *or* run each submission N times and record score
   variance. This is the highest-leverage fix for clean deltas. **If not fixed, document the noise
   floor** and treat sub-threshold deltas with suspicion.
2. **Persist or recompute the score ledger (G3)** so a verifier-container restart mid-sweep can't
   collapse the baseline to 0 and manufacture a regression — the property Claim A rests on. At
   minimum, avoid restarting the verifier mid-run and log if it happens.
3. **Widen the cycle exception net (S1–S3)** — a catch-all in `run_cycle` that records-and-continues,
   plus typed errors at `put_object` and `RemoteVerifier.dispatch` (catch base `TransportError`) — so
   a transient I/O or verifier hiccup degrades one cycle instead of silently truncating a run's
   `RunReport`.
4. **Cap captured child output (V3)** so a runaway submission can't OOM the verifier host and take
   down the sweep.
5. **Resolve the goal-wording confound (X1)** — run all conditions from a single-goal spec, or state
   the one-clause difference in the writeup.
6. **Add the answer-key intersection backstop (I1)** and strip the target from the agent's `test.csv`
   (I2) — cheap defense-in-depth that turns "isolation by construction" into an enforced invariant.
7. **Instrument transcripts (§5)** — promote the transcript to an always-on, harvested,
   report-referenced artifact so successful cycles are analyzable; this is additive and low-risk.

**Latent / track-don't-block:** the empty-hard-stage silent accept (G1), `supersedes` last-wins
(G2), and the payload-as-second-channel / `llm_judge` leak (R1) are not reachable by the shipped
`fe-holdout` / `fe-kaggle` pipelines but are unguarded kernel/SDK gaps worth an issue each, in line
with the repo's "prefer general invariant enforcement" norm.
