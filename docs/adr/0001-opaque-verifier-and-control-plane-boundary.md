# ADR 0001 — Opaque verifier + the control-plane-never-parses boundary

- **Status:** Accepted — implemented on `feat/verifier-service` (PR #7)
- **Date:** 2026-06-07
- **Affects spec:** §3.3, §3.4, §3.6, §7, §8.3, §10 (amendments below), §6 (one added edge)
- **Supersedes implementation:** the per-gate dispatch + control-plane staging shipped in Phase 2
  (PR #7). Folds in issue #8 (supersede-on-beat); makes issue #10 (capability advertisement)
  load-bearing.

## Context

The spec as written puts the gate **binding** — which gates run, in what order, the cheap/hard
split, and each gate's declared inputs — in the **control plane**, which dispatches gates one at a
time and advances status by pipeline position:

- §3.6: *"a binding (held in the control plane) names which primitives run in what order."*
- §7.3–4: the control plane *"runs the pipeline in order… for each gate… dispatches… advances
  status by pipeline position."*

Phase 2 implemented exactly this: `VerifierRequest` carries a single `gate` name, the commit path
loops the binding's gates (`_run_stage`, cheap → `tentative`, hard → `accepted`), and `SdkVerifier`
maps one gate name → one plugin.

The product vision is different and sharper: **the verifier is an opaque, user-selected package.**
The user chooses a verifier from a list or builds their own; the control plane does **not** sequence
its gates and does **not** parse the proposal. The control plane's only look at a proposal is a
**presence check** — "is it well-formed, are all the expected parts there?" — and if that fails the
verifier is never triggered. There is no quality review in the control plane, and **no parsing**:
all interpretation lives in the verifier.

This is a divergence from the spec, so it is raised here rather than drifted into.

## Decision

Three coupled changes, plus the invariants the control plane keeps regardless.

### (a) Opaque verifier — one handoff, a verdict bundle back

The control plane dispatches the **whole proposal** (the artifact + its declared store-slice +
object attachments) to the **one** verifier the task selected, in a **single** call. It carries no
gate name and implies no pipeline.

The verifier — opaque; runs whatever checks, in whatever order, internally — returns a **verdict
bundle**:

```
VerdictBundle {
  status:     accepted | tentative | rejected | revised   # the terminal status it recommends
  decisions:  [ { gate, verdict, rationale, score?, defects? }, ... ]  # one row per check that ruled
  supersedes: artifact_id?                                # the incumbent this beats, if any (issue #8)
}
```

The control plane stays the unintelligent sole-mutator: it **records** the decision rows, **applies
supersession** if named (atomically with accept), and **sets** the recommended status — but only
after two cheap, non-interpretive validations:

1. **Transition legality (§6).** The recommended status must be a legal transition from `proposed`.
   The control plane will not write an illegal edge even on the verifier's say-so.
2. **Bundle self-consistency.** A `reject` decision ⇒ status `rejected`; a `refine` decision ⇒
   status `revised`. This keeps a misbehaving verifier from, e.g., reporting `accepted` over a
   reject. (This is enforcement of the bundle's internal coherence, not a review of the artifact.)

The cheap/hard split and the `tentative` decision move **into** the verifier: a verifier returns
`tentative` when its cheap checks passed but a hard or human check remains. `tentative` therefore
survives — it is reported, not inferred from pipeline position.

### (b) Declarative shape spec — presence only, mechanical

The proposal-shape spec (§3.4, §7.0) becomes **declarative data** — the required parts and their
types — checked mechanically by the control plane. The per-domain `ShapeValidator` *function* (which
runs in the control-plane process and could parse) is replaced by a declarative `ProposalShape`
(e.g. required fields, each with a primitive/object/list/object-ref type). The check answers only
"are the expected parts present and of the declared kind?" — never what they contain. This makes
"the control plane cannot parse" a **structural fact**, not a convention a domain author must honor.

### (c) The control plane never mutates the payload

Object↔artifact links are recorded as a **sidecar** on the artifact — a dedicated, control-plane-
owned association (`artifact → {name → ObjectRef}`) — **not** injected into the domain payload. The
agent's proposed payload is stored verbatim. (This reverses the `_attach_objects` payload injection
added during the audit.) §4.1's "the artifact references its object" is satisfied by the sidecar; the
control plane content-addresses and binds the harvested bytes without reaching into the proposal.

### Invariants the control plane still owns (unchanged in force, relocated in mechanism)

| Invariant | How it holds in the opaque model |
|---|---|
| **No implicit accept (§5.7)** | A committable type must be **registered as gated** (a thin `type → {declared_inputs}` registration). An unregistered type → `NoImplicitAccept`, before any dispatch. A registered type whose verifier returns *no* verdict is also an error, never a silent accept. (Capability advertisement, issue #10, upgrades this from commit-time to configure-time.) |
| **Proposer ≠ gate (§7.2)** | Structural: the verifier is a separate package/identity the proposer cannot reach. Asserted once (verifier identity ≠ proposal `created_by`), not per gate. |
| **Independence (§8.3, §10)** | The control plane transmits **only** the declared store-slice for the type, and the §10 static check (resolved inputs ⊆ declared allowlist) holds **per type** instead of per gate. Rationale still cannot ride along (slice is `Artifact`-only; the request carries no proposer metadata). |
| **Sole mutator / lifecycle (§3.3, §6)** | The control plane records decisions, sets status, harvests/stores objects, and validates every transition. The verifier advises; it never writes. |

## Spec amendments (precise edits to lift upstream)

- **§3.3 mapping table.** Replace the two gate-registry rows with:
  - *Verifier coverage + per-type store-slice declaration* → **Control plane** (enforces no-implicit-
    accept; cuts the declared slice).
  - *The gate pipeline — which checks, order, cheap/hard split, the verdicts* → **Verifier package**.
  Strike "slices store-context to each gate's declared allowlist"; replace with "…to the verifier's
  declared allowlist for the type."
- **§3.6.** Strike *"a binding (held in the control plane) names which primitives run in what
  order."* Replace with: the verifier package owns its internal pipeline and returns a **verdict
  bundle** (recommended status + per-check decisions + optional supersedes); it **advertises** the
  types it covers and the store-slice each needs, so coverage and independence are checkable at
  configure time.
- **§7 step 0.** "Validate shape" is a mechanical check against the **declarative** proposal-shape
  spec (presence + declared types). No domain code runs in the control plane here.
- **§7 step 1.** "Resolve the gate pipeline" → "Resolve verifier **coverage**: the type must be
  registered as gated, else fail (no implicit accept)."
- **§7 step 3.** "Run the pipeline in order… for each gate, dispatch…" → "Dispatch the proposal + its
  declared store-slice + objects to the verifier **once**; receive a verdict bundle. Record a
  `decisions` row for every check the bundle reports."
- **§7 step 4.** "Advance status by pipeline position" → "**Set the status the bundle recommends**,
  after validating it is a legal transition (§6) and self-consistent with the decisions."
- **§6.** Add the edge **`proposed → accepted`** (the all-cheap / verifier-reports-accepted case;
  already permitted by §6's prose, now required by the table since the verifier owns staging). The
  control plane still validates every transition.
- **§8.3.** The two-layer split changes home: the **binding/pipeline** (order, cheap/hard, thresholds)
  moves **into the verifier package**; the control plane keeps a **coverage + declared-inputs**
  registration per gated type. "No implicit accept" is enforced by the coverage registration.
- **§10.** "each gate's resolved inputs ⊆ its declared allowlist" → "the verifier's resolved inputs
  **for the type** ⊆ the declared allowlist."
- **§3.4 / §4.1.** The control plane records object references as an artifact-level association and
  does not modify the proposal payload; the proposal-shape spec is declarative.

## Refactor plan (code)

Ordered so each step is green before the next; the Phase-2 assets that **survive** unchanged are the
gate-primitive SDK, the container `CodeRunner`, `verity.contracts` (extended), the independence slice
mechanism, and the two domains' *gates* (re-homed into the verifier package).

1. **Contracts.** `VerifierRequest` drops `gate`; carries `proposal`, `store_slice`, `objects`. Add
   `VerdictBundle {status, decisions, supersedes?}` and a `GateDecision` value type. `VerifierPort.
   dispatch(request) -> VerdictBundle`.
2. **Verifier.** `SdkVerifier` gains an internal **per-type pipeline** (ordered plugins + cheap/hard
   split) and a runner that executes it and assembles a `VerdictBundle` (computing the terminal
   status, collecting decisions, surfacing a `supersedes` from the selection check). The cheap/hard
   staging logic moves here from the commit path. Unknown/ uncovered type → an explicit "not covered"
   signal (not a silent pass). Primitives unchanged.
3. **Commit path.** Delete `_run_stage` and the per-gate loop. New shape: shape-check (declarative) →
   coverage check (`NoImplicitAccept` if unregistered) → proposer ≠ verifier → dispatch once → record
   decisions → validate + apply (supersede/refine/status). Keep the refine cap and feedback channel.
4. **Lifecycle.** Add the `proposed → accepted` edge; the commit path validates the bundle's status
   against the table.
5. **Registries.** `GateRegistry` (ordered `GateSpec`s) → a slim `GatedTypeRegistry`:
   `type → declared_inputs`; `resolve(type)` returns the inputs or `None` (→ no implicit accept).
   `GateSpec`/`GateBinding` move to the verifier-side pipeline config.
6. **Independence.** `resolve_declared_slice` keys off the **type's** declared inputs; the §10 check
   is unchanged in substance.
7. **Shape spec.** Replace `TaskConfig.shape_validator: Callable` with a declarative `ProposalShape`
   (+ a mechanical checker in the control plane). Domains declare shapes as data.
8. **Object sidecar.** Remove `_attach_objects`. Add an artifact-level object association (a new
   `Artifact.objects` field in `verity.contracts` + a store column / serialization, **or** a store
   association table), recorded at harvest. Provenance returns it; payload untouched.
9. **Domains (`fake`, `code`).** Each now supplies: a declarative shape, a `type → declared_inputs`
   registration, and a verifier package (an `SdkVerifier` configured with the type's internal
   pipeline incl. the cheap/hard split). The marker / parse / run gates are unchanged in logic.
10. **Tests.** Update `test_commit`, `test_verifier`, `test_independence`, `test_api`,
    `test_exit_criteria`, `test_code_loop`, `test_reproducibility`. New: verdict-bundle shape,
    coverage-based no-implicit-accept, declarative shape checking, the object sidecar, supersede-on-
    beat via the bundle.

## Issues resolved / affected

- **Resolves #8** (supersede-on-beat) — the verdict bundle's `supersedes` is how the selection check
  names the incumbent it beats.
- **Makes #10 load-bearing** (capability advertisement) — coverage advertisement upgrades no-implicit-
  accept from a commit-time error to a configure-time check.
- **Reverses** the audit's `_attach_objects` payload injection in favor of the sidecar (cleaner under
  this boundary).

## Sequencing recommendation

PR #7 is a complete, tested increment whose *substance* (SDK, container runner, contracts,
independence, domains) survives this change; only the control-plane↔verifier **seam** is reworked.
Recommendation: **merge #7**, then land this correction as a **focused follow-up PR** (this ADR as
its opening commit), rather than expanding #7 into an unreviewable rearchitecture. Alternative: hold
#7 and stack the refactor — at the cost of a much larger single PR.
