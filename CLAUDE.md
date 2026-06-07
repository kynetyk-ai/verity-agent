# Verity

Implementation of the **Self-Revising Discovery Harness** — a domain-agnostic kernel that
maintains an agent's state as a *typed provenance record* (artifacts, the operations that produced
them, and the gate decisions about them) rather than a chat transcript or a vector store, behind a
verifier-aware commit lifecycle. Shorthand: *git + a type system + a verifier-aware lifecycle for
agent artifacts.*

The name carries the point: **verity** = truth, and *verify* — the system exists to make what an
agent believes trustworthy and auditable.

## The spec is the source of truth

This repo implements a finalized natural-language specification, **vendored locally** so the repo is
self-contained (it may be worked on on a different machine):

```
spec/self-revising-discovery-harness.md      the spec (v0.2) — authoritative
spec/references/                              the grounding paper + prior-art analyses
```

Treat `spec/self-revising-discovery-harness.md` as authoritative. Implementation choices should trace
back to a spec section (cited as `§N`); if the code needs to diverge from the spec, raise it rather
than drifting. The `spec/` tree is a copy synced from the sibling spec repo (`NL-specs`,
`agents-and-harnesses/self-revising-discovery/`) — re-sync it if upstream changes.

## Architecture in brief

Three cooperating services (spec §3.3–§3.6):

- **Control plane** — the *only* service that mutates durable state; deliberately unintelligent. Owns
  the typed-provenance store, the commit path, the lifecycle, context assembly, the registries, the
  task config, the orchestration policy (loop control), and the invariant **workspace contract** +
  composed system prompt.
- **Sandbox** — the agent runtime + workspace where the loop runs and tools execute; ephemeral
  (regenerated each cycle); proposes but never writes; writes objects to the **outbox** for harvest.
- **Verifier** — advisory; hosts the gate plugins; takes a shaped proposal + a declared store-slice
  (+ object attachments) and returns a verdict; never writes.

The loop is **read → propose → gate → commit**. Load-bearing rules: **no implicit accept** (a type
with no declared gate cannot be committed), the proposer is never its own gate, status is richer than
accept/reject (`proposed → tentative → accepted`, plus `rejected` / `superseded` / `revised`), and
the `refine` verdict recovers a mostly-sound artifact with a localized defect.

## Build order (spec §17)

1. Kernel contracts — Store interface + data model, audit-contract invariants (as property tests),
   the lifecycle state machine, the commit path.
2. Extension-point interfaces — schema / tool / gate registries + retrieval policy (with a fake
   domain, including a gateless type to prove "no implicit accept").
3. Context assembly — stable-prefix / volatile-tail split; regenerate-per-turn; bounded-context test.
4. Service scaffolding — split into the three services; prove the shape-error, advisory-verifier,
   `refine`, and object-harvest round-trips.
5. Feature-engineering domain (§12) — first real end-to-end run.
6. v1 acceptance criteria (§13) — twelve runnable checks; v1 is not done until they pass.

## Status

Greenfield scaffold — no implementation code yet. Stack is not yet fixed; **Python** is the working
assumption (given the v1 feature-engineering domain and the agent-loop tooling). Set conventions here
as the first service lands.
