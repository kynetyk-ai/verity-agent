# Verity

**An implementation of the Self-Revising Discovery Harness** — a domain-agnostic kernel that gives an
agent a *typed provenance record* (artifacts, the operations that produced them, and the gate
decisions about them) as its durable state, behind a verifier-aware commit lifecycle. The shorthand
is *git + a type system + a verifier-aware lifecycle for agent artifacts.*

The name carries the point: **verity** = truth, and *verify*. The system exists to answer one
question — *can you trust, and audit, what the system believes?*

> **Status:** greenfield scaffold. No implementation code yet; this repo currently holds the vendored
> specification and project orientation. Stack: Python, managed with [uv](https://docs.astral.sh/uv/)
> (see *Coding habits* in [CLAUDE.md](CLAUDE.md)).

## The specification (self-contained)

The full natural-language spec and its grounding references are **vendored into this repo** so it is
self-contained and can be worked on anywhere:

```
spec/
  self-revising-discovery-harness.md      the spec (v0.2) — the authority for this implementation
  references/
    2606.01444v1.pdf                      Wang & Buehler, the conceptual-origin paper
    scienceclaw-evaluation.md             prior-art analysis (typed provenance, no gate)
    pilar-comparison.md                   prior-art analysis (enforced soft gate)
```

These are a copy synced from the sibling spec repo (`NL-specs`,
`agents-and-harnesses/self-revising-discovery/`). If the upstream spec changes, re-sync the `spec/`
tree. Treat `spec/self-revising-discovery-harness.md` as the source of truth for what to build;
implementation decisions should trace to a spec section (cited `§N`).

## Architecture in brief

Three cooperating services (spec §3.3–§3.6):

- **Control plane** — the only service that mutates durable state; deliberately unintelligent. Owns
  the typed-provenance store, the commit path, the lifecycle, context assembly, the registries, the
  task config, loop control, and the invariant workspace contract + composed system prompt.
- **Sandbox** — the agent runtime + workspace; ephemeral (regenerated each cycle); proposes but never
  writes; emits objects to the workspace **outbox** for harvest.
- **Verifier** — advisory; hosts the gate plugins; takes a shaped proposal + a declared store-slice
  and returns a verdict; never writes.

The loop is **read → propose → gate → commit**. Load-bearing rules: **no implicit accept**, the
proposer is never its own gate, a status richer than accept/reject
(`proposed → tentative → accepted`, plus `rejected` / `superseded` / `revised`), and the `refine`
verdict for recovering a mostly-sound artifact with a localized defect.

## Roadmap

The path to MVP lives in **[ROADMAP.md](ROADMAP.md)** — it is the canonical "where are we, what's
next." In short: build the **control plane** first and prove it against bespoke test doubles
(Phase 1), then swap in the real **verifier** (Phase 2) and **sandbox** (Phase 3), and reach **MVP**
with the feature-engineering domain passing the spec §13 acceptance criteria (Phase 4).

## Develop

```
just install     # uv sync
just check       # lint + type-check + test (must be green before any PR)
just test        # pytest
```

## Repo layout

```
README.md     this file
CLAUDE.md     orientation for coding agents working in this repo
ROADMAP.md    the living path to MVP
pyproject.toml / justfile   uv project + dev commands
src/verity/   the package (control_plane/ is the Phase 1 deliverable; mostly stubs today)
tools/harness/  non-product integration doubles (stub agent + stub verifier)
tests/        test suite
spec/         vendored specification + references (see above)
```

---

The vendored specification is © 2026 Kynetyk Holdings LLC; all rights reserved.
