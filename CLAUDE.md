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

## Roadmap and build order

The live plan is **[ROADMAP.md](ROADMAP.md)** — the canonical "where are we, what's next." Read the
lowest ⬜/🚧 item there before starting work, and keep it current (update it in the same PR that
changes the plan's reality).

The arc: **Phase 1** builds the control plane and proves it against bespoke test doubles (a stub agent
and a stub verifier in `tools/harness/`); **Phases 2–3** swap in the real verifier and sandbox;
**Phase 4** wires the feature-engineering domain (§12) and reaches MVP when the twelve §13 acceptance
criteria pass. This follows the spec's §17 build order (kernel contracts → extension interfaces →
context assembly → service scaffolding → domain → acceptance).

## Coding habits

Non-negotiable working norms for this repo:

- **Python**, managed with **uv**. All dependencies and runs go through `uv` (`uv add`, `uv sync`,
  `uv run`). **No system-level or global `pip install`** — ever.
- **Don't defer basic infrastructure to reach an MVP.** The plumbing that makes a system debuggable
  and trustworthy is built from the start, not retrofitted:
  - **Structured logging from day one** — every service logs; no `print`-and-hope.
  - **Tests written alongside the code** — good coverage as we go, not bolted on at the end.
- **Always work on a branch.** Never commit directly to `main`; branch, then open a PR.
- **Never open a PR on buggy or embarrassing code.** It runs, it's tested, and it's clean before it
  goes up for review. A PR is a finished thought, not a work-in-progress dump.

## Status

Greenfield scaffold — no implementation code yet. **Stack: Python, managed with uv** (see
*Coding habits*). Set further conventions here as the first service lands.
