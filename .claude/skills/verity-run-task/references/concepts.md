# What Verity is, and where it generalizes

Conceptual orientation only — stable ideas from the vendored specification. For *what is actually
installed and runnable*, read the live catalog (`verity catalog`), not this file.

## The kernel is domain-agnostic

Verity implements the **Self-Revising Discovery Harness** (spec
`spec/self-revising-discovery-harness.md`): a domain-agnostic **kernel** that keeps an agent's durable
state as a **typed provenance record** — artifacts, the operations that produced them, and the gate
decisions about them — behind a **verifier-aware commit lifecycle** (§1). Shorthand: *git + a type
system + a verifier-aware lifecycle for agent artifacts*. The loop is **read → propose → gate →
commit**, with load-bearing rules: no implicit accept, the proposer is never its own gate, and a status
richer than accept/reject (`proposed → tentative → accepted`, plus `rejected` / `superseded` /
`revised`).

## What is invariant vs. what a domain plugs in

> "**The gate is the moat. Everything else in this system generalizes across domains; the gate does
> not.**" (§2.4)

A domain supplies only its **schema** (artifact types + operation signatures), its **tools**, and — the
irreducible per-domain work — its **gates** (what is worth keeping). The kernel reads only the
meta-schema (names + typed signatures); it never interprets payload semantics, and the actual types stay
per-domain (§2.8). "Worth keeping" is two questions: **validity** (is it sound?) and **selection** (does
it beat the incumbent, net of cost?) (§2.5, §11).

## Where it generalizes (illustrative — read the live catalog for what's installed)

Any task where work products are **typed artifacts with provenance** and "worth keeping" is decidable as
*validity + selection* is a candidate. Spec-grounded examples:

- **Feature-engineering discovery** (§12) — the MVP domain: an agent designs features judged by running
  its script on a reserved hold-out it never sees.
- **Soft-gate consulting / medical writing** (§14) — a deliverable gated by independent,
  context-restricted reviewers + deterministic commit-time validators (the *pilar* lineage, §3.8).
- **Code-execution tasks** — submissions gated by parse + run-clean checks.
- More broadly, self-revising **scientific discovery** — the conceptual origin (Wang & Buehler,
  *Self-Revising Discovery Systems for Science*, arXiv:2606.01444, in `spec/references/`), and the
  typed-provenance prior art *ScienceClaw* (§3.7).

The thesis payoff (Phase 6): *good proposals from cheap models* — the disciplined record + gate let
smaller/local models contribute, because trust comes from the verifier, not the proposer.

## Pointers (in the Verity source repo — not bundled with this skill)

These resolve only if you have the repo checked out; at runtime the live source of truth is
`verity catalog`.

- `spec/self-revising-discovery-harness.md` — authoritative specification (cited `§N`).
- `spec/references/` — the origin paper + prior-art analyses.
- `docs/api-surface.md` — the control-plane API reference (incl. the standing-daemon + HTTP surfaces).
- `ROADMAP.md` — where the implementation is and what's next.
