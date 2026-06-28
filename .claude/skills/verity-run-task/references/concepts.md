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

## Acceptance modes — and how to set up each

"What counts as done" isn't a fixed behaviour or a single flag — it falls out of **how a task's
verifier composes its gates**, plus the run's **`--stop-on-accept`** policy and the task's
**object-provisioning** mode. Two facts drive it: only a **hard** gate reaches `accepted` (cheap checks
rest at `tentative`), and **supersession happens only when a gate's verdict names the incumbent it
replaces**. A task type's live `verifier_approach` (`verity catalog --type <t>`) describes which mode it
implements. The three shapes, and how to build each:

- **Optimizer — supplant the incumbent with a better one** (one best answer survives).
  *Verifier:* a hard **scoring** gate that accepts only when the new artifact beats the incumbent **and
  emits a supersession** (names the artifact it replaces). *Provisioning (a free, separate axis — see
  below):* `best_revised_or_accepted` hands the agent its single best prior; `all_accepted_or_superseded`
  hands it the whole **lineage of bests** (the prior accepts the gate has since superseded). *Run:* leave
  `--stop-on-accept` off (keep improving). This is the feature-engineering / `fe-kaggle` behaviour.
- **Accumulate — keep every acceptable answer** (a collection, not a winner).
  *Verifier:* a hard **validity** gate that accepts on soundness and **never supersedes** → all sound
  artifacts coexist as `accepted`. *Provisioning:* `all_accepted` (or `all`). *Run:* `--stop-on-accept` off.
- **First-acceptable — take the first good one and stop** (no optimization once you have one).
  *Verifier:* a validity gate (accept on soundness). *Run:* **`--stop-on-accept`** — the loop ends at
  the first `accepted`. *Provisioning:* whatever you like (it won't iterate).

**Provisioning is configured independently of the acceptance behaviour** — it's *what the agent sees next
cycle*, not *what counts as done*. Set `policy.provisioning` to a preset name (`best_revised_or_accepted`,
`all_accepted_or_superseded`, `all_accepted`, `none`, …) **or** an explicit
`{statuses: [...], select: all|last|best}` — any status subset (including `superseded`) × a selection. The
acceptance choice never constrains it: an optimizer gate that supersedes prior bests can still provision
the full superseded lineage. (For back-compat it also rides `verifier.knobs.provisioning` /
`provisioning_mode`.)

Practically: the **acceptance mode lives in the verifier + composition** (an authoring choice when a task
type is built), while **`--stop-on-accept`** and **`policy.provisioning`** are the per-run knobs you set at
`create`/`run`. If you're only *running* installed task types, read each one's `verifier_approach`
(`verity catalog --type <t>`) to know which acceptance mode you're getting.

## Pointers (in the Verity source repo — not bundled with this skill)

These resolve only if you have the repo checked out; at runtime the live source of truth is
`verity catalog`.

- `spec/self-revising-discovery-harness.md` — authoritative specification (cited `§N`).
- `spec/references/` — the origin paper + prior-art analyses.
- `docs/api-surface.md` — the control-plane API reference (incl. the standing-daemon + HTTP surfaces).
- `ROADMAP.md` — where the implementation is and what's next.
