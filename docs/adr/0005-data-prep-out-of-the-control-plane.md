# ADR 0005 — Data prep out of the control plane: the user prepares role-keyed inputs; the CP routes opaque blobs

- **Status:** Accepted — shipped (ROADMAP 9.2 / #74; the CP routes opaque role→blob maps, prep lives in `tools/`).
- **Date:** 2026-06-19
- **Affects spec:** §3.3 (the deliberately-unintelligent control plane), §3.4 (the control-plane
  service + its data plane), §3.5/§3.6 (role isolation — the answer key never reaches the agent),
  §12 (the feature-engineering domain's split). **Resolves the open spec-question in ROADMAP 9.2 /
  issue #74.** Builds on **[ADR 0003](0003-control-plane-and-ephemeral-worker-provisioning.md)** (the
  CP routes inputs to ephemeral workers) and **[ADR 0004](0004-long-lived-configurable-control-plane.md)**
  (the long-lived service + data plane; ADR 0004 itself flags the CP-side split as "the riskiest new
  logic").
- **Supersedes:** nothing. It *removes* logic from the control plane and tightens the data-plane
  contract; the kernel is untouched.

## Context

ROADMAP Phase 9 ("the dumb control plane") is extracting the last domain code out of the
control-plane image. **9.1 ✅** moved the verifier (gates + `kaggle`) into a sibling container. The
remaining residual is **9.2 / #74: task-specific data preparation still runs in the CP process.**

Concretely, today the `fe-kaggle` catalog builder prepares the dataset *inside the control plane*:

```python
# composition/fe_kaggle.py:329-335  (build_fe_kaggle_task)
raw       = cp.store.get_object(request.data.data_ref)
real_test = cp.store.get_object(request.data.test_ref)
sub       = subsample(raw, per_class=request.data.per_class, target=request.data.target)
split     = stratified_split(sub, target=request.data.target,
                             id_column=request.data.id_column,
                             reserved_fraction=request.data.reserved_fraction)
```

`subsample` + `stratified_split` (`composition/dataset.py`) interpret dataset semantics — they read
the `target` column, the `id_column`, stratify, and **derive the answer key** (`reserved_labels`,
the held-out true targets). That logic ships in the CP image and runs in the CP process. Two
consequences, both at odds with the founding concept (§3.3: the CP is *deliberately unintelligent*):

1. **The CP understands the domain.** `DataRequest` carries `target` / `id_column` / `per_class` /
   `reserved_fraction` (`task_request.py:165-170`) — knobs that only mean something to a CSV
   classification task. A new task type with a different data shape can't be added without teaching
   the CP, i.e. **a CP rebuild** — the exact thing Phase 9 exists to kill.
2. **The answer key is born in the CP.** The reserved labels are *derived* CP-side and then routed to
   the verifier. Isolation (the agent must never see them, §3.5) is preserved by **careful CP-side
   carving** rather than **by construction** — a fragile place for the one invariant we most need to
   be structural.

The data plane that feeds this is also domain-shaped: `create_task(data=…, test_data=…)` ingests
exactly **two positional blobs** (`control_service.py:147-167`) — "train" and "test" — which only the
FE task's mental model explains.

### The open question 9.2 named

ROADMAP 9.2 left the *prep boundary* open: **where should prep live** — (a) in the verifier
container, (b) in a dedicated task-package build step, or (c) the user's responsibility entirely? It
overlaps #3 (the data plane) and #64 (making the domain concept optional). This ADR settles it.

## Decision

**Data preparation is the user's responsibility, performed outside Verity. The control plane routes
opaque, role-keyed named blobs and interprets none of them.**

Three parts:

### 1. Prep is user-side (resolves the open question → option (c))

The user (or their task-package tooling) prepares the per-role inputs *before* handing them to
Verity. Verity does not split, subsample, stratify, or derive an answer key — it never reads a
`target` or an `id_column`. This is the strongest form of "dumb CP": prep that requires domain
knowledge lives with whoever has the domain knowledge — the user.

We reject the alternatives:

- **(a) Prep in the verifier container** — better than CP-side, but still bakes one task's
  prep into a service image and still requires the *raw* dataset to transit the CP and be split
  somewhere inside Verity. It moves the domain code without removing it from the trusted path, and it
  couples "what the agent sees" to a verifier-side computation. The agent's inputs should not depend
  on the verifier running.
- **(b) A dedicated task-package build step inside Verity** — this is just option (c) with Verity
  hosting the tool. It re-imports the domain code we are trying to expel. If a user wants a reusable
  prep tool, it ships *with the task package*, not in the CP/verifier images.

What we keep from the rejected options: the prep *logic itself* is still valuable and stays in the
repo as a **user-side helper, not on any service's import path** — see Consequences.

### 2. The routing contract: role → {filename: blob}

The data plane carries a mapping from **role** (`agent`, `verifier`) to a set of **named files**, each
an opaque content-addressed blob. The CP stores each blob and routes the role's file set, by name,
into that role's worker — nothing more. Files in different roles may share a name (`train.csv`); the
role namespace disambiguates, which is why the example ships them in per-role subdirectories.

This replaces the positional `data_ref` / `test_ref` + split-knob shape. The CP no longer knows which
file is "train", which is "test", or that an answer key exists — those are just entries in the
verifier role's set that the verifier (which *does* have the domain knowledge) interprets.

### 3. The answer key is isolated by construction

Because the reserved labels arrive as an ordinary file in the **`verifier` role's** set and the CP
routes role sets verbatim, the answer key **structurally cannot** reach the `agent` role — there is no
CP-side derivation step that could leak it, and no code path that copies a verifier-role blob to the
agent role. The §3.5 isolation invariant becomes a property of the routing contract, not of careful
carving. The existing byte-provenance tests (the answer key absent from every agent worker) re-point
from "the split withheld it" to "routing never placed it there."

## The worked example (this change ships it)

`results/prototyping_datasci_test/` now carries the prepared, role-keyed inputs (generated once from the
stellar `train.csv`/`test.csv` with the *same* `subsample(per_class=300)` → `stratified_split(
reserved_fraction=0.5)` the CP did, run user-side):

```
results/prototyping_datasci_test/
  agent/
    train.csv            # labeled, hold-out rows removed — what the agent trains on
    test.csv             # the real Kaggle test set, unlabeled
  verifier/
    train.csv            # the full labeled subsample
    test.csv             # the real Kaggle test set
    holdout.csv          # the reserved rows, target column dropped (the cheap proxy's eval set)
    holdout_labels.csv   # the answer key (id,class) — verifier-role only, never routed to the agent
```

Integrity: `agent/train.csv` (450 rows) + `verifier/holdout.csv` (450) = `verifier/train.csv` (900);
the agent's train keeps `class`, the hold-out drops it, the labels file is the held-out `id→class`.
The two `test.csv` copies are byte-identical (each role is a self-contained bundle the CP routes
without knowing they overlap).

## Consequences

**Removed from the control-plane image / process:**

- `composition/dataset.py`'s `subsample` / `stratified_split` leave the CP import path. The code is
  preserved as a **user-side prep helper** (a `tools/`-level script / task-package utility) — kept,
  tested, but imported by *no* service. A guard test asserts `composition.fe_kaggle` (and the CP
  image) imports neither.
- `DataRequest` loses `target` / `id_column` / `per_class` / `reserved_fraction`. The CP holds no
  dataset semantics.

**Changed contracts:**

- `DataRequest` (or its successor) carries a role→named-files reference map instead of
  `data_ref`/`test_ref`. `with_data_ref`/`with_test_ref` give way to a role-file stamping API.
- `ControlService.create_task` / the `ingest` + `POST /objects` data plane accept **named,
  role-tagged** uploads rather than two positional blobs. (This is the natural multi-file
  generalization #3 anticipates; the over-the-wire file API stays the #3 deferral, but the *shape*
  lands here.)
- `build_fe_kaggle_task` stops splitting: it reads the routed verifier-role blobs (`train`,
  `holdout`, `holdout_labels`, `test`) and assembles the `VerifierSetup` from them; it routes the
  agent-role blobs into `static_contents`. No `subsample`/`stratified_split` call remains.

**Held invariants (tests re-pointed, not weakened):** byte-provenance (answer key absent from every
agent worker) — now by routing, not carving; per-task store isolation; verifier opacity; the AST
genericity guard (`verity.control_plane` imports no domain) — strengthened, since even the
*composition* CP-image path now interprets no dataset.

**Migration of the existing FE-kaggle task config:** `task.json` / `task.smoke.json` drop the `data`
split knobs and instead point at the role directories (or the client ingests the role files by name).
The skill (`verity-run-task`) and `PROTOCOL.md` update to the prepare-then-ingest flow.

**Deferred / out of scope (unchanged):** the over-the-wire streaming file API and removal of the
shared exchange volume (#3 proper); a generic, domain-agnostic prep SDK for *other* task types (#64);
the plugin loader that lets a task type register without a rebuild (ROADMAP 9.3 — the piece that makes
9.1+9.2 fully "no CP rebuild").

## Litmus test (the 9.2 done-line)

No CP-image-resident code interprets dataset/domain semantics — no split, no `target`/`id_column`,
no answer-key derivation. A new task type supplies its own (user-prepared) role inputs and the CP
routes them unchanged. The answer-key isolation invariant holds **by construction at the routing
boundary**.
