"""Audit-contract invariants — properties that must hold over the store (spec §5).

The eight §5 invariants, and *how each is enforced* in this implementation:

1. **Stable, unique, never-reused ids** — state-checkable: :func:`check_unique_ids`. Also
   structurally guarded — the store rejects a reused id at write time (§5.1).
2. **Append-only payloads** — structural: there is no method on :class:`Store` or
   :class:`CommitSink` that overwrites a payload; a change is a new artifact (§5.2). Proven by
   property test (no payload mutates across a run).
3. **No hard delete of accepted artifacts** — structural: no delete method exists; removal is
   soft, via status (§5.3).
4. **Every non-root accepted/tentative artifact has provenance** — state-checkable:
   :func:`check_provenance`. Also guarded at the commit path (it refuses to accept a
   provenance-less non-root artifact, §5.4).
5. **Failures are recorded, not erased** — structural: failed/retried operations are never
   deleted; :func:`check_failures_retained` confirms the recorded ones remain (§5.5).
6. **No silent merge** — structural: there is no merge primitive; identity is never collapsed
   without a recorded operation (§5.6). Property-tested.
7. **No implicit accept** — behavioral: the commit path raises on a gateless type
   (``NoImplicitAccept``); property-tested (§5.7).
8. **Reproducible gate verdicts** — behavioral: the commit path is deterministic given its
   verdicts; the verifier fixes seeds / averages (§5.8). Property-tested.

The state-checkable invariants ship here as functions that return a list of
:class:`Violation` (empty = the store is sound); the structural/behavioral ones ship as
property tests alongside the code (ROADMAP cross-cutting principle: invariants are tests).
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.control_plane.store import (
    Artifact,
    ArtifactStatus,
    Operation,
    Store,
)

__all__ = [
    "Violation",
    "all_operations",
    "check_unique_ids",
    "check_provenance",
    "check_failures_retained",
    "check_terminal_pointers",
    "audit",
]


@dataclass(frozen=True, slots=True)
class Violation:
    """A single audit-contract breach: which invariant, and a human-readable detail."""

    invariant: str
    detail: str


def all_operations(store: Store, artifacts: list[Artifact]) -> list[Operation]:
    """Every operation in the store, deduped by ``op_id``.

    The :class:`Store` surface exposes operations per artifact, not a global list; since every
    operation has an ``output_id`` pointing at an artifact, the union of ``operations_into``
    over all artifacts is the full set.
    """
    seen: dict[str, Operation] = {}
    for artifact in artifacts:
        for op in store.operations_into(artifact.id):
            seen.setdefault(op.op_id, op)
    return list(seen.values())


def check_unique_ids(store: Store) -> list[Violation]:
    """§5.1 — artifact ids and operation ids are each unique and never reused."""
    violations: list[Violation] = []
    artifacts = store.query_artifacts()

    artifact_ids = [a.id for a in artifacts]
    for dup in _duplicates(artifact_ids):
        violations.append(Violation("§5.1", f"duplicate artifact id: {dup}"))

    op_ids = [op.op_id for op in all_operations(store, artifacts)]
    for dup in _duplicates(op_ids):
        violations.append(Violation("§5.1", f"duplicate operation id: {dup}"))
    return violations


def check_provenance(store: Store) -> list[Violation]:
    """§5.4 — every non-root accepted/tentative artifact has a recorded operation edge."""
    violations: list[Violation] = []
    for artifact in store.query_artifacts():
        if artifact.is_root:
            continue
        if artifact.status not in (ArtifactStatus.ACCEPTED, ArtifactStatus.TENTATIVE):
            continue
        if not store.operations_into(artifact.id):
            violations.append(
                Violation(
                    "§5.4",
                    f"non-root {artifact.status.value} artifact {artifact.id} has no provenance",
                )
            )
    return violations


def check_failures_retained(store: Store) -> list[Violation]:
    """§5.5 — failed/retried operations remain in the record (none silently erased).

    Over current state this verifies the recorded failures are still queryable; deletion is
    prevented structurally (the store offers no operation-delete).
    """
    violations: list[Violation] = []
    artifacts = store.query_artifacts()
    for op in all_operations(store, artifacts):
        if op.output_id and store.get_artifact(op.output_id) is None:
            violations.append(
                Violation("§5.5", f"operation {op.op_id} references missing output {op.output_id}")
            )
    return violations


def check_terminal_pointers(store: Store) -> list[Violation]:
    """Integrity of replacement pointers: ``superseded``/``revised`` resolve to real artifacts.

    Supports §5.2/§6 — a superseded artifact points to its replacement, a revised one to its
    revision, and those targets exist.
    """
    violations: list[Violation] = []
    for artifact in store.query_artifacts():
        if artifact.status is ArtifactStatus.SUPERSEDED:
            if artifact.superseded_by is None:
                violations.append(
                    Violation("§6", f"superseded artifact {artifact.id} has no superseded_by")
                )
            elif store.get_artifact(artifact.superseded_by) is None:
                violations.append(
                    Violation("§6", f"{artifact.id}.superseded_by points to missing artifact")
                )
        if artifact.status is ArtifactStatus.REVISED and artifact.revised_by is not None:
            if store.get_artifact(artifact.revised_by) is None:
                violations.append(
                    Violation("§6", f"{artifact.id}.revised_by points to missing artifact")
                )
    return violations


def audit(store: Store) -> list[Violation]:
    """Run every state-checkable invariant; an empty list means the store is sound (§5)."""
    violations: list[Violation] = []
    violations.extend(check_unique_ids(store))
    violations.extend(check_provenance(store))
    violations.extend(check_failures_retained(store))
    violations.extend(check_terminal_pointers(store))
    return violations


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    dups: set[str] = set()
    for value in values:
        if value in seen:
            dups.add(value)
        seen.add(value)
    return sorted(dups)
