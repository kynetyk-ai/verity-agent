"""Shared test helpers for the kernel-contract tests (ROADMAP Phase 1.1; ADR 0001).

Deterministic clocks/ids, a proposer, and scripted commit-path collaborators for the **opaque
verifier** model: a coverage resolver, a bundle dispatcher, and verdict-bundle builders, so the
store, lifecycle, and commit path can be exercised in isolation.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    GateDecision,
    GateVerdict,
    Operation,
    OperationStatus,
    Payload,
    VerdictBundle,
    VerdictKind,
)
from verity.control_plane.commit import ShapeError
from verity.control_plane.independence import StoreInput
from verity.control_plane.store import SqliteStore

VERIFIER = "verifier"  # a verifier identity distinct from the proposer
AGENT = "agent"  # the default proposer identity


def make_store() -> SqliteStore:
    """An in-memory store with a deterministic clock and sequential ids."""
    counter = itertools.count(1)
    ticks = itertools.count(1)
    return SqliteStore(
        clock=lambda: f"t{next(ticks):04d}",
        new_id=lambda: f"id-{next(counter):04d}",
    )


def propose(
    store: SqliteStore,
    *,
    artifact_id: str,
    artifact_type: str = "Thing",
    created_by: str = AGENT,
    parents: Iterable[str] = (),
    op_name: str = "produce",
    is_root: bool = False,
    payload: Payload | None = None,
) -> Artifact:
    """Record a fresh ``proposed`` artifact plus its producing operation."""
    op = Operation(
        op_id=f"op-{artifact_id}",
        op_name=op_name,
        parents=tuple(parents),
        output_id=artifact_id,
        status=OperationStatus.SUCCESS,
        created_at="",
    )
    artifact = Artifact(
        id=artifact_id,
        type=artifact_type,
        payload=payload if payload is not None else {"v": artifact_id},
        status=ArtifactStatus.PROPOSED,
        created_by=created_by,
        created_at="",
        is_root=is_root,
    )
    return store.propose(artifact, op)


# ------------------------------------------------------------------- commit collaborators


def coverage(*gated: tuple[str, frozenset[StoreInput]] | str):
    """A :class:`CoverageResolver`: gated types resolve to their inputs, others to ``None``.

    Pass a bare type name for an empty-slice gated type, or ``(type, declared_inputs)`` to grant it.
    """
    table: dict[str, frozenset[StoreInput]] = {}
    for entry in gated:
        if isinstance(entry, str):
            table[entry] = frozenset()
        else:
            table[entry[0]] = entry[1]

    def resolve(artifact_type: str) -> frozenset[StoreInput] | None:
        return table.get(artifact_type)

    return resolve


def decision(
    gate: str = "g",
    kind: VerdictKind = VerdictKind.ACCEPT,
    rationale: str = "ok",
    *,
    defects: tuple[str, ...] | None = None,
    score: float | None = None,
) -> GateDecision:
    return GateDecision(gate=gate, kind=kind, rationale=rationale, defects=defects, score=score)


def bundle(
    status: ArtifactStatus,
    *decisions: GateDecision,
    supersedes: str | None = None,
) -> VerdictBundle:
    return VerdictBundle(status=status, decisions=tuple(decisions), supersedes=supersedes)


def returns(result: VerdictBundle):
    """A :class:`BundleDispatch` that returns a fixed bundle, recording the artifacts it judged."""
    seen: list[Artifact] = []

    def dispatch(artifact: Artifact) -> VerdictBundle:
        seen.append(artifact)
        return result

    dispatch.seen = seen  # type: ignore[attr-defined]
    return dispatch


def shape_ok():
    """A :class:`ShapeValidator` that passes everything."""

    def validate(_artifact: Artifact) -> ShapeError | None:
        return None

    return validate


def shape_fail(message: str = "malformed"):
    """A :class:`ShapeValidator` that fails everything with ``message``."""

    def validate(_artifact: Artifact) -> ShapeError | None:
        return ShapeError(message)

    return validate


def accept(rationale: str = "ok", score: float | None = None) -> GateVerdict:
    return GateVerdict(VerdictKind.ACCEPT, rationale, score=score)


def reject(rationale: str = "no") -> GateVerdict:
    return GateVerdict(VerdictKind.REJECT, rationale)


def refine(*defects: str, rationale: str = "fixable") -> GateVerdict:
    return GateVerdict(VerdictKind.REFINE, rationale, defects=tuple(defects) or ("defect",))
