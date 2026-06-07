"""Shared test helpers for the kernel-contract tests (ROADMAP Phase 1.1).

Deterministic clocks/ids and scripted commit-path collaborators (shape validator, gate
binding resolver, gate runner) so the store, lifecycle, and commit path can be exercised in
isolation before the registries (§8) and verifier dispatch (§3.6) exist.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

from verity.control_plane.commit import (
    GateBinding,
    GateSpec,
    GateVerdict,
    ShapeError,
)
from verity.control_plane.store import (
    Artifact,
    ArtifactStatus,
    Operation,
    OperationStatus,
    Payload,
    SqliteStore,
    VerdictKind,
)

VERIFIER = "verifier"  # a gate identity distinct from the proposer
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


def gate(
    name: str = "g",
    *,
    identity: str = VERIFIER,
    is_hard: bool = False,
    requires_human: bool = False,
) -> GateSpec:
    return GateSpec(name=name, identity=identity, is_hard=is_hard, requires_human=requires_human)


def binding(artifact_type: str, *gates: GateSpec) -> GateBinding:
    return GateBinding(artifact_type=artifact_type, gates=tuple(gates))


def resolver(*bindings: GateBinding):
    """A :class:`GateBindingResolver` over a fixed map; unknown types resolve to ``None``."""
    table = {b.artifact_type: b for b in bindings}

    def resolve(artifact_type: str) -> GateBinding | None:
        return table.get(artifact_type)

    return resolve


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


def runner(verdicts: dict[str, GateVerdict | None]):
    """A :class:`GateRunner` mapping gate name → verdict (``None`` = cannot auto-resolve)."""

    def run(g: GateSpec, _artifact: Artifact) -> GateVerdict | None:
        return verdicts.get(g.name, accept())

    return run
