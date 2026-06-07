"""The commit path — the single privileged route from proposal to durable state (spec §7).

Control-plane-orchestrated protocol: validate shape → resolve gate binding (no binding =
hard error) → enforce proposer ≠ gate → run the pipeline advisorily → record decisions →
advance status → resolve supersession / refine → ensure provenance. It is the only code
permitted to set status or write decisions; the verifier advises but never writes (§7).

The protocol depends on three injected collaborators so it can be validated in isolation
before the registries (§8) and the verifier dispatch (§3.6) exist:

* :class:`ShapeValidator` — checks the proposal is *composed of* the required parts (§7.0).
* :class:`GateBindingResolver` — returns the type's ordered gate pipeline, or ``None`` (the
  "no implicit accept" trigger, §5.7); later backed by the gate registry (§8.3).
* :class:`GateRunner` — renders one gate's verdict; later backed by verifier dispatch over a
  declared store-slice (§3.6). Returning ``None`` means the gate cannot auto-resolve (e.g. a
  ``requires_human`` gate), so the artifact rests at ``tentative`` rather than auto-accepting.

Acceptance always routes ``proposed → tentative → accepted`` (§6); see :mod:`.lifecycle`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from verity.contracts.model import Artifact, ArtifactStatus, GateVerdict, VerdictKind
from verity.control_plane import lifecycle
from verity.control_plane.store import (
    Clock,
    CommitSink,
    Decision,
    Store,
    default_clock,
)
from verity.logging import get_logger

__all__ = [
    "GateVerdict",
    "GateSpec",
    "GateBinding",
    "ShapeError",
    "ShapeValidator",
    "GateBindingResolver",
    "GateRunner",
    "CommitOutcome",
    "CommitResult",
    "CommitError",
    "NoImplicitAccept",
    "ProposerIsGate",
    "run_commit",
]

log = get_logger("verity.control_plane.commit")


# ----------------------------------------------------------------- collaborator types


# GateVerdict (the verifier's reply) is a cross-service value type — defined in
# verity.contracts.model, imported above, and re-exported here so the commit path's callers keep a
# stable surface. GateSpec / GateBinding below are control-plane-internal config (the binding), not
# part of the wire contract.


@dataclass(frozen=True, slots=True)
class GateSpec:
    """A single gate in a type's pipeline (spec §8.3).

    ``is_hard`` splits the pipeline: clearing the cheap (``is_hard=False``) gates earns
    ``tentative``; clearing the hard gates earns ``accepted`` (§6). ``identity`` is the gate's
    distinct identity, used to enforce proposer ≠ gate (§7.2). ``requires_human`` marks a gate
    that must not auto-resolve (§8.3); when its runner returns no verdict the artifact rests
    at ``tentative``.
    """

    name: str
    identity: str
    is_hard: bool = False
    requires_human: bool = False


@dataclass(frozen=True, slots=True)
class GateBinding:
    """A type's ordered gate pipeline (spec §8.3). ``gates`` is cheap-first, then hard."""

    artifact_type: str
    gates: tuple[GateSpec, ...]


@dataclass(frozen=True, slots=True)
class ShapeError:
    """A correctable, *unrecorded* malformed-proposal response (spec §3.4, §7.0)."""

    message: str


class ShapeValidator(Protocol):
    """Checks a proposal's *composition* against its type's shape spec (spec §7.0)."""

    def __call__(self, artifact: Artifact, /) -> ShapeError | None: ...


class GateBindingResolver(Protocol):
    """Resolves a type's gate pipeline; ``None`` ⇒ no implicit accept (spec §5.7, §8.3)."""

    def __call__(self, artifact_type: str, /) -> GateBinding | None: ...


class GateRunner(Protocol):
    """Renders a gate's verdict; ``None`` ⇒ cannot auto-resolve (spec §3.6, §8.3)."""

    def __call__(self, gate: GateSpec, artifact: Artifact, /) -> GateVerdict | None: ...


# --------------------------------------------------------------------------- results


class CommitOutcome(StrEnum):
    """The terminal outcome of one commit attempt."""

    SHAPE_ERROR = "shape_error"
    REJECTED = "rejected"
    TENTATIVE = "tentative"
    ACCEPTED = "accepted"
    REVISED = "revised"


@dataclass(frozen=True, slots=True)
class CommitResult:
    """What a commit attempt produced (spec §7).

    On ``SHAPE_ERROR`` nothing is recorded — no decision row, no trial-count increment (§7.0):
    ``status`` is ``None`` and ``shape_error`` carries the correction. Otherwise ``status`` is
    the artifact's new status, ``decisions`` are the rows written, and ``defects`` is the
    localized defect list on a ``REVISED`` outcome (§7.5b).
    """

    outcome: CommitOutcome
    artifact_id: str
    status: ArtifactStatus | None = None
    shape_error: ShapeError | None = None
    defects: tuple[str, ...] | None = None
    decisions: tuple[Decision, ...] = ()


class CommitError(RuntimeError):
    """A misuse of the commit path (unknown artifact, not ``proposed``)."""


class NoImplicitAccept(CommitError):
    """A type with no declared gate cannot be committed (spec §5.7, §7.1)."""

    def __init__(self, artifact_type: str) -> None:
        super().__init__(
            f"no gate binding for type {artifact_type!r}: committing a gateless type is an "
            f"error, not a default-accept (no implicit accept, §5.7)"
        )
        self.artifact_type = artifact_type


class ProposerIsGate(CommitError):
    """A gate's identity equals the proposer's — the Builder/Breaker rule (spec §6, §7.2)."""

    def __init__(self, identity: str, gate_name: str) -> None:
        super().__init__(
            f"gate {gate_name!r} has the same identity as the proposer ({identity!r}): the "
            f"proposer is never its own gate (§6, §7.2)"
        )
        self.identity = identity
        self.gate_name = gate_name


# ------------------------------------------------------------------- the commit path


@dataclass
class _CommitRun:
    """Mutable bookkeeping for a single commit walk (one artifact through the protocol)."""

    artifact: Artifact
    status: ArtifactStatus
    decisions: list[Decision] = field(default_factory=list)


def run_commit(
    artifact_id: str,
    *,
    store: Store,
    sink: CommitSink,
    resolve_binding: GateBindingResolver,
    validate_shape: ShapeValidator,
    run_gate: GateRunner,
    supersedes: str | None = None,
    clock: Clock = default_clock,
) -> CommitResult:
    """Run the §7 commit protocol on a ``proposed`` artifact and return the outcome.

    ``supersedes`` names an incumbent accepted artifact this proposal replaces (§7.5a); the
    supersession is applied atomically with acceptance. All status writes go through
    :class:`CommitSink`, gated by :func:`lifecycle.assert_transition` so no illegal edge can
    be written even by this privileged path.
    """
    artifact = store.get_artifact(artifact_id)
    if artifact is None:
        raise CommitError(f"unknown artifact: {artifact_id}")
    if artifact.status is not ArtifactStatus.PROPOSED:
        raise CommitError(
            f"commit operates on a 'proposed' artifact, got '{artifact.status.value}'"
        )

    # Step 0 — validate shape. A malformed proposal is corrected, not recorded (§7.0).
    shape_error = validate_shape(artifact)
    if shape_error is not None:
        log.info("shape_error", artifact_id=artifact_id, message=shape_error.message)
        return CommitResult(CommitOutcome.SHAPE_ERROR, artifact_id, shape_error=shape_error)

    # Step 1 — resolve the gate pipeline. No binding ⇒ no implicit accept (§5.7, §7.1).
    binding = resolve_binding(artifact.type)
    if binding is None or not binding.gates:
        raise NoImplicitAccept(artifact.type)

    # Step 2 — proposer ≠ gate (§6, §7.2).
    for gate in binding.gates:
        if gate.identity == artifact.created_by:
            raise ProposerIsGate(gate.identity, gate.name)

    run = _CommitRun(artifact=artifact, status=artifact.status)
    cheap = [g for g in binding.gates if not g.is_hard]
    hard = [g for g in binding.gates if g.is_hard]

    # Step 3 (cheap stage) — clearing the cheap gates earns 'tentative' (§6, §7.4).
    terminal = _run_stage(cheap, run, sink=sink, run_gate=run_gate, clock=clock)
    if terminal is not None:
        return terminal
    _transition(run, ArtifactStatus.TENTATIVE, sink)

    # Step 3 (hard stage) — clearing the hard gates earns 'accepted' (§6, §7.4).
    terminal = _run_stage(hard, run, sink=sink, run_gate=run_gate, clock=clock)
    if terminal is not None:
        return terminal
    if _deferred_to_human(hard, run):
        # A hard gate could not auto-resolve; the artifact rests at 'tentative' (§8.3).
        log.info("rests_tentative", artifact_id=artifact_id)
        return _result(CommitOutcome.TENTATIVE, run)

    # Step 5a / 6 — accept (atomically superseding an incumbent if named), with provenance.
    _ensure_provenance(store, run.artifact)
    if supersedes is not None:
        lifecycle.assert_transition(run.status, ArtifactStatus.SUPERSEDED)  # incumbent edge
        lifecycle.assert_transition(run.status, ArtifactStatus.ACCEPTED)
        sink.accept_superseding(run.artifact.id, supersedes)
        run.status = ArtifactStatus.ACCEPTED
        log.info("accepted_superseding", artifact_id=artifact_id, superseded=supersedes)
    else:
        _transition(run, ArtifactStatus.ACCEPTED, sink)
    return _result(CommitOutcome.ACCEPTED, run)


def _run_stage(
    gates: list[GateSpec],
    run: _CommitRun,
    *,
    sink: CommitSink,
    run_gate: GateRunner,
    clock: Clock,
) -> CommitResult | None:
    """Run a stage of gates in order. Returns a terminal result, or ``None`` to continue.

    On the first ``reject`` the artifact is set ``rejected`` and the walk stops; on a
    ``refine`` it is set ``revised`` with the gate's defects (§7.3, §7.5b). A ``None`` verdict
    (a gate that cannot auto-resolve) stops the stage without a terminal result, so the caller
    can let the artifact rest at ``tentative``.
    """
    for gate in gates:
        verdict = run_gate(gate, run.artifact)
        if verdict is None:
            return None  # cannot auto-resolve; stop the stage (rest at current status)
        decision = Decision(
            artifact_id=run.artifact.id,
            gate=gate.name,
            verdict=verdict.kind,
            rationale=verdict.rationale,
            defects=verdict.defects,
            score=verdict.score,
            created_at=clock(),
        )
        sink.record_decision(decision)
        run.decisions.append(decision)

        if verdict.kind is VerdictKind.REJECT:
            _transition(run, ArtifactStatus.REJECTED, sink)
            return _result(CommitOutcome.REJECTED, run)
        if verdict.kind is VerdictKind.REFINE:
            _transition(run, ArtifactStatus.REVISED, sink)
            return _result(CommitOutcome.REVISED, run, defects=verdict.defects)
    return None


def _deferred_to_human(hard: list[GateSpec], run: _CommitRun) -> bool:
    """True if a hard gate was not resolved into a decision (a ``requires_human`` deferral)."""
    ruled = {d.gate for d in run.decisions}
    return any(g.name not in ruled for g in hard)


def _transition(run: _CommitRun, to: ArtifactStatus, sink: CommitSink) -> None:
    lifecycle.assert_transition(run.status, to)
    sink.set_status(run.artifact.id, to)
    run.status = to


def _ensure_provenance(store: Store, artifact: Artifact) -> None:
    """Every non-root accepted/tentative artifact must have a recorded operation edge (§5.4)."""
    if artifact.is_root:
        return
    if not store.operations_into(artifact.id):
        raise CommitError(
            f"artifact {artifact.id} is non-root but has no provenance edge (§5.4); "
            f"cannot accept a provenance-less artifact"
        )


def _result(
    outcome: CommitOutcome, run: _CommitRun, *, defects: tuple[str, ...] | None = None
) -> CommitResult:
    return CommitResult(
        outcome=outcome,
        artifact_id=run.artifact.id,
        status=run.status,
        defects=defects,
        decisions=tuple(run.decisions),
    )
