"""The commit path — the single privileged route from a proposal to durable state (spec §7).

Under **ADR 0001** the verifier is opaque: the control plane does not sequence gates or parse the
proposal. Given a ``proposed`` artifact the commit path:

1. **No implicit accept (§5.7).** The type must be *covered* — registered as gated — else fail.
2. **Proposer ≠ gate (§7.2).** The verifier's identity must differ from the proposer's; structural.
3. **One handoff (§3.6, amended).** Dispatch the proposal (its declared slice + objects are bound
   by the injected ``dispatch`` collaborator) and receive a :class:`VerdictBundle` — the terminal
   status the verifier recommends plus the per-check decisions.
4. **Record + enforce.** Write a ``decisions`` row for every decision in the bundle; validate the
   bundle is self-consistent and its status is a legal transition (§6); then set the status,
   superseding atomically if the bundle names an incumbent (§7.5a). Ensure provenance on accept.

The control plane never decides *what* to run or *whether the artifact is good* — only that it is
covered, that the verdict it was handed is coherent and legal, and that the write is recorded. Shape
validation (§7.0) is a pre-gate filter performed at intake (it decides whether to dispatch at all),
not here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from verity.contracts.model import Artifact, ArtifactStatus, VerdictBundle, VerdictKind
from verity.control_plane import lifecycle
from verity.control_plane.independence import StoreInput
from verity.control_plane.store import (
    Clock,
    CommitSink,
    Decision,
    Store,
    default_clock,
)
from verity.logging import get_logger

__all__ = [
    "ShapeError",
    "ShapeValidator",
    "CoverageResolver",
    "BundleDispatch",
    "CommitOutcome",
    "CommitResult",
    "CommitError",
    "NoImplicitAccept",
    "ProposerIsGate",
    "BundleInconsistent",
    "run_commit",
]

log = get_logger("verity.control_plane.commit")


# ----------------------------------------------------------------- shape (§7.0, pre-gate)


@dataclass(frozen=True, slots=True)
class ShapeError:
    """A correctable, *unrecorded* malformed-proposal response (spec §3.4, §7.0)."""

    message: str


class ShapeValidator(Protocol):
    """Checks a proposal's *composition* against its type's shape spec (spec §7.0).

    A presence/type check only — never the quality of the contents (ADR 0001). Performed at intake;
    a malformed proposal never reaches the verifier.
    """

    def __call__(self, artifact: Artifact, /) -> ShapeError | None: ...


# ----------------------------------------------------------------- injected collaborators

# Resolves a type's declared store-inputs, or ``None`` if the type is not gated (→ no implicit
# accept, §5.7). The control plane's whole knowledge of "gates" is this coverage + slice record.
CoverageResolver = Callable[[str], "frozenset[StoreInput] | None"]

# The single verifier handoff: hand over the proposal (its declared slice + objects are bound by the
# closure) and receive the verdict bundle (ADR 0001). Backed by the async verifier dispatch (§3.6).
BundleDispatch = Callable[[Artifact], VerdictBundle]


# --------------------------------------------------------------------------- results


class CommitOutcome(StrEnum):
    """The terminal outcome of one commit attempt (the status the verifier's bundle resolved to)."""

    REJECTED = "rejected"
    TENTATIVE = "tentative"
    ACCEPTED = "accepted"
    REVISED = "revised"


@dataclass(frozen=True, slots=True)
class CommitResult:
    """What a commit attempt produced (spec §7). ``status`` is the artifact's new status.

    ``decisions`` are the rows written (one per check the bundle reported); ``defects`` is the
    localized defect list on a ``REVISED`` outcome (§7.5b).
    """

    outcome: CommitOutcome
    artifact_id: str
    status: ArtifactStatus
    defects: tuple[str, ...] | None = None
    decisions: tuple[Decision, ...] = ()


class CommitError(RuntimeError):
    """A misuse of the commit path (unknown artifact, not ``proposed``)."""


class NoImplicitAccept(CommitError):
    """A type with no declared gate cannot be committed (spec §5.7, §7.1)."""

    def __init__(self, artifact_type: str) -> None:
        super().__init__(
            f"type {artifact_type!r} is not covered by a gate: committing it is an error, not a "
            f"default-accept (no implicit accept, §5.7)"
        )
        self.artifact_type = artifact_type


class ProposerIsGate(CommitError):
    """The verifier shares the proposer's identity — the Builder/Breaker rule (spec §6, §7.2)."""

    def __init__(self, identity: str) -> None:
        super().__init__(
            f"the verifier's identity equals the proposer's ({identity!r}): the proposer is never "
            f"its own gate (§6, §7.2)"
        )
        self.identity = identity


class BundleInconsistent(CommitError):
    """The verifier's bundle contradicts itself (e.g. a reject decision under an accept status)."""


# ------------------------------------------------------------------- the commit path


def run_commit(
    artifact_id: str,
    *,
    store: Store,
    sink: CommitSink,
    resolve_coverage: CoverageResolver,
    dispatch: BundleDispatch,
    verifier_identity: str,
    clock: Clock = default_clock,
) -> CommitResult:
    """Run the §7 commit protocol on a ``proposed`` artifact and return the outcome (ADR 0001).

    All status writes go through :class:`CommitSink`, gated by :func:`lifecycle.assert_transition`,
    so no illegal edge can be written even by this privileged path — or by the verifier's bundle.
    """
    artifact = store.get_artifact(artifact_id)
    if artifact is None:
        raise CommitError(f"unknown artifact: {artifact_id}")
    if artifact.status is not ArtifactStatus.PROPOSED:
        raise CommitError(
            f"commit operates on a 'proposed' artifact, got '{artifact.status.value}'"
        )

    # 1 — no implicit accept (§5.7): the type must be covered by a gate.
    if resolve_coverage(artifact.type) is None:
        raise NoImplicitAccept(artifact.type)

    # 2 — proposer ≠ gate (§7.2): the verifier's identity differs from the proposer's.
    if verifier_identity == artifact.created_by:
        raise ProposerIsGate(verifier_identity)

    # 3 — one opaque handoff (§3.6, amended): the verifier returns the bundle.
    bundle = dispatch(artifact)
    _assert_consistent(bundle)

    # 4 — record every decision the bundle reports (§4.1, §7.3).
    decisions: list[Decision] = []
    for ruling in bundle.decisions:
        decision = Decision(
            artifact_id=artifact.id,
            gate=ruling.gate,
            verdict=ruling.kind,
            rationale=ruling.rationale,
            defects=ruling.defects,
            score=ruling.score,
            created_at=clock(),
        )
        sink.record_decision(decision)
        decisions.append(decision)

    # — set the recommended status, after validating the transition is legal (§6).
    status = bundle.status
    lifecycle.assert_transition(ArtifactStatus.PROPOSED, status)

    if status is ArtifactStatus.REJECTED:
        sink.set_status(artifact.id, status)
        return _result(CommitOutcome.REJECTED, artifact.id, status, decisions)

    if status is ArtifactStatus.REVISED:
        sink.set_status(artifact.id, status)
        defects = _defects_from(bundle)
        return _result(CommitOutcome.REVISED, artifact.id, status, decisions, defects=defects)

    if status is ArtifactStatus.TENTATIVE:
        _ensure_provenance(store, artifact)
        sink.set_status(artifact.id, status)
        return _result(CommitOutcome.TENTATIVE, artifact.id, status, decisions)

    if status is ArtifactStatus.ACCEPTED:
        _ensure_provenance(store, artifact)
        _accept(artifact, bundle.supersedes, store=store, sink=sink)
        return _result(CommitOutcome.ACCEPTED, artifact.id, status, decisions)

    raise BundleInconsistent(f"verifier reported an unsupported terminal status: {status.value!r}")


def _assert_consistent(bundle: VerdictBundle) -> None:
    """The bundle's status must agree with its decisions (ADR 0001) — coherence, not judgement."""
    kinds = {d.kind for d in bundle.decisions}
    if VerdictKind.REJECT in kinds and bundle.status is not ArtifactStatus.REJECTED:
        raise BundleInconsistent(
            f"bundle carries a reject decision but recommends {bundle.status.value!r}"
        )
    if VerdictKind.REFINE in kinds and bundle.status is not ArtifactStatus.REVISED:
        raise BundleInconsistent(
            f"bundle carries a refine decision but recommends {bundle.status.value!r}"
        )


def _accept(
    artifact: Artifact, supersedes: str | None, *, store: Store, sink: CommitSink
) -> None:
    """Set ``accepted``, atomically superseding the named incumbent if any (§7.5a)."""
    if supersedes is None:
        sink.set_status(artifact.id, ArtifactStatus.ACCEPTED)
        return
    incumbent = store.get_artifact(supersedes)
    if incumbent is None:
        raise CommitError(f"bundle names an unknown incumbent to supersede: {supersedes!r}")
    lifecycle.assert_transition(incumbent.status, ArtifactStatus.SUPERSEDED)
    sink.accept_superseding(artifact.id, supersedes)
    log.info("accepted_superseding", artifact_id=artifact.id, superseded=supersedes)


def _defects_from(bundle: VerdictBundle) -> tuple[str, ...] | None:
    for ruling in bundle.decisions:
        if ruling.kind is VerdictKind.REFINE and ruling.defects:
            return ruling.defects
    return None


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
    outcome: CommitOutcome,
    artifact_id: str,
    status: ArtifactStatus,
    decisions: list[Decision],
    *,
    defects: tuple[str, ...] | None = None,
) -> CommitResult:
    return CommitResult(
        outcome=outcome,
        artifact_id=artifact_id,
        status=status,
        defects=defects,
        decisions=tuple(decisions),
    )
