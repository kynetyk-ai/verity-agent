"""Commit-path tests for the opaque-verifier model (spec §7; ADR 0001) — ROADMAP Phase 1.1/refactor.

The commit path takes one verdict bundle and records + enforces: coverage (no implicit accept),
proposer ≠ gate, decision recording, transition legality, bundle self-consistency, and supersession.
"""

from __future__ import annotations

import pytest

from tests.helpers import AGENT, VERIFIER, bundle, coverage, decision, make_store, propose, returns
from verity.contracts import ArtifactStatus, VerdictKind
from verity.control_plane.commit import (
    BundleInconsistent,
    CommitError,
    CommitOutcome,
    NoImplicitAccept,
    ProposerIsGate,
    run_commit,
)
from verity.control_plane.store import SqliteStore

T = "Thing"


def _commit(store, artifact_id, *, result, gated=(T,), identity=VERIFIER, refine_exhausted=False):
    return run_commit(
        artifact_id,
        store=store,
        sink=store,
        resolve_coverage=coverage(*gated),
        dispatch=returns(result),
        verifier_identity=identity,
        refine_exhausted=refine_exhausted,
    )


# --------------------------------------------------------------------------- misuse


def test_unknown_artifact_is_an_error() -> None:
    store = make_store()
    with pytest.raises(CommitError):
        _commit(store, "nope", result=bundle(ArtifactStatus.ACCEPTED))


def test_non_proposed_artifact_is_an_error() -> None:
    store = make_store()
    propose(store, artifact_id="a", artifact_type=T, is_root=True)
    store.set_status("a", ArtifactStatus.ACCEPTED)
    with pytest.raises(CommitError):
        _commit(store, "a", result=bundle(ArtifactStatus.ACCEPTED))


# --------------------------------------------------------------------------- invariants


def test_uncovered_type_is_no_implicit_accept() -> None:
    store = make_store()
    propose(store, artifact_id="a", artifact_type="Orphan", is_root=True)
    with pytest.raises(NoImplicitAccept):
        _commit(store, "a", result=bundle(ArtifactStatus.ACCEPTED), gated=(T,))


def test_verifier_sharing_proposer_identity_is_refused() -> None:
    store = make_store()
    propose(store, artifact_id="a", artifact_type=T, created_by="same", is_root=True)
    with pytest.raises(ProposerIsGate):
        _commit(store, "a", result=bundle(ArtifactStatus.ACCEPTED), identity="same")


# --------------------------------------------------------------------------- outcomes


def test_accept_sets_accepted_and_records_decisions() -> None:
    store = make_store()
    propose(store, artifact_id="a", artifact_type=T, is_root=True)
    result = _commit(
        store, "a", result=bundle(ArtifactStatus.ACCEPTED, decision("worth-keeping", score=0.9))
    )
    assert result.outcome is CommitOutcome.ACCEPTED
    assert store.get_artifact("a").status is ArtifactStatus.ACCEPTED
    assert [d.gate for d in store.decisions_for("a")] == ["worth-keeping"]


def test_tentative_rests_at_tentative() -> None:
    store = make_store()
    propose(store, artifact_id="a", artifact_type=T, is_root=True)
    result = _commit(store, "a", result=bundle(ArtifactStatus.TENTATIVE, decision("cheap")))
    assert result.outcome is CommitOutcome.TENTATIVE
    assert store.get_artifact("a").status is ArtifactStatus.TENTATIVE


def test_reject_is_recorded_and_terminal() -> None:
    store = make_store()
    propose(store, artifact_id="a", artifact_type=T, is_root=True)
    result = _commit(
        store, "a", result=bundle(ArtifactStatus.REJECTED, decision("g", VerdictKind.REJECT, "no"))
    )
    assert result.outcome is CommitOutcome.REJECTED
    assert store.get_artifact("a").status is ArtifactStatus.REJECTED


def test_refine_carries_defects() -> None:
    store = make_store()
    propose(store, artifact_id="a", artifact_type=T, is_root=True)
    result = _commit(
        store,
        "a",
        result=bundle(
            ArtifactStatus.REVISED, decision("g", VerdictKind.REFINE, "fix", defects=("tone",))
        ),
    )
    assert result.outcome is CommitOutcome.REVISED
    assert result.defects == ("tone",)
    assert store.get_artifact("a").status is ArtifactStatus.REVISED


def test_refine_exhausted_terminates_in_rejected() -> None:
    # §3.4 refine_cap: when the lineage's refine budget is spent, a refine verdict is recorded but
    # the control plane terminates the artifact in 'rejected' rather than sending it back again.
    store = make_store()
    propose(store, artifact_id="a", artifact_type=T, is_root=True)
    result = _commit(
        store,
        "a",
        result=bundle(
            ArtifactStatus.REVISED, decision("g", VerdictKind.REFINE, "fix", defects=("tone",))
        ),
        refine_exhausted=True,
    )
    assert result.outcome is CommitOutcome.REJECTED
    assert result.defects == ("tone",)  # the refine defect is preserved as the rejection reason
    assert store.get_artifact("a").status is ArtifactStatus.REJECTED


# --------------------------------------------------------------------------- supersession


def test_accept_supersedes_the_named_incumbent() -> None:
    store = make_store()
    propose(store, artifact_id="old", artifact_type=T, is_root=True)
    store.set_status("old", ArtifactStatus.ACCEPTED)
    propose(store, artifact_id="new", artifact_type=T, is_root=True)
    result = _commit(store, "new", result=bundle(ArtifactStatus.ACCEPTED, supersedes="old"))
    assert result.outcome is CommitOutcome.ACCEPTED
    assert store.get_artifact("new").status is ArtifactStatus.ACCEPTED
    old = store.get_artifact("old")
    assert old.status is ArtifactStatus.SUPERSEDED and old.superseded_by == "new"


def test_superseding_an_unknown_incumbent_errors() -> None:
    store = make_store()
    propose(store, artifact_id="a", artifact_type=T, is_root=True)
    with pytest.raises(CommitError):
        _commit(store, "a", result=bundle(ArtifactStatus.ACCEPTED, supersedes="ghost"))


# --------------------------------------------------------------------------- coherence


def test_a_reject_decision_under_an_accept_status_is_inconsistent() -> None:
    store = make_store()
    propose(store, artifact_id="a", artifact_type=T, is_root=True)
    inconsistent = bundle(ArtifactStatus.ACCEPTED, decision("g", VerdictKind.REJECT, "no"))
    with pytest.raises(BundleInconsistent):
        _commit(store, "a", result=inconsistent)


def test_non_root_artifact_keeps_provenance_to_accept() -> None:
    store = make_store()
    propose(store, artifact_id="root", artifact_type=T, is_root=True)
    propose(store, artifact_id="child", artifact_type=T, parents=("root",))
    result = _commit(store, "child", result=bundle(ArtifactStatus.ACCEPTED))
    assert result.outcome is CommitOutcome.ACCEPTED
    assert any(op.op_name == "produce" for op in store.operations_into("child"))


def test_proposer_constant_distinct_from_verifier() -> None:
    assert AGENT != VERIFIER


# --------------------------------------------------------------------------- atomicity (5.1)


class _FailsOnSetStatus(SqliteStore):
    """A store whose ``set_status`` always raises — simulates a write failing part-way through."""

    def set_status(self, artifact_id: str, status: ArtifactStatus) -> None:
        raise RuntimeError("simulated store error mid-commit")


def test_a_mid_commit_failure_rolls_back_the_decision_rows() -> None:
    # A commit records its decisions and sets the terminal status as one unit (ROADMAP 5.1): if the
    # status write fails after the decisions are recorded, the decisions must roll back too — never
    # leave decision rows on a still-'proposed' artifact.
    store = _FailsOnSetStatus()
    propose(store, artifact_id="a", artifact_type=T, is_root=True)
    with pytest.raises(RuntimeError, match="simulated store error"):
        _commit(store, "a", result=bundle(ArtifactStatus.ACCEPTED, decision("worth-keeping")))

    assert store.decisions_for("a") == []  # the decision row rolled back with the failed status
    assert store.get_artifact("a").status is ArtifactStatus.PROPOSED  # untouched
