"""Commit-path protocol tests (spec §7) — ROADMAP Phase 1.1.

Exercises every branch of the §7 protocol against scripted collaborators: shape-error,
no-implicit-accept, proposer≠gate, the cheap→tentative / hard→accepted walk, reject, refine,
supersession, and the requires-human rest-at-tentative case.
"""

from __future__ import annotations

import pytest

from tests.helpers import (
    AGENT,
    accept,
    binding,
    gate,
    make_store,
    propose,
    refine,
    reject,
    resolver,
    runner,
    shape_fail,
    shape_ok,
)
from verity.control_plane.commit import (
    CommitError,
    CommitOutcome,
    NoImplicitAccept,
    ProposerIsGate,
    run_commit,
)
from verity.control_plane.store import ArtifactStatus, VerdictKind


def test_cheap_only_pipeline_accepts_through_tentative() -> None:
    store = make_store()
    propose(store, artifact_id="a1", artifact_type="Thing")
    result = run_commit(
        "a1",
        store=store,
        sink=store,
        resolve_binding=resolver(binding("Thing", gate("cheap"))),
        validate_shape=shape_ok(),
        run_gate=runner({"cheap": accept()}),
    )
    assert result.outcome is CommitOutcome.ACCEPTED
    got = store.get_artifact("a1")
    assert got is not None and got.status is ArtifactStatus.ACCEPTED
    assert [d.gate for d in store.decisions_for("a1")] == ["cheap"]


def test_cheap_then_hard_pipeline_accepts() -> None:
    store = make_store()
    propose(store, artifact_id="a1", artifact_type="Thing")
    result = run_commit(
        "a1",
        store=store,
        sink=store,
        resolve_binding=resolver(
            binding("Thing", gate("cheap"), gate("hard", is_hard=True))
        ),
        validate_shape=shape_ok(),
        run_gate=runner({"cheap": accept(), "hard": accept(score=0.7)}),
    )
    assert result.outcome is CommitOutcome.ACCEPTED
    assert {d.gate for d in store.decisions_for("a1")} == {"cheap", "hard"}


def test_reject_at_cheap_gate_stops_and_records() -> None:
    store = make_store()
    propose(store, artifact_id="a1", artifact_type="Thing")
    result = run_commit(
        "a1",
        store=store,
        sink=store,
        resolve_binding=resolver(
            binding("Thing", gate("cheap"), gate("hard", is_hard=True))
        ),
        validate_shape=shape_ok(),
        run_gate=runner({"cheap": reject("bad"), "hard": accept()}),
    )
    assert result.outcome is CommitOutcome.REJECTED
    got = store.get_artifact("a1")
    assert got is not None and got.status is ArtifactStatus.REJECTED
    # the hard gate never ran — only the cheap decision is recorded
    assert [d.gate for d in store.decisions_for("a1")] == ["cheap"]
    assert [a.id for a in store.rejected_log()] == ["a1"]


def test_reject_at_hard_gate_rejects_from_tentative() -> None:
    store = make_store()
    propose(store, artifact_id="a1", artifact_type="Thing")
    result = run_commit(
        "a1",
        store=store,
        sink=store,
        resolve_binding=resolver(
            binding("Thing", gate("cheap"), gate("hard", is_hard=True))
        ),
        validate_shape=shape_ok(),
        run_gate=runner({"cheap": accept(), "hard": reject()}),
    )
    assert result.outcome is CommitOutcome.REJECTED
    assert {d.gate for d in store.decisions_for("a1")} == {"cheap", "hard"}


def test_refine_marks_revised_with_defects() -> None:
    store = make_store()
    propose(store, artifact_id="a1", artifact_type="Thing")
    result = run_commit(
        "a1",
        store=store,
        sink=store,
        resolve_binding=resolver(binding("Thing", gate("cheap"))),
        validate_shape=shape_ok(),
        run_gate=runner({"cheap": refine("feature-3")}),
    )
    assert result.outcome is CommitOutcome.REVISED
    assert result.defects == ("feature-3",)
    got = store.get_artifact("a1")
    assert got is not None and got.status is ArtifactStatus.REVISED
    decision = store.decisions_for("a1")[0]
    assert decision.verdict is VerdictKind.REFINE
    assert decision.defects == ("feature-3",)


def test_no_binding_is_no_implicit_accept() -> None:
    store = make_store()
    propose(store, artifact_id="a1", artifact_type="Gateless")
    with pytest.raises(NoImplicitAccept):
        run_commit(
            "a1",
            store=store,
            sink=store,
            resolve_binding=resolver(),  # no binding for 'Gateless'
            validate_shape=shape_ok(),
            run_gate=runner({}),
        )
    # nothing was recorded and nothing accepted
    assert store.decisions_for("a1") == []
    got = store.get_artifact("a1")
    assert got is not None and got.status is ArtifactStatus.PROPOSED


def test_empty_pipeline_is_no_implicit_accept() -> None:
    store = make_store()
    propose(store, artifact_id="a1", artifact_type="Thing")
    with pytest.raises(NoImplicitAccept):
        run_commit(
            "a1",
            store=store,
            sink=store,
            resolve_binding=resolver(binding("Thing")),  # bound, but zero gates
            validate_shape=shape_ok(),
            run_gate=runner({}),
        )


def test_proposer_cannot_be_its_own_gate() -> None:
    store = make_store()
    propose(store, artifact_id="a1", artifact_type="Thing", created_by="same-id")
    with pytest.raises(ProposerIsGate):
        run_commit(
            "a1",
            store=store,
            sink=store,
            resolve_binding=resolver(binding("Thing", gate("g", identity="same-id"))),
            validate_shape=shape_ok(),
            run_gate=runner({"g": accept()}),
        )


def test_shape_error_is_correctable_and_records_nothing() -> None:
    store = make_store()
    propose(store, artifact_id="a1", artifact_type="Thing")
    result = run_commit(
        "a1",
        store=store,
        sink=store,
        resolve_binding=resolver(binding("Thing", gate("cheap"))),
        validate_shape=shape_fail("missing description"),
        run_gate=runner({"cheap": accept()}),
    )
    assert result.outcome is CommitOutcome.SHAPE_ERROR
    assert result.shape_error is not None
    assert result.status is None
    # §7.0: no decision row, no status change — still proposed
    assert store.decisions_for("a1") == []
    got = store.get_artifact("a1")
    assert got is not None and got.status is ArtifactStatus.PROPOSED
    assert store.rejected_log() == []


def test_requires_human_hard_gate_rests_at_tentative() -> None:
    store = make_store()
    propose(store, artifact_id="a1", artifact_type="Thing")
    result = run_commit(
        "a1",
        store=store,
        sink=store,
        resolve_binding=resolver(
            binding("Thing", gate("cheap"), gate("human", is_hard=True, requires_human=True))
        ),
        validate_shape=shape_ok(),
        run_gate=runner({"cheap": accept(), "human": None}),  # human can't auto-resolve
    )
    assert result.outcome is CommitOutcome.TENTATIVE
    got = store.get_artifact("a1")
    assert got is not None and got.status is ArtifactStatus.TENTATIVE
    assert [d.gate for d in store.decisions_for("a1")] == ["cheap"]


def test_acceptance_can_supersede_an_incumbent_atomically() -> None:
    store = make_store()
    # an accepted incumbent
    propose(store, artifact_id="old", artifact_type="Thing")
    store.set_status("old", ArtifactStatus.TENTATIVE)
    store.set_status("old", ArtifactStatus.ACCEPTED)
    # a better submission supersedes it
    propose(store, artifact_id="new", artifact_type="Thing")
    result = run_commit(
        "new",
        store=store,
        sink=store,
        resolve_binding=resolver(binding("Thing", gate("hard", is_hard=True))),
        validate_shape=shape_ok(),
        run_gate=runner({"hard": accept(score=0.9)}),
        supersedes="old",
    )
    assert result.outcome is CommitOutcome.ACCEPTED
    old = store.get_artifact("old")
    new = store.get_artifact("new")
    assert old is not None and new is not None
    assert old.status is ArtifactStatus.SUPERSEDED and old.superseded_by == "new"
    assert new.status is ArtifactStatus.ACCEPTED


def test_commit_requires_proposed_status() -> None:
    store = make_store()
    propose(store, artifact_id="a1", artifact_type="Thing")
    store.set_status("a1", ArtifactStatus.TENTATIVE)
    with pytest.raises(CommitError):
        run_commit(
            "a1",
            store=store,
            sink=store,
            resolve_binding=resolver(binding("Thing", gate("g"))),
            validate_shape=shape_ok(),
            run_gate=runner({"g": accept()}),
        )


def test_refine_then_revision_keeps_intact_lineage() -> None:
    """refine → revised → revises op → re-commit; provenance spans the revision (§6, §7)."""
    store = make_store()
    propose(store, artifact_id="v1", artifact_type="Thing")
    bound = resolver(binding("Thing", gate("cheap")))
    first = run_commit(
        "v1",
        store=store,
        sink=store,
        resolve_binding=bound,
        validate_shape=shape_ok(),
        run_gate=runner({"cheap": refine("bad-part")}),
    )
    assert first.outcome is CommitOutcome.REVISED

    # the agent re-proposes a revision via a 'revises' operation
    propose(store, artifact_id="v2", artifact_type="Thing", parents=["v1"], op_name="revises")
    store.link_revision("v1", "v2")
    second = run_commit(
        "v2",
        store=store,
        sink=store,
        resolve_binding=bound,
        validate_shape=shape_ok(),
        run_gate=runner({"cheap": accept()}),
    )
    assert second.outcome is CommitOutcome.ACCEPTED

    v1 = store.get_artifact("v1")
    assert v1 is not None and v1.revised_by == "v2"
    prov = store.get_provenance("v2")
    assert "v1" in {a.id for a in prov.ancestors}
    assert any(op.op_name == "revises" for op in prov.operations)


def test_proposer_identity_default_differs_from_gate() -> None:
    # sanity: the helper proposer (AGENT) and gate identity (VERIFIER) differ by default
    assert AGENT != "verifier"
