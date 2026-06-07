"""Fake-domain end-to-end tests (spec §8, §17.2) — ROADMAP Phase 1.2.

Drives the §7 commit path through the *real* registries (not test fakes), proving the
extension-point contracts compose — including "no implicit accept" on the gateless type.
"""

from __future__ import annotations

import pytest

from tests.helpers import make_store, propose
from verity.control_plane.commit import CommitOutcome, NoImplicitAccept, run_commit
from verity.control_plane.store import ArtifactStatus
from verity.domains.fake import NOTE, ORPHAN, SOURCE, build_fake_domain


def _commit_note(store, note_id):
    domain = build_fake_domain()
    return run_commit(
        note_id,
        store=store,
        sink=store,
        resolve_binding=domain.gates.resolve,
        validate_shape=domain.shape_validator,
        run_gate=domain.run_gate,
    )


def test_fake_domain_registers_expected_schema() -> None:
    domain = build_fake_domain()
    assert {t.name for t in domain.schema.types()} == {SOURCE, NOTE, ORPHAN}
    source = domain.schema.type(SOURCE)
    assert source is not None and source.is_root
    # the typed Note-producing edge is an operation signature (no tool registry, by design)
    assert domain.schema.operation("author") is not None


def test_well_formed_note_walks_to_accepted() -> None:
    store = make_store()
    propose(store, artifact_id="src", artifact_type=SOURCE, is_root=True)
    propose(store, artifact_id="n1", artifact_type=NOTE, parents=["src"], payload={"text": "hi"})
    result = _commit_note(store, "n1")
    assert result.outcome is CommitOutcome.ACCEPTED
    got = store.get_artifact("n1")
    assert got is not None and got.status is ArtifactStatus.ACCEPTED
    assert {d.gate for d in store.decisions_for("n1")} == {"well-formed", "worth-keeping"}


def test_note_marked_reject_is_rejected_at_cheap_gate() -> None:
    store = make_store()
    propose(store, artifact_id="src", artifact_type=SOURCE, is_root=True)
    propose(
        store,
        artifact_id="n1",
        artifact_type=NOTE,
        parents=["src"],
        payload={"text": "x", "reject": True},
    )
    result = _commit_note(store, "n1")
    assert result.outcome is CommitOutcome.REJECTED
    assert [d.gate for d in store.decisions_for("n1")] == ["well-formed"]


def test_note_marked_defect_is_revised_with_defects() -> None:
    store = make_store()
    propose(store, artifact_id="src", artifact_type=SOURCE, is_root=True)
    propose(
        store,
        artifact_id="n1",
        artifact_type=NOTE,
        parents=["src"],
        payload={"text": "x", "defect": "tone"},
    )
    result = _commit_note(store, "n1")
    assert result.outcome is CommitOutcome.REVISED
    assert result.defects == ("tone",)
    got = store.get_artifact("n1")
    assert got is not None and got.status is ArtifactStatus.REVISED


def test_note_failing_hard_gate_rejects_from_tentative() -> None:
    store = make_store()
    propose(store, artifact_id="src", artifact_type=SOURCE, is_root=True)
    propose(
        store,
        artifact_id="n1",
        artifact_type=NOTE,
        parents=["src"],
        payload={"text": "x", "keep": False},
    )
    result = _commit_note(store, "n1")
    assert result.outcome is CommitOutcome.REJECTED
    assert {d.gate for d in store.decisions_for("n1")} == {"well-formed", "worth-keeping"}


def test_malformed_note_returns_shape_error() -> None:
    store = make_store()
    propose(store, artifact_id="src", artifact_type=SOURCE, is_root=True)
    # missing the required 'text' field
    propose(store, artifact_id="n1", artifact_type=NOTE, parents=["src"], payload={"oops": 1})
    result = _commit_note(store, "n1")
    assert result.outcome is CommitOutcome.SHAPE_ERROR
    assert store.decisions_for("n1") == []


def test_gateless_type_cannot_be_committed() -> None:
    store = make_store()
    propose(store, artifact_id="src", artifact_type=SOURCE, is_root=True)
    propose(store, artifact_id="orph", artifact_type=ORPHAN, parents=["src"], payload={"x": 1})
    domain = build_fake_domain()
    with pytest.raises(NoImplicitAccept):
        run_commit(
            "orph",
            store=store,
            sink=store,
            resolve_binding=domain.gates.resolve,
            validate_shape=domain.shape_validator,
            run_gate=domain.run_gate,
        )
    got = store.get_artifact("orph")
    assert got is not None and got.status is ArtifactStatus.PROPOSED
