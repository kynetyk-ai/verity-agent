"""Fake-domain end-to-end tests (spec §8, §17.2; ADR 0001) — ROADMAP Phase 1.2/refactor.

Drives the §7 commit path through the *real* registries and the *real* opaque verifier
(:func:`build_fake_verifier`), proving the extension-point contracts compose — including "no
implicit accept" on the gateless type.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.helpers import make_store, propose
from verity.contracts import Artifact, ArtifactStatus, VerifierRequest
from verity.control_plane.commit import CommitOutcome, NoImplicitAccept, run_commit
from verity.domains.fake import (
    NOTE,
    ORPHAN,
    SOURCE,
    build_fake_domain,
    build_fake_verifier,
)


def _commit_note(store, note_id: str):
    domain = build_fake_domain()
    verifier = build_fake_verifier()

    def dispatch(artifact):
        return asyncio.run(verifier.dispatch(VerifierRequest(proposal=artifact)))

    return run_commit(
        note_id,
        store=store,
        sink=store,
        resolve_coverage=domain.gated_types.resolve,
        dispatch=dispatch,
        verifier_identity=verifier.identity,
    )


def test_fake_domain_registers_expected_schema() -> None:
    domain = build_fake_domain()
    assert {t.name for t in domain.schema.types()} == {SOURCE, NOTE, ORPHAN}
    source = domain.schema.type(SOURCE)
    assert source is not None and source.is_root
    # the typed Note-producing edge is an operation signature (no tool registry, by design)
    assert domain.schema.operation("author") is not None


def test_well_formed_note_is_accepted() -> None:
    store = make_store()
    propose(store, artifact_id="src", artifact_type=SOURCE, is_root=True)
    propose(store, artifact_id="n1", artifact_type=NOTE, parents=["src"], payload={"text": "hi"})
    result = _commit_note(store, "n1")
    assert result.outcome is CommitOutcome.ACCEPTED
    got = store.get_artifact("n1")
    assert got is not None and got.status is ArtifactStatus.ACCEPTED
    assert {d.gate for d in store.decisions_for("n1")} == {"well-formed", "worth-keeping"}


def test_note_marked_reject_is_rejected_at_the_cheap_check() -> None:
    store = make_store()
    propose(store, artifact_id="src", artifact_type=SOURCE, is_root=True)
    propose(
        store, artifact_id="n1", artifact_type=NOTE, parents=["src"],
        payload={"text": "x", "reject": True},
    )
    result = _commit_note(store, "n1")
    assert result.outcome is CommitOutcome.REJECTED
    assert [d.gate for d in store.decisions_for("n1")] == ["well-formed"]


def test_note_marked_defect_is_revised_with_defects() -> None:
    store = make_store()
    propose(store, artifact_id="src", artifact_type=SOURCE, is_root=True)
    propose(
        store, artifact_id="n1", artifact_type=NOTE, parents=["src"],
        payload={"text": "x", "defect": "tone"},
    )
    result = _commit_note(store, "n1")
    assert result.outcome is CommitOutcome.REVISED
    assert result.defects == ("tone",)
    got = store.get_artifact("n1")
    assert got is not None and got.status is ArtifactStatus.REVISED


def test_note_failing_the_hard_check_is_rejected() -> None:
    store = make_store()
    propose(store, artifact_id="src", artifact_type=SOURCE, is_root=True)
    propose(
        store, artifact_id="n1", artifact_type=NOTE, parents=["src"],
        payload={"text": "x", "keep": False},
    )
    result = _commit_note(store, "n1")
    assert result.outcome is CommitOutcome.REJECTED
    assert {d.gate for d in store.decisions_for("n1")} == {"well-formed", "worth-keeping"}


def test_malformed_note_fails_the_shape_check() -> None:
    domain = build_fake_domain()
    bad = Artifact("n1", NOTE, {"oops": 1}, ArtifactStatus.PROPOSED, "agent", "")
    assert domain.shape_validator(bad) is not None  # missing 'text' → shape error, before any gate


def test_gateless_type_cannot_be_committed() -> None:
    store = make_store()
    propose(store, artifact_id="src", artifact_type=SOURCE, is_root=True)
    propose(store, artifact_id="orph", artifact_type=ORPHAN, parents=["src"], payload={"x": 1})
    with pytest.raises(NoImplicitAccept):
        _commit_note(store, "orph")
    got = store.get_artifact("orph")
    assert got is not None and got.status is ArtifactStatus.PROPOSED
