"""Store data-model and contract tests (spec §4) — ROADMAP Phase 1.1."""

from __future__ import annotations

import pytest

from tests.helpers import AGENT, make_store, propose
from verity.control_plane.store import (
    Artifact,
    ArtifactStatus,
    Decision,
    ObjectRef,
    Operation,
    OperationStatus,
    SchemaVersion,
    StoreError,
    VerdictKind,
)


def test_propose_roundtrips_artifact_and_operation() -> None:
    store = make_store()
    propose(store, artifact_id="a1", artifact_type="Thing")
    got = store.get_artifact("a1")
    assert got is not None
    assert got.type == "Thing"
    assert got.status is ArtifactStatus.PROPOSED
    assert got.created_by == AGENT
    assert got.created_at  # stamped by the injected clock
    ops = store.operations_into("a1")
    assert [op.op_name for op in ops] == ["produce"]


def test_propose_rejects_non_proposed_status() -> None:
    store = make_store()
    artifact = Artifact(
        id="a1",
        type="Thing",
        payload={"v": 1},
        status=ArtifactStatus.ACCEPTED,
        created_by=AGENT,
        created_at="t",
    )
    op = Operation("op1", "produce", (), "a1", OperationStatus.SUCCESS, "t")
    with pytest.raises(StoreError):
        store.propose(artifact, op)


def test_ids_are_never_reused(store_artifact: str = "a1") -> None:
    store = make_store()
    propose(store, artifact_id=store_artifact)
    with pytest.raises(StoreError):
        propose(store, artifact_id=store_artifact)


def test_object_store_roundtrip_and_content_addressing() -> None:
    store = make_store()
    ref = store.put_object(b"print('hello')")
    assert isinstance(ref, ObjectRef)
    assert ref.blob_ref == f"sha256:{ref.content_hash}"
    assert store.get_object(ref.content_hash) == b"print('hello')"
    # identical bytes hash identically (content-addressed)
    assert store.put_object(b"print('hello')").content_hash == ref.content_hash


def test_object_not_found_raises() -> None:
    store = make_store()
    with pytest.raises(StoreError):
        store.get_object("deadbeef")


def test_object_ref_payload_roundtrips() -> None:
    store = make_store()
    ref = store.put_object(b"the code")
    propose(store, artifact_id="a1", payload=ref)
    got = store.get_artifact("a1")
    assert got is not None
    assert got.payload == ref
    assert isinstance(got.payload, ObjectRef)
    assert store.get_object(got.payload.content_hash) == b"the code"


def test_query_by_type_and_status() -> None:
    store = make_store()
    propose(store, artifact_id="a1", artifact_type="Thing")
    propose(store, artifact_id="a2", artifact_type="Other")
    store.set_status("a2", ArtifactStatus.TENTATIVE)
    assert {a.id for a in store.query_artifacts(type="Thing")} == {"a1"}
    assert {a.id for a in store.query_artifacts(status=ArtifactStatus.TENTATIVE)} == {"a2"}
    assert {a.id for a in store.query_artifacts()} == {"a1", "a2"}


def test_operations_into_and_out_of() -> None:
    store = make_store()
    propose(store, artifact_id="root", is_root=True)
    propose(store, artifact_id="child", parents=["root"])
    assert [op.output_id for op in store.operations_into("child")] == ["child"]
    out = store.operations_out_of("root")
    assert [op.output_id for op in out] == ["child"]


def test_accept_superseding_is_atomic() -> None:
    store = make_store()
    propose(store, artifact_id="old")
    store.set_status("old", ArtifactStatus.TENTATIVE)
    store.set_status("old", ArtifactStatus.ACCEPTED)
    propose(store, artifact_id="new")
    store.set_status("new", ArtifactStatus.TENTATIVE)
    store.accept_superseding("new", "old")

    old = store.get_artifact("old")
    new = store.get_artifact("new")
    assert old is not None and new is not None
    assert old.status is ArtifactStatus.SUPERSEDED
    assert old.superseded_by == "new"
    assert new.status is ArtifactStatus.ACCEPTED
    assert [a.id for a in store.superseded_log()] == ["old"]


def test_link_revision_sets_pointer_and_status() -> None:
    store = make_store()
    propose(store, artifact_id="orig")
    propose(store, artifact_id="rev", parents=["orig"], op_name="revises")
    store.link_revision("orig", "rev")
    orig = store.get_artifact("orig")
    assert orig is not None
    assert orig.status is ArtifactStatus.REVISED
    assert orig.revised_by == "rev"


def test_decisions_are_recorded_and_queryable() -> None:
    store = make_store()
    propose(store, artifact_id="a1")
    store.record_decision(
        Decision("a1", "grounding", VerdictKind.ACCEPT, "looks grounded", score=0.9)
    )
    decisions = store.decisions_for("a1")
    assert len(decisions) == 1
    assert decisions[0].gate == "grounding"
    assert decisions[0].verdict is VerdictKind.ACCEPT
    assert decisions[0].score == 0.9
    assert decisions[0].created_at  # stamped


def test_rejected_log_retains_rejections() -> None:
    store = make_store()
    propose(store, artifact_id="a1")
    store.set_status("a1", ArtifactStatus.REJECTED)
    assert [a.id for a in store.rejected_log()] == ["a1"]
    # still fully queryable afterward (not erased, §5.3)
    assert store.get_artifact("a1") is not None


def test_get_provenance_returns_transitive_lineage() -> None:
    store = make_store()
    propose(store, artifact_id="root", is_root=True)
    propose(store, artifact_id="mid", parents=["root"])
    propose(store, artifact_id="leaf", parents=["mid"])
    store.record_decision(Decision("leaf", "g", VerdictKind.ACCEPT, "ok"))
    prov = store.get_provenance("leaf")
    assert prov.artifact.id == "leaf"
    assert {a.id for a in prov.ancestors} == {"root", "mid"}
    # the lineage includes each producing operation, including root's own (∅ → root)
    assert {op.output_id for op in prov.operations} == {"root", "mid", "leaf"}
    assert any(d.gate == "g" for d in prov.decisions)


def test_schema_versions_register_and_read() -> None:
    store = make_store()
    assert store.current_schema_version() is None
    store.register_schema_version(SchemaVersion(1, ("Thing",), ("produce",), "t1"))
    store.register_schema_version(SchemaVersion(2, ("Thing", "Other"), ("produce",), "t2"))
    current = store.current_schema_version()
    assert current is not None
    assert current.version == 2
    assert current.types == ("Thing", "Other")
