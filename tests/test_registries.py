"""Extension-point registry tests (spec §8) — ROADMAP Phase 1.2."""

from __future__ import annotations

import pytest

from tests.helpers import make_store, propose
from verity.control_plane.independence import StoreInput
from verity.control_plane.registries import (
    ArtifactTypeDef,
    DefaultRetrievalPolicy,
    GatedTypeRegistry,
    OperationSignature,
    RegistryError,
    SchemaRegistry,
)
from verity.control_plane.store import ArtifactStatus

# ----------------------------------------------------------------------- schema (§8.1)


def test_schema_registers_types_and_operations() -> None:
    schema = SchemaRegistry()
    schema.register_type(ArtifactTypeDef("Source", is_root=True))
    schema.register_type(ArtifactTypeDef("Note"))
    schema.register_operation(OperationSignature("author", inputs=("Source",), output="Note"))
    assert schema.is_registered("Note")
    assert schema.type("Source") == ArtifactTypeDef("Source", is_root=True)
    assert {t.name for t in schema.types()} == {"Source", "Note"}
    assert schema.operation("author") is not None


def test_schema_rejects_duplicate_type() -> None:
    schema = SchemaRegistry()
    schema.register_type(ArtifactTypeDef("Note"))
    with pytest.raises(RegistryError):
        schema.register_type(ArtifactTypeDef("Note"))


def test_schema_operation_must_reference_known_types() -> None:
    schema = SchemaRegistry()
    schema.register_type(ArtifactTypeDef("Note"))
    with pytest.raises(RegistryError):
        schema.register_operation(OperationSignature("author", inputs=("Source",), output="Note"))


def test_schema_snapshot_is_stampable() -> None:
    schema = SchemaRegistry()
    schema.register_type(ArtifactTypeDef("Note"))
    schema.register_operation(OperationSignature("noop", inputs=(), output="Note"))
    snap = schema.snapshot(version=1, created_at="t1")
    assert snap.version == 1
    assert snap.types == ("Note",)
    assert snap.op_signatures == ("noop",)


# -------------------------------------------------------------------------- gates (§8.3)


def test_gated_type_registry_resolves_inputs_and_unknown_is_none() -> None:
    gated = GatedTypeRegistry()
    gated.gate("Note", declared_inputs=frozenset({StoreInput.INCUMBENTS}))
    assert gated.resolve("Note") == frozenset({StoreInput.INCUMBENTS})
    # the no-implicit-accept trigger: an unregistered type resolves to None (§5.7)
    assert gated.resolve("Orphan") is None
    assert gated.is_gated("Note") and not gated.is_gated("Orphan")


def test_gated_type_with_empty_inputs_is_still_gated() -> None:
    gated = GatedTypeRegistry()
    gated.gate("Note")  # gated, but its verifier sees only the artifact under test
    assert gated.resolve("Note") == frozenset()
    assert gated.is_gated("Note")


def test_gated_type_registry_rejects_duplicate() -> None:
    gated = GatedTypeRegistry()
    gated.gate("Note")
    with pytest.raises(RegistryError):
        gated.gate("Note")


# --------------------------------------------------------------------- retrieval (§8.4)


def test_default_retrieval_prefers_accepted_over_tentative_and_excludes_terminal() -> None:
    store = make_store()
    propose(store, artifact_id="acc")
    propose(store, artifact_id="ten")
    propose(store, artifact_id="rej")
    store.set_status("acc", ArtifactStatus.TENTATIVE)
    store.set_status("acc", ArtifactStatus.ACCEPTED)
    store.set_status("ten", ArtifactStatus.TENTATIVE)
    store.set_status("rej", ArtifactStatus.REJECTED)

    policy = DefaultRetrievalPolicy()
    selected = policy.select("goal", store, limit=10)
    ids = [a.id for a in selected]
    assert ids[0] == "acc"  # accepted outranks tentative
    assert "ten" in ids
    assert "rej" not in ids  # terminal excluded by default


def test_default_retrieval_respects_limit_and_can_include_terminal() -> None:
    store = make_store()
    for i in range(5):
        propose(store, artifact_id=f"a{i}")
    store.set_status("a0", ArtifactStatus.REJECTED)

    assert len(DefaultRetrievalPolicy().select("g", store, limit=2)) == 2
    inclusive = DefaultRetrievalPolicy(include_terminal=True).select("g", store, limit=10)
    assert "a0" in {a.id for a in inclusive}
