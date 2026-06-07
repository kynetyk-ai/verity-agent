"""Extension-point registry tests (spec §8) — ROADMAP Phase 1.2."""

from __future__ import annotations

import pytest

from tests.helpers import make_store, propose
from verity.control_plane.commit import GateSpec
from verity.control_plane.registries import (
    ArtifactTypeDef,
    DefaultRetrievalPolicy,
    GateRegistry,
    OperationSignature,
    RegistryError,
    SchemaRegistry,
    ToolRegistry,
    ToolSpec,
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


# ------------------------------------------------------------------------- tools (§8.2)


def test_tool_registry_validates_against_schema() -> None:
    schema = SchemaRegistry()
    schema.register_type(ArtifactTypeDef("Source", is_root=True))
    schema.register_type(ArtifactTypeDef("Note"))
    tools = ToolRegistry(schema)
    tools.register(ToolSpec("author_note", inputs=("Source",), output="Note"))
    assert tools.get("author_note") is not None
    with pytest.raises(RegistryError):
        tools.register(ToolSpec("bad", inputs=("Ghost",), output="Note"))


# -------------------------------------------------------------------------- gates (§8.3)


def test_gate_registry_resolves_bindings_and_unknown_is_none() -> None:
    gates = GateRegistry()
    gates.bind("Note", GateSpec("g", identity="verifier"))
    assert gates.resolve("Note") is not None
    # the no-implicit-accept trigger: an unbound type resolves to None (§5.7)
    assert gates.resolve("Orphan") is None


def test_gate_registry_refuses_empty_pipeline() -> None:
    gates = GateRegistry()
    with pytest.raises(RegistryError):
        gates.bind("Note")  # an empty pipeline is an implicit accept (§5.7)


def test_gate_registry_rejects_duplicate_binding() -> None:
    gates = GateRegistry()
    gates.bind("Note", GateSpec("g", identity="verifier"))
    with pytest.raises(RegistryError):
        gates.bind("Note", GateSpec("g2", identity="verifier"))


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
