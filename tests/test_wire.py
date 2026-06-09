"""The boundary-type JSON codecs (ROADMAP Phase 7.1). Pure round-trips — no network, no Docker.

The whole serialization surface is offline-provable: ``from_dict(to_dict(x)) == x`` for every type
that crosses a port (frozen dataclasses give free structural equality). Plus the two load-bearing
invariants: object bytes survive intact, and the proposer's rationale is structurally absent from a
VerifierRequest.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    GateDecision,
    ObjectRef,
    Operation,
    OperationStatus,
    ProposalEnvelope,
    ServedContext,
    VerdictBundle,
    VerdictKind,
    VerifierRequest,
)
from verity.contracts.wire import (
    OBJECT_REF_MARKER,
    WireError,
    artifact_from_dict,
    artifact_to_dict,
    gate_decision_from_dict,
    gate_decision_to_dict,
    objects_from_wire,
    objects_to_wire,
    operation_from_dict,
    operation_to_dict,
    payload_from_json,
    payload_to_json,
    proposal_envelope_from_dict,
    proposal_envelope_to_dict,
    served_context_from_dict,
    served_context_to_dict,
    verdict_bundle_from_dict,
    verdict_bundle_to_dict,
    verifier_request_from_dict,
    verifier_request_to_dict,
)
from verity.control_plane.store import SqliteStore

_REF = ObjectRef(blob_ref="sha256:abc", content_hash="abc")
_ARTIFACT = Artifact(
    id="a1", type="Submission", payload={"entrypoint": "s.py"}, status=ArtifactStatus.PROPOSED,
    created_by="agent", created_at="t1",
)
_ARTIFACT_WITH_OBJECTS = Artifact(
    id="a2", type="Submission", payload=_REF, status=ArtifactStatus.ACCEPTED,
    created_by="agent", created_at="t2", superseded_by="a3", is_root=False,
    objects=(("submission.py", _REF), ("requirements.txt", ObjectRef("sha256:def", "def"))),
)
_OP = Operation("op1", "submit", ("ds",), "a1", OperationStatus.SUCCESS, "t1")


def _roundtrip(to, frm, value):  # type: ignore[no-untyped-def]
    return frm(to(value))


def test_object_ref_and_payload_union() -> None:
    assert payload_from_json(payload_to_json(_REF)) == _REF
    assert payload_from_json(payload_to_json({"x": 1, "y": [1, 2]})) == {"x": 1, "y": [1, 2]}
    assert payload_from_json(payload_to_json(None)) is None
    # an ObjectRef payload rides under the shared marker
    assert OBJECT_REF_MARKER in payload_to_json(_REF)


def test_artifact_roundtrips_both_payload_arms_and_sidecar() -> None:
    assert _roundtrip(artifact_to_dict, artifact_from_dict, _ARTIFACT) == _ARTIFACT
    assert _roundtrip(artifact_to_dict, artifact_from_dict, _ARTIFACT_WITH_OBJECTS) == (
        _ARTIFACT_WITH_OBJECTS
    )


def test_operation_roundtrips() -> None:
    assert _roundtrip(operation_to_dict, operation_from_dict, _OP) == _OP


def test_gate_decision_roundtrips_with_and_without_optionals() -> None:
    bare = GateDecision(gate="runs-clean", kind=VerdictKind.ACCEPT, rationale="ok")
    full = GateDecision(
        gate="selection", kind=VerdictKind.REJECT, rationale="below bar",
        defects=("too complex",), score=0.42,
    )
    assert _roundtrip(gate_decision_to_dict, gate_decision_from_dict, bare) == bare
    assert _roundtrip(gate_decision_to_dict, gate_decision_from_dict, full) == full


def test_verdict_bundle_roundtrips_with_supersedes() -> None:
    bundle = VerdictBundle(
        status=ArtifactStatus.ACCEPTED,
        decisions=(GateDecision("selection", VerdictKind.ACCEPT, "beats", score=0.9),),
        supersedes="incumbent-1",
    )
    assert _roundtrip(verdict_bundle_to_dict, verdict_bundle_from_dict, bundle) == bundle


def test_objects_survive_base64_including_binary() -> None:
    objs = {"a.bin": bytes(range(256)), "u.txt": "héllo→".encode(), "empty": b""}
    assert objects_from_wire(objects_to_wire(objs)) == objs


def test_unsupported_object_encoding_raises() -> None:
    with pytest.raises(WireError, match="unsupported object encoding"):
        objects_from_wire({"x": {"ref": {"blob_ref": "b", "content_hash": "h"}}})


def test_served_context_roundtrips_with_workspace_objects() -> None:
    ctx = ServedContext(
        system_prompt="SYS", tail="goal", feedback="fix x",
        workspace_objects={"scratch": {"prior/submission.py": b"print(1)"}},
    )
    assert _roundtrip(served_context_to_dict, served_context_from_dict, ctx) == ctx
    assert _roundtrip(served_context_to_dict, served_context_from_dict, ServedContext("S")) == (
        ServedContext("S")
    )


def test_proposal_envelope_roundtrips_with_metadata_and_telemetry() -> None:
    env = ProposalEnvelope(
        artifact=_ARTIFACT, operation=_OP, metadata="I tried X because Y",
        objects={"submission.py": b"print(1)"},
        agent_telemetry={"input_tokens": 100, "model": "claude"},
    )
    assert _roundtrip(proposal_envelope_to_dict, proposal_envelope_from_dict, env) == env
    minimal = ProposalEnvelope(artifact=_ARTIFACT, operation=_OP)
    assert _roundtrip(proposal_envelope_to_dict, proposal_envelope_from_dict, minimal) == minimal


def test_verifier_request_roundtrips_with_slice_and_objects() -> None:
    req = VerifierRequest(
        proposal=_ARTIFACT, store_slice=(_ARTIFACT_WITH_OBJECTS,),
        objects={"submission.py": b"print(1)"},
    )
    assert _roundtrip(verifier_request_to_dict, verifier_request_from_dict, req) == req


def test_rationale_is_structurally_absent_from_verifier_request() -> None:
    # The proposer's rationale rides on the envelope but must NEVER reach a gate (§10). A
    # VerifierRequest derived from the same proposal has no metadata field; assert the marker string
    # is byte-absent from the serialized request.
    marker = "SECRET-RATIONALE-c0ffee"
    env = ProposalEnvelope(artifact=_ARTIFACT, operation=_OP, metadata=marker)
    req = VerifierRequest(proposal=env.artifact)  # what the control plane forwards
    assert marker in json.dumps(proposal_envelope_to_dict(env))  # present on the envelope
    assert marker not in json.dumps(verifier_request_to_dict(req))  # absent on the wire to the gate


def test_wire_payload_agrees_with_the_store_codec() -> None:
    # The shared OBJECT_REF_MARKER: the on-wire payload encoding and the store's payload column
    # encoding must produce the same JSON for an ObjectRef, or on-disk and on-wire would drift.
    assert json.loads(SqliteStore._dump_payload(_REF)) == payload_to_json(_REF)
    assert SqliteStore._load_payload(json.dumps(payload_to_json(_REF))) == _REF


# ----------------------------------------------------------------------------- property tests

_status = st.sampled_from(list(ArtifactStatus))
_json_scalar = st.none() | st.booleans() | st.integers() | st.text()


@given(
    art_id=st.text(min_size=1), art_type=st.text(min_size=1), payload=_json_scalar,
    status=_status, is_root=st.booleans(),
)
def test_artifact_roundtrip_property(
    art_id: str, art_type: str, payload: object, status: ArtifactStatus, is_root: bool
) -> None:
    a = Artifact(
        id=art_id, type=art_type, payload=payload, status=status,
        created_by="agent", created_at="t", is_root=is_root,
    )
    assert artifact_from_dict(artifact_to_dict(a)) == a


@given(
    status=_status,
    scores=st.lists(st.none() | st.floats(allow_nan=False, allow_infinity=False), max_size=4),
)
def test_verdict_bundle_roundtrip_property(
    status: ArtifactStatus, scores: list[float | None]
) -> None:
    bundle = VerdictBundle(
        status=status,
        decisions=tuple(
            GateDecision(f"g{i}", VerdictKind.ACCEPT, "r", score=s) for i, s in enumerate(scores)
        ),
    )
    assert verdict_bundle_from_dict(verdict_bundle_to_dict(bundle)) == bundle
