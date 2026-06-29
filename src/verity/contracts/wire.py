"""JSON codecs for the boundary value types (ROADMAP Phase 7.1 — the wire foundation).

The control plane talks to the sandbox and verifier through the async ports (§3.3–§3.6); today that
call is in-process, tomorrow it crosses a network. This module is the bedrock that makes the hop
possible: a total, dependency-free ``*_to_dict`` / ``*_from_dict`` codec for every type that crosses
a port — :class:`Artifact`, :class:`Operation`, :class:`GateDecision`, :class:`VerdictBundle`,
:class:`ServedContext`, :class:`ProposalEnvelope`, :class:`VerifierRequest`.

The codecs are **functions here, not methods** on the value types: :mod:`verity.contracts.model`
keeps those frozen types codec-free (the "published vocabulary, owned by none"), and the enum /
union / object-bytes handling is cross-cutting and versioned, so it belongs in one auditable place
with a :data:`WIRE_VERSION` (mirroring ``REPORT_SCHEMA_VERSION``). It is pure — no I/O, no network,
no third-party imports — so the whole serialization surface is offline-testable.

**Object bytes** (the attachment payloads on :class:`ServedContext`, :class:`ProposalEnvelope`,
:class:`VerifierRequest`) ride as a tagged :data:`WireObject` union — ``{"inline": "<base64>"}`` in
v1. (A by-reference form ``{"ref": {...}}`` against a shared blob store is a later optimization; the
sandbox→control-plane harvest direction has no shared store, so inline is the semantic floor.)

**Rationale segregation is structural, not disciplinary.** :func:`verifier_request_to_dict` emits
only ``{proposal, store_slice, objects}`` — :class:`VerifierRequest` has no rationale field, so the
proposer's reasoning cannot ride to a gate (§10). :func:`proposal_envelope_to_dict` *does* carry
``metadata`` (provenance-bound for the control plane's rationale channel).
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any, cast

from verity.contracts.model import (
    Artifact,
    ArtifactStatus,
    GateDecision,
    ObjectRef,
    Operation,
    OperationStatus,
    Payload,
    VerdictBundle,
    VerdictKind,
)
from verity.contracts.ports import (
    ProposalEnvelope,
    ServedContext,
    VerifierRequest,
    VerifierSetup,
)

__all__ = [
    "WIRE_VERSION",
    "OBJECT_REF_MARKER",
    "WireError",
    "object_ref_to_dict",
    "object_ref_from_dict",
    "payload_to_json",
    "payload_from_json",
    "artifact_to_dict",
    "artifact_from_dict",
    "operation_to_dict",
    "operation_from_dict",
    "gate_decision_to_dict",
    "gate_decision_from_dict",
    "verdict_bundle_to_dict",
    "verdict_bundle_from_dict",
    "objects_to_wire",
    "objects_from_wire",
    "served_context_to_dict",
    "served_context_from_dict",
    "proposal_envelope_to_dict",
    "proposal_envelope_from_dict",
    "verifier_request_to_dict",
    "verifier_request_from_dict",
    "verifier_setup_to_dict",
    "verifier_setup_from_dict",
]

# Bump when the wire JSON shape changes, so a receiver can refuse an incompatible peer.
WIRE_VERSION = 1

# The tag that marks an ``ObjectRef`` inside a JSON payload. **Shared** with the store's payload
# column codec (`store.py` imports this) so the on-disk and on-wire encodings provably agree.
OBJECT_REF_MARKER = "__object_ref__"

JsonDict = dict[str, Any]


class WireError(ValueError):
    """A value could not be encoded to / decoded from the wire shape."""


# ---------------------------------------------------------------- ObjectRef + payload union


def object_ref_to_dict(ref: ObjectRef) -> JsonDict:
    return {"blob_ref": ref.blob_ref, "content_hash": ref.content_hash}


def object_ref_from_dict(d: Mapping[str, Any]) -> ObjectRef:
    return ObjectRef(blob_ref=d["blob_ref"], content_hash=d["content_hash"])


def payload_to_json(payload: Payload) -> Any:
    """Encode a ``JSONValue | ObjectRef`` payload; an ObjectRef rides under the marker tag."""
    if isinstance(payload, ObjectRef):
        return {OBJECT_REF_MARKER: object_ref_to_dict(payload)}
    return payload


def payload_from_json(value: Any) -> Payload:
    if isinstance(value, dict) and OBJECT_REF_MARKER in value:
        return object_ref_from_dict(value[OBJECT_REF_MARKER])
    return cast("Payload", value)


# ------------------------------------------------------------------------- model.py types


def artifact_to_dict(a: Artifact) -> JsonDict:
    return {
        "id": a.id,
        "type": a.type,
        "payload": payload_to_json(a.payload),
        "status": a.status.value,
        "created_by": a.created_by,
        "created_at": a.created_at,
        "superseded_by": a.superseded_by,
        "revised_by": a.revised_by,
        "is_root": a.is_root,
        "objects": [[name, object_ref_to_dict(ref)] for name, ref in a.objects],
    }


def artifact_from_dict(d: Mapping[str, Any]) -> Artifact:
    return Artifact(
        id=d["id"],
        type=d["type"],
        payload=payload_from_json(d["payload"]),
        status=ArtifactStatus(d["status"]),
        created_by=d["created_by"],
        created_at=d["created_at"],
        superseded_by=d.get("superseded_by"),
        revised_by=d.get("revised_by"),
        is_root=d.get("is_root", False),
        objects=tuple(
            (name, object_ref_from_dict(ref)) for name, ref in d.get("objects", [])
        ),
    )


def operation_to_dict(o: Operation) -> JsonDict:
    return {
        "op_id": o.op_id,
        "op_name": o.op_name,
        "parents": list(o.parents),
        "output_id": o.output_id,
        "status": o.status.value,
        "created_at": o.created_at,
    }


def operation_from_dict(d: Mapping[str, Any]) -> Operation:
    return Operation(
        op_id=d["op_id"],
        op_name=d["op_name"],
        parents=tuple(d["parents"]),
        output_id=d["output_id"],
        status=OperationStatus(d["status"]),
        created_at=d["created_at"],
    )


def gate_decision_to_dict(g: GateDecision) -> JsonDict:
    return {
        "gate": g.gate,
        "kind": g.kind.value,
        "rationale": g.rationale,
        "defects": list(g.defects) if g.defects is not None else None,
        "score": g.score,
    }


def gate_decision_from_dict(d: Mapping[str, Any]) -> GateDecision:
    defects = d.get("defects")
    return GateDecision(
        gate=d["gate"],
        kind=VerdictKind(d["kind"]),
        rationale=d["rationale"],
        defects=tuple(defects) if defects is not None else None,
        score=d.get("score"),
    )


def verdict_bundle_to_dict(b: VerdictBundle) -> JsonDict:
    return {
        "status": b.status.value,
        "decisions": [gate_decision_to_dict(d) for d in b.decisions],
        "supersedes": b.supersedes,
    }


def verdict_bundle_from_dict(d: Mapping[str, Any]) -> VerdictBundle:
    return VerdictBundle(
        status=ArtifactStatus(d["status"]),
        decisions=tuple(gate_decision_from_dict(x) for x in d.get("decisions", [])),
        supersedes=d.get("supersedes"),
    )


# --------------------------------------------------------------------- object bytes <-> wire


def _object_to_wire(data: bytes) -> JsonDict:
    return {"inline": base64.b64encode(data).decode("ascii")}


def _object_from_wire(w: Mapping[str, Any]) -> bytes:
    if "inline" in w:
        return base64.b64decode(w["inline"])
    # The by-reference form needs a blob fetcher (a later optimization); v1 is inline-only.
    raise WireError(f"unsupported object encoding: {sorted(w)}")


def objects_to_wire(objects: Mapping[str, bytes]) -> dict[str, JsonDict]:
    return {name: _object_to_wire(data) for name, data in objects.items()}


def objects_from_wire(d: Mapping[str, Any]) -> dict[str, bytes]:
    return {name: _object_from_wire(w) for name, w in d.items()}


# --------------------------------------------------------------------- ports.py boundary types


def served_context_to_dict(c: ServedContext) -> JsonDict:
    return {
        "system_prompt": c.system_prompt,
        "tail": c.tail,
        "feedback": c.feedback,
        "workspace_objects": {
            role: objects_to_wire(files) for role, files in c.workspace_objects.items()
        },
    }


def served_context_from_dict(d: Mapping[str, Any]) -> ServedContext:
    return ServedContext(
        system_prompt=d["system_prompt"],
        tail=d.get("tail", ""),
        feedback=d.get("feedback", ""),
        workspace_objects={
            role: objects_from_wire(files)
            for role, files in d.get("workspace_objects", {}).items()
        },
    )


def proposal_envelope_to_dict(e: ProposalEnvelope) -> JsonDict:
    return {
        "artifact": artifact_to_dict(e.artifact),
        "operation": operation_to_dict(e.operation),
        "metadata": e.metadata,  # the rationale channel — provenance-bound, never sent to a gate
        "objects": objects_to_wire(e.objects),
        "agent_telemetry": dict(e.agent_telemetry) if e.agent_telemetry is not None else None,
        # The step transcript rides as the same tagged inline-base64 blob as any object attachment.
        "transcript": _object_to_wire(e.transcript) if e.transcript is not None else None,
    }


def proposal_envelope_from_dict(d: Mapping[str, Any]) -> ProposalEnvelope:
    telemetry = d.get("agent_telemetry")
    transcript = d.get("transcript")
    return ProposalEnvelope(
        artifact=artifact_from_dict(d["artifact"]),
        operation=operation_from_dict(d["operation"]),
        metadata=d.get("metadata", ""),
        objects=objects_from_wire(d.get("objects", {})),
        agent_telemetry=dict(telemetry) if telemetry is not None else None,
        transcript=_object_from_wire(transcript) if transcript is not None else None,
    )


def verifier_request_to_dict(r: VerifierRequest) -> JsonDict:
    # No ``metadata`` key: VerifierRequest carries no rationale field, so the proposer's reasoning
    # cannot ride to a gate (§10). This is the structural enforcement of rationale segregation.
    return {
        "proposal": artifact_to_dict(r.proposal),
        "store_slice": [artifact_to_dict(a) for a in r.store_slice],
        "objects": objects_to_wire(r.objects),
        # Durably-recorded per-gate measurements (not rationale): {artifact_id: {gate: score}}.
        "scores": {aid: dict(gates) for aid, gates in r.scores.items()},
    }


def verifier_request_from_dict(d: Mapping[str, Any]) -> VerifierRequest:
    return VerifierRequest(
        proposal=artifact_from_dict(d["proposal"]),
        store_slice=tuple(artifact_from_dict(a) for a in d.get("store_slice", [])),
        objects=objects_from_wire(d.get("objects", {})),
        scores={
            aid: {g: float(s) for g, s in gates.items()}
            for aid, gates in d.get("scores", {}).items()
        },
    )


def verifier_setup_to_dict(s: VerifierSetup) -> JsonDict:
    # ``objects`` ride as the tagged inline-base64 union (like every other byte attachment);
    # ``params`` are plain JSON knobs. No rationale field — segregation is structural (§10).
    return {"objects": objects_to_wire(s.objects), "params": dict(s.params)}


def verifier_setup_from_dict(d: Mapping[str, Any]) -> VerifierSetup:
    return VerifierSetup(
        objects=objects_from_wire(d.get("objects", {})),
        params=dict(d.get("params", {})),
    )
