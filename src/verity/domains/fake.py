"""A trivial fake domain to prove the extension-point contracts (spec §8, §17.2).

Not a real domain — it exists only to exercise the registries and the commit path before the
feature-engineering domain (§12, Phase 4) lands. It is deliberately minimal and **deterministic**
(no LLM, no real evaluation), and it includes a **gateless type** so "no implicit accept" (§5.7)
can be demonstrated: registering a type with no gate binding and attempting to commit it must
error, not default-accept.

Types:
* ``Source`` — a root (raw input) type.
* ``Note`` — the gated type: a cheap "well-formed" gate (→ ``tentative``) then a hard
  "worth-keeping" gate (→ ``accepted``).
* ``Orphan`` — a registered type with **no** gate binding (the no-implicit-accept probe).

The in-process gate runner here stands in for the verifier *for this fake domain only*; the real
verifier is a separate service (§3.6, Phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.contracts import JSONValue, VerifierRequest
from verity.control_plane.commit import (
    GateRunner,
    GateSpec,
    GateVerdict,
    ShapeError,
    ShapeValidator,
)
from verity.control_plane.independence import StoreInput
from verity.control_plane.registries import (
    ArtifactTypeDef,
    GateRegistry,
    OperationSignature,
    SchemaRegistry,
)
from verity.control_plane.store import Artifact, VerdictKind
from verity.verifier import CheckOutcome, SdkVerifier, deterministic_check

__all__ = [
    "SOURCE",
    "NOTE",
    "ORPHAN",
    "FakeDomain",
    "build_fake_domain",
    "build_fake_verifier",
]

SOURCE = "Source"
NOTE = "Note"
ORPHAN = "Orphan"

# Gate identities — distinct from any proposer, so proposer ≠ gate holds (§7.2).
_GATE_IDENTITY = "fake-verifier"


@dataclass(frozen=True, slots=True)
class FakeDomain:
    """The bundle a task wires in: the registries, the shape validator, and the gate runner."""

    schema: SchemaRegistry
    gates: GateRegistry
    shape_validator: ShapeValidator
    run_gate: GateRunner


def build_fake_domain() -> FakeDomain:
    schema = SchemaRegistry()
    schema.register_type(ArtifactTypeDef(SOURCE, is_root=True))
    schema.register_type(ArtifactTypeDef(NOTE))
    schema.register_type(ArtifactTypeDef(ORPHAN))
    # 'author' is the typed edge a Note-producing tool realizes; the executable tool itself is a
    # harness-bound, sandbox-adapter concern (Phase 3), not a control-plane registration.
    schema.register_operation(OperationSignature("author", inputs=(SOURCE,), output=NOTE))
    schema.register_operation(OperationSignature("note_orphan", inputs=(SOURCE,), output=ORPHAN))

    gates = GateRegistry()
    # 'Note' has a cheap structural gate then a hard selection gate; 'Orphan' is left UNBOUND.
    # The cheap gate is self-contained (declares no inputs → sees only the artifact); the hard
    # selection gate declares the incumbents it must beat and the rejected-log (§8.3, §10).
    gates.bind(
        NOTE,
        GateSpec("well-formed", identity=_GATE_IDENTITY, is_hard=False),
        GateSpec(
            "worth-keeping",
            identity=_GATE_IDENTITY,
            is_hard=True,
            declared_inputs=frozenset({StoreInput.INCUMBENTS, StoreInput.REJECTED_LOG}),
        ),
    )

    return FakeDomain(
        schema=schema,
        gates=gates,
        shape_validator=_validate_shape,
        run_gate=_run_gate,
    )


def _validate_shape(artifact: Artifact) -> ShapeError | None:
    """A ``Note`` must be a dict carrying a ``text`` field (the proposal-shape spec, §7.0)."""
    if artifact.type != NOTE:
        return None
    payload = artifact.payload
    if not isinstance(payload, dict) or "text" not in payload:
        return ShapeError("a Note payload must be an object with a 'text' field")
    return None


def _run_gate(gate: GateSpec, artifact: Artifact) -> GateVerdict | None:
    """Deterministic verdicts keyed off marker fields in the payload (stands in for a verifier).

    Markers (all optional) on a ``Note`` payload drive the outcome so tests can script any branch:
    ``reject`` → reject; ``defect`` → refine(that defect); ``keep=False`` → hard-gate reject.
    """
    payload = artifact.payload if isinstance(artifact.payload, dict) else {}

    if gate.name == "well-formed":
        if payload.get("reject"):
            return GateVerdict(VerdictKind.REJECT, "marked reject")
        defect = payload.get("defect")
        if defect:
            return GateVerdict(VerdictKind.REFINE, "marked defect", defects=(str(defect),))
        return GateVerdict(VerdictKind.ACCEPT, "well-formed")

    if gate.name == "worth-keeping":
        if payload.get("keep", True) is False:
            return GateVerdict(VerdictKind.REJECT, "not worth keeping", score=0.0)
        return GateVerdict(VerdictKind.ACCEPT, "worth keeping", score=1.0)

    return GateVerdict(VerdictKind.ACCEPT, f"unknown gate {gate.name!r} default-accepts in fake")


# -- the same marker logic as gate *plugins*, so the fake domain runs on the real verifier (§3.6) --


def build_fake_verifier() -> SdkVerifier:
    """An :class:`SdkVerifier` whose plugins reproduce :func:`_run_gate`'s marker semantics.

    Lets the deterministic fake domain drive the **real** verifier (not the stub) end-to-end, so the
    control-plane §13 demonstrations run against the product path while staying fully deterministic.
    """
    return SdkVerifier(
        plugins={
            "well-formed": deterministic_check(_well_formed_check),
            "worth-keeping": deterministic_check(_worth_keeping_check),
        }
    )


def _markers(request: VerifierRequest) -> dict[str, JSONValue]:
    payload = request.proposal.payload
    return payload if isinstance(payload, dict) else {}


def _well_formed_check(request: VerifierRequest) -> CheckOutcome:
    payload = _markers(request)
    if payload.get("reject"):
        return CheckOutcome(ok=False, rationale="marked reject")
    defect = payload.get("defect")
    if defect:
        return CheckOutcome(ok=False, rationale="marked defect", defects=(str(defect),))
    return CheckOutcome(ok=True, rationale="well-formed")


def _worth_keeping_check(request: VerifierRequest) -> CheckOutcome:
    if _markers(request).get("keep", True) is False:
        return CheckOutcome(ok=False, rationale="not worth keeping")
    return CheckOutcome(ok=True, rationale="worth keeping")
