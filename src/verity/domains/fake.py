"""A trivial fake domain to prove the extension-point contracts (spec §8, §17.2; ADR 0001).

Not a real domain — it exercises the registries, the commit path, and the **opaque verifier** before
the feature-engineering domain (§12, Phase 4) lands. Deterministic (no LLM), and it includes a
**gateless type** so "no implicit accept" (§5.7) can be demonstrated: a type not registered as gated
cannot be committed.

Types:
* ``Source`` — a root (raw input) type.
* ``Note`` — the gated type. Its verifier runs a cheap "well-formed" check (→ ``tentative``) then a
  hard "worth-keeping" check (→ ``accepted``); the control plane declares it gated and grants it the
  incumbents + rejected-log slice (§8.3, §10).
* ``Orphan`` — a registered type with **no** gate (the no-implicit-accept probe).

The control plane only declares *that* ``Note`` is gated and *what slice* its verifier may see; the
pipeline itself lives in :func:`build_fake_verifier` (the opaque verifier package).
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.contracts import JSONValue, VerifierRequest
from verity.control_plane.commit import ShapeError, ShapeValidator
from verity.control_plane.independence import StoreInput
from verity.control_plane.registries import (
    ArtifactTypeDef,
    GatedTypeRegistry,
    OperationSignature,
    SchemaRegistry,
)
from verity.control_plane.store import Artifact
from verity.verifier import CheckOutcome, GateStep, SdkVerifier, deterministic_check

__all__ = [
    "SOURCE",
    "NOTE",
    "ORPHAN",
    "FAKE_VERIFIER_IDENTITY",
    "FakeDomain",
    "build_fake_domain",
    "build_fake_verifier",
    "declared_objects",
]

SOURCE = "Source"
NOTE = "Note"
ORPHAN = "Orphan"

# The verifier package's identity — distinct from any proposer, so proposer ≠ gate holds (§7.2).
FAKE_VERIFIER_IDENTITY = "fake-verifier"


@dataclass(frozen=True, slots=True)
class FakeDomain:
    """The control-plane side a task wires in: the schema, the gated-type coverage, the shape spec.

    The verifier side (the pipeline) is :func:`build_fake_verifier`, selected by ``verifier_key``.
    """

    schema: SchemaRegistry
    gated_types: GatedTypeRegistry
    shape_validator: ShapeValidator


def build_fake_domain() -> FakeDomain:
    schema = SchemaRegistry()
    schema.register_type(ArtifactTypeDef(SOURCE, is_root=True))
    schema.register_type(ArtifactTypeDef(NOTE))
    schema.register_type(ArtifactTypeDef(ORPHAN))
    schema.register_operation(OperationSignature("author", inputs=(SOURCE,), output=NOTE))
    schema.register_operation(OperationSignature("note_orphan", inputs=(SOURCE,), output=ORPHAN))

    gated_types = GatedTypeRegistry()
    # 'Note' is gated; its verifier may see the incumbents it must beat and the rejected-log (§8.3,
    # §10). 'Orphan' is left UNGATED → committing it is a NoImplicitAccept error (§5.7).
    gated_types.gate(
        NOTE, declared_inputs=frozenset({StoreInput.INCUMBENTS, StoreInput.REJECTED_LOG})
    )

    return FakeDomain(schema=schema, gated_types=gated_types, shape_validator=_validate_shape)


def build_fake_verifier() -> SdkVerifier:
    """The opaque verifier package for the fake domain — its pipeline (spec §3.6, ADR 0001).

    Deterministic, marker-driven plugins reproduce the classic fake semantics: a payload ``reject``
    marker rejects, ``defect`` refines, ``keep=False`` fails the hard check. Lets the fake domain
    drive the **real** verifier end-to-end while staying fully reproducible.
    """
    return SdkVerifier(
        identity=FAKE_VERIFIER_IDENTITY,
        pipelines={
            NOTE: (
                GateStep("well-formed", deterministic_check(_well_formed_check), is_hard=False),
                GateStep("worth-keeping", deterministic_check(_worth_keeping_check), is_hard=True),
            )
        },
    )


def _validate_shape(artifact: Artifact) -> ShapeError | None:
    """A ``Note`` must be a dict carrying a ``text`` field (the proposal-shape spec, §7.0)."""
    if artifact.type != NOTE:
        return None
    payload = artifact.payload
    if not isinstance(payload, dict) or "text" not in payload:
        return ShapeError("a Note payload must be an object with a 'text' field")
    return None


def declared_objects(artifact: Artifact) -> frozenset[str]:
    """A proposal's declared object names — read from an optional payload ``objects`` list.

    The generic test-double analogue of a real domain's declaration: the control plane harvests only
    these from the outbox. A proposal with no ``objects`` list declares none.
    """
    payload = artifact.payload
    if not isinstance(payload, dict):
        return frozenset()
    names = payload.get("objects")
    if not isinstance(names, list):
        return frozenset()
    return frozenset(name for name in names if isinstance(name, str))


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
