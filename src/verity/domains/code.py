"""A minimal code-execution domain — real submitted code, gated by execution (§12 shape; ADR 0001).

The opaque verifier package for this domain judges a **real** object attachment: a cheap ``parses``
check (real ``ast.parse`` — no execution) then a hard ``runs-clean`` check that runs the code in
isolation via the auto-code-runner. The control plane only declares that ``Submission`` is gated;
the pipeline lives in the verifier (selected by ``verifier_key``). The runner is injected, so the
same domain runs against the fake runner (offline) or the real container runner (integration).

It is the smallest domain that **earns** its verdicts by parsing and execution — the Phase 2.4
vehicle before the full feature-engineering domain (§12, Phase 4).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from verity.contracts import Artifact, VerifierRequest
from verity.control_plane.commit import ShapeError, ShapeValidator
from verity.control_plane.registries import (
    ArtifactTypeDef,
    GatedTypeRegistry,
    OperationSignature,
    SchemaRegistry,
)
from verity.verifier import (
    CheckOutcome,
    CodeRunner,
    GateStep,
    SdkVerifier,
    auto_code_runner,
    deterministic_check,
)

__all__ = [
    "DATASET",
    "SUBMISSION",
    "ENTRYPOINT",
    "CODE_VERIFIER_IDENTITY",
    "CodeDomain",
    "build_code_domain",
    "declared_objects",
]

DATASET = "Dataset"
SUBMISSION = "Submission"
ENTRYPOINT = "submission.py"

# The verifier package's identity, distinct from any proposer (proposer ≠ gate, §7.2).
CODE_VERIFIER_IDENTITY = "code-verifier"


@dataclass(frozen=True, slots=True)
class CodeDomain:
    """The bundle a task wires in: the control-plane registrations + the opaque verifier package.

    ``schema`` / ``gated_types`` / ``shape_validator`` are the control-plane side; ``verifier`` is
    the selected verifier package (built over the injected runner), holding the internal pipeline.
    """

    schema: SchemaRegistry
    gated_types: GatedTypeRegistry
    shape_validator: ShapeValidator
    verifier: SdkVerifier


def build_code_domain(runner: CodeRunner) -> CodeDomain:
    schema = SchemaRegistry()
    schema.register_type(ArtifactTypeDef(DATASET, is_root=True))
    schema.register_type(ArtifactTypeDef(SUBMISSION))
    schema.register_operation(OperationSignature("submit", inputs=(DATASET,), output=SUBMISSION))

    # 'Submission' is gated; its checks judge the object attachment, not the store, so the declared
    # slice is empty (the verifier sees only the artifact under test, §8.3, §10).
    gated_types = GatedTypeRegistry()
    gated_types.gate(SUBMISSION, declared_inputs=frozenset())

    verifier = SdkVerifier(
        identity=CODE_VERIFIER_IDENTITY,
        pipelines={
            SUBMISSION: (
                GateStep("parses", deterministic_check(_parses), is_hard=False),
                GateStep(
                    "runs-clean", auto_code_runner(runner, entrypoint=ENTRYPOINT), is_hard=True
                ),
            )
        },
    )

    return CodeDomain(
        schema=schema,
        gated_types=gated_types,
        shape_validator=_validate_shape,
        verifier=verifier,
    )


def _validate_shape(artifact: Artifact) -> ShapeError | None:
    """A ``Submission`` payload must be an object with a string ``entrypoint`` (shape, §7.0)."""
    if artifact.type != SUBMISSION:
        return None
    payload = artifact.payload
    if not isinstance(payload, dict) or not isinstance(payload.get("entrypoint"), str):
        return ShapeError("a Submission payload must be an object with a string 'entrypoint'")
    return None


def declared_objects(artifact: Artifact) -> frozenset[str]:
    """The object a Submission declares — its ``entrypoint`` script (the only file harvested)."""
    if artifact.type != SUBMISSION or not isinstance(artifact.payload, dict):
        return frozenset()
    entrypoint = artifact.payload.get("entrypoint")
    return frozenset({entrypoint}) if isinstance(entrypoint, str) and entrypoint else frozenset()


def _parses(request: VerifierRequest) -> CheckOutcome:
    """The cheap check: the submitted code is present and parses as Python (no execution)."""
    code = request.objects.get(ENTRYPOINT)
    if code is None:
        return CheckOutcome(ok=False, rationale=f"no {ENTRYPOINT!r} attachment to judge")
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return CheckOutcome(ok=False, rationale=f"syntax error: {exc}", defects=(str(exc),))
    return CheckOutcome(ok=True, rationale="submission parses as Python")
