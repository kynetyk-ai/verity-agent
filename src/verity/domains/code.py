"""A minimal code-execution domain — real submitted code, gated by execution (spec §11, §12 shape).

Unlike the marker-driven fake domain (:mod:`.fake`), this domain's gates judge a **real** object
attachment rather than a scripted payload flag:

* the cheap ``parses`` gate parses the submitted Python (structural validity — no execution), and
* the hard ``runs-clean`` gate runs it in isolation via the auto-code-runner (does it execute?).

It is the smallest domain that **earns** its verdicts by parsing and execution, the Phase 2.4
vehicle for driving genuine artifacts through read → propose → gate → commit before the full
feature-engineering domain (§12, Phase 4) lands. The runner is injected, so the same domain runs
against the deterministic :class:`~verity.verifier.FakeCodeRunner` (offline suite) or the real
container runner (integration).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from verity.contracts import Artifact, VerifierRequest
from verity.control_plane.commit import GateSpec, ShapeError, ShapeValidator
from verity.control_plane.registries import (
    ArtifactTypeDef,
    GateRegistry,
    OperationSignature,
    SchemaRegistry,
)
from verity.verifier import (
    CheckOutcome,
    CodeRunner,
    GatePrimitive,
    auto_code_runner,
    deterministic_check,
)

__all__ = [
    "DATASET",
    "SUBMISSION",
    "ENTRYPOINT",
    "CodeDomain",
    "build_code_domain",
]

DATASET = "Dataset"
SUBMISSION = "Submission"
ENTRYPOINT = "submission.py"

# A gate identity distinct from any proposer, so proposer ≠ gate holds (§7.2).
_GATE_IDENTITY = "code-verifier"


@dataclass(frozen=True, slots=True)
class CodeDomain:
    """The bundle a task wires in: the control-plane registrations + the verifier's gate plugins.

    ``schema`` / ``gates`` / ``shape_validator`` are the control-plane side (the binding,
    §8.1/§8.3); ``plugins`` is the verifier side (gate name → plugin, §3.6) over the runner.
    """

    schema: SchemaRegistry
    gates: GateRegistry
    shape_validator: ShapeValidator
    plugins: dict[str, GatePrimitive]


def build_code_domain(runner: CodeRunner) -> CodeDomain:
    schema = SchemaRegistry()
    schema.register_type(ArtifactTypeDef(DATASET, is_root=True))
    schema.register_type(ArtifactTypeDef(SUBMISSION))
    schema.register_operation(
        OperationSignature("submit", inputs=(DATASET,), output=SUBMISSION)
    )

    gates = GateRegistry()
    # Cheap structural gate (declares no store-inputs → sees only the artifact + its objects), then
    # the hard execution gate. Both judge the object attachment, not the store-slice (§8.3, §10).
    gates.bind(
        SUBMISSION,
        GateSpec("parses", identity=_GATE_IDENTITY, is_hard=False),
        GateSpec("runs-clean", identity=_GATE_IDENTITY, is_hard=True),
    )

    plugins: dict[str, GatePrimitive] = {
        "parses": deterministic_check(_parses),
        "runs-clean": auto_code_runner(runner, entrypoint=ENTRYPOINT),
    }

    return CodeDomain(
        schema=schema, gates=gates, shape_validator=_validate_shape, plugins=plugins
    )


def _validate_shape(artifact: Artifact) -> ShapeError | None:
    """A ``Submission`` payload must be an object with a string ``entrypoint`` (shape, §7.0)."""
    if artifact.type != SUBMISSION:
        return None
    payload = artifact.payload
    if not isinstance(payload, dict) or not isinstance(payload.get("entrypoint"), str):
        return ShapeError("a Submission payload must be an object with a string 'entrypoint'")
    return None


def _parses(request: VerifierRequest) -> CheckOutcome:
    """The cheap gate: the submitted code is present and is syntactically valid Python (no exec)."""
    code = request.objects.get(ENTRYPOINT)
    if code is None:
        return CheckOutcome(ok=False, rationale=f"no {ENTRYPOINT!r} attachment to judge")
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return CheckOutcome(ok=False, rationale=f"syntax error: {exc}", defects=(str(exc),))
    return CheckOutcome(ok=True, rationale="submission parses as Python")
