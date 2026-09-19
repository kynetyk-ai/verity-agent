"""Published task-type contracts (ROADMAP 8.2, ADR 0004 (c)) — registry self-description.

A :class:`TaskTypeDescription` is what a client (a human or Claude) reads to judge fit *before*
creating a task: the **output-shape contract** the agent must produce (the domain's artifact types +
operation signatures with their required payload keys — the definable, formal half, sourced from the
same schema that renders the composed system prompt) and the **approval semantics** the verifier
applies (human-readable prose: the gate pipeline, cheap-vs-hard staging, what acceptance means — the
fuzzy half). We trust the user to select a sensible pairing; a poor one fails gracefully at run time
(degrade-don't-crash), not at selection.

Pure data (no domain / control-plane imports), so the catalog and the builders can both depend on it
without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["OperationDescription", "TaskTypeDescription"]


@dataclass(frozen=True, slots=True)
class OperationDescription:
    """One operation the agent may propose: its inputs, output type, and required payload keys."""

    name: str
    inputs: tuple[str, ...]
    output: str
    required_payload_keys: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": list(self.inputs),
            "output": self.output,
            "required_payload_keys": list(self.required_payload_keys),
        }


@dataclass(frozen=True, slots=True)
class TaskTypeDescription:
    """A task type's published contract: the shape the agent produces + the approval semantics."""

    type_name: str
    artifact_types: tuple[str, ...]
    gated_types: tuple[str, ...]
    operations: tuple[OperationDescription, ...]
    domain_instructions: str
    verifier_approach: str
    sandbox_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type_name": self.type_name,
            "artifact_types": list(self.artifact_types),
            "gated_types": list(self.gated_types),
            "operations": [o.to_dict() for o in self.operations],
            "domain_instructions": self.domain_instructions,
            "verifier_approach": self.verifier_approach,
            "sandbox_notes": self.sandbox_notes,
        }
