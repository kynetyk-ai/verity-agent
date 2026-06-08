"""The host <-> container cycle-input protocol (ROADMAP Phase 3, Sprint 2).

The container driver writes one ``CycleInput`` JSON file and mounts it read-only; the in-container
entrypoint (:mod:`verity.sandbox.container_entry`) reads it and builds the agent. Framework-neutral
(no Deep Agents import) so both sides share one contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from verity.control_plane.registries import OperationSignature

__all__ = [
    "CONTAINER_WORKSPACE",
    "CONTAINER_OUTBOX",
    "CONTAINER_INPUT_DIR",
    "CONTAINER_INPUT_PATH",
    "CycleInput",
]

# Fixed in-container mount points.
CONTAINER_WORKSPACE = "/work"
CONTAINER_OUTBOX = "/work/outbox"
CONTAINER_INPUT_DIR = "/sandbox"
CONTAINER_INPUT_PATH = "/sandbox/input.json"


@dataclass(frozen=True, slots=True)
class CycleInput:
    """Everything the container needs to run one cycle (the model spec travels via env)."""

    system_prompt: str
    user_message: str
    operations: tuple[OperationSignature, ...]
    recursion_limit: int = 80
    deadline_s: float | None = None  # soft wrap-up budget (5.2); usually the container hard timeout
    tool_names: tuple[str, ...] = ()  # extra sandbox tools to bind, by registry name (5.4, #6)

    def to_json(self) -> bytes:
        return json.dumps(
            {
                "system_prompt": self.system_prompt,
                "user_message": self.user_message,
                "operations": [
                    {"name": s.name, "inputs": list(s.inputs), "output": s.output}
                    for s in self.operations
                ],
                "recursion_limit": self.recursion_limit,
                "deadline_s": self.deadline_s,
                "tool_names": list(self.tool_names),
            }
        ).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> CycleInput:
        obj = json.loads(data)
        operations = tuple(
            OperationSignature(name=o["name"], inputs=tuple(o["inputs"]), output=o["output"])
            for o in obj["operations"]
        )
        return cls(
            system_prompt=obj["system_prompt"],
            user_message=obj["user_message"],
            operations=operations,
            recursion_limit=int(obj.get("recursion_limit", 80)),
            deadline_s=obj.get("deadline_s"),
            tool_names=tuple(obj.get("tool_names", ())),
        )
