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
    "effective_step_budget",
]

# Fixed in-container mount points.
CONTAINER_WORKSPACE = "/work"
CONTAINER_OUTBOX = "/work/outbox"
CONTAINER_INPUT_DIR = "/sandbox"
CONTAINER_INPUT_PATH = "/sandbox/input.json"

_STEP_BUDGET_FRACTION = 0.8  # derived step budget as a fraction of the framework recursion limit


def effective_step_budget(recursion_limit: int, step_budget: int | None) -> int:
    """The per-cycle step budget to enforce, deriving a graceful default when none is set (#103).

    An explicit ``step_budget`` always wins. When it is ``None`` we derive one from the framework's
    ``recursion_limit`` so *every* cycle gets the soft "wrap up and submit" nudge + a **typed**
    ``StepBudgetExceeded`` stop *before* the framework's ungraceful ``GraphRecursionError`` — the
    always-on safety net (kept strictly below ``recursion_limit`` so it fires first). Lives here in
    the framework-neutral cycle-input module so the host-side drivers can derive it without
    importing the Deep Agents stack.
    """
    if step_budget is not None:
        return step_budget
    return max(1, int(_STEP_BUDGET_FRACTION * recursion_limit))


@dataclass(frozen=True, slots=True)
class CycleInput:
    """Everything the container needs to run one cycle (the model spec travels via env)."""

    system_prompt: str
    user_message: str
    operations: tuple[OperationSignature, ...]
    recursion_limit: int = 80
    deadline_s: float | None = None  # soft wrap-up budget (5.2); usually the container hard timeout
    tool_names: tuple[str, ...] = ()  # extra sandbox tools to bind, by registry name (5.4, #6)
    step_budget: int | None = None  # per-cycle model-step budget (5.1); None -> driver-derived

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
                "step_budget": self.step_budget,
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
            step_budget=obj.get("step_budget"),
        )
