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
    "effective_recursion_limit",
]

# Fixed in-container mount points.
CONTAINER_WORKSPACE = "/work"
CONTAINER_OUTBOX = "/work/outbox"
CONTAINER_INPUT_DIR = "/sandbox"
CONTAINER_INPUT_PATH = "/sandbox/input.json"

# The graceful per-cycle MODEL-STEP budget (#103): a fixed default in model-step units. Deliberately
# NOT derived from recursion_limit, which counts LangGraph super-steps (model + tool + middleware
# nodes) — a different unit. Generous over a real run (the accepted FE run used ~53 model steps) but
# bounded, so a wandering agent gets nudged + stopped gracefully rather than crashing on recursion.
_DEFAULT_STEP_BUDGET = 80
# LangGraph super-steps per model turn (empirically ~5 for the Deep Agents stack): the headroom
# factor so the recursion backstop sits comfortably ABOVE the model-step budget (8 = ~60% margin).
_RECURSION_HEADROOM = 8


def effective_step_budget(recursion_limit: int, step_budget: int | None) -> int:
    """The per-cycle MODEL-STEP budget to enforce (#103).

    An explicit ``step_budget`` always wins; otherwise use a fixed graceful default in MODEL-STEP
    units. The companion :func:`effective_recursion_limit` then sizes the framework's super-step
    backstop ABOVE this, so the graceful ``StepBudgetExceeded`` fires before the framework's
    ``GraphRecursionError``. (``recursion_limit`` is accepted for signature symmetry with that
    companion; the default budget does not derive from it — the two are different units.)
    """
    if step_budget is not None:
        return step_budget
    return _DEFAULT_STEP_BUDGET


def effective_recursion_limit(recursion_limit: int, step_budget: int | None) -> int:
    """The framework recursion backstop (LangGraph super-steps), clamped UP only.

    Returns at least ``step_budget × _RECURSION_HEADROOM`` so the graceful model-step cap is always
    reached first — the safety-net ordering holds *by construction*, independent of the exact
    super-steps-per-turn constant. An explicitly-requested ``recursion_limit`` already above the
    floor is honored (raised, never lowered), so callers asking for more headroom keep it.
    """
    floor = effective_step_budget(recursion_limit, step_budget) * _RECURSION_HEADROOM
    return max(recursion_limit, floor)


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
                    {
                        "name": s.name,
                        "inputs": list(s.inputs),
                        "output": s.output,
                        "required_payload_keys": list(s.required_payload_keys),
                        "object_payload_keys": list(s.object_payload_keys),
                    }
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
            OperationSignature(
                name=o["name"], inputs=tuple(o["inputs"]), output=o["output"],
                required_payload_keys=tuple(o.get("required_payload_keys", ())),
                object_payload_keys=tuple(o.get("object_payload_keys", ())),
            )
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
