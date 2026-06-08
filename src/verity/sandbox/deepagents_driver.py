"""Deep Agents agent build + the in-process driver (ROADMAP Phase 3).

The Deep Agents / LangChain import is quarantined to this module and
:mod:`~verity.sandbox.container_entry` (both behind the mypy override). The agent is a general
coding agent in **YOLO mode** — Deep Agents' native coding toolset (file ops + ``execute`` shell +
planning + subagents), **no** permission prompts / human-in-the-loop, plus the generated propose
tools — whose only task-specific inputs come through the control plane (the system prompt, the
served context, and the operations from the schema). Isolation, not in-loop gating, is the boundary.

Shell execution is the **backend's** native ``execute``: the container entrypoint uses a
shell-capable backend (``LocalShellBackend``) so ``execute`` runs code in the isolated container.
The in-process driver uses a plain ``FilesystemBackend`` (file tools only — it is the dev/test
harness and must not run untrusted code on the host).

:func:`build_deepagents_agent` + :func:`run_agent` are the shared build/run used by both drivers, so
the tool-binding and descriptor contract are single-sourced. Each domain ``OperationSignature``
becomes one propose tool (name = op name) that writes the descriptor to the outbox; the host core
harvests and mints (see :mod:`verity.sandbox.descriptor`).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from verity.control_plane.registries import OperationSignature
from verity.control_plane.workspace import ProvisionedWorkspace
from verity.logging import get_logger
from verity.sandbox.descriptor import (
    RESERVED_PROPOSAL_NAME,
    RESERVED_TELEMETRY_NAME,
    ProposalDescriptor,
)
from verity.sandbox.errors import SandboxError
from verity.sandbox.tools import resolve_tools

__all__ = [
    "DeepAgentsInProcessDriver",
    "DeadlineMiddleware",
    "StepBudgetMiddleware",
    "StepBudgetExceeded",
    "build_deepagents_agent",
    "run_agent",
]

log = get_logger("verity.sandbox.deepagents_driver")

_DEFAULT_RECURSION_LIMIT = 80
_DEADLINE_WARN_FRACTION = 0.8  # warn the agent once it is this far into its time/step budget


class DeadlineMiddleware(AgentMiddleware):
    """A soft-deadline nudge (ROADMAP 5.2): inject a wrap-up message before the hard timeout kills.

    The hard sandbox timeout is the backstop; this biases the agent to *finalize* first. Once the
    run is ``warn_fraction`` of the way through ``deadline_s`` (wall-clock), ``before_model``
    injects a one-time wrap-up message telling the agent to submit. ``now`` injectable for tests.
    """

    def __init__(
        self,
        deadline_s: float,
        *,
        now: Callable[[], float] = time.monotonic,
        warn_fraction: float = _DEADLINE_WARN_FRACTION,
    ) -> None:
        super().__init__()
        self._deadline_s = deadline_s
        self._now = now
        self._warn_at = warn_fraction * deadline_s
        self._start: float | None = None
        self._warned = False

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self._start = self._now()
        return None

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if self._start is None:
            self._start = self._now()
        elapsed = self._now() - self._start
        if self._warned or elapsed < self._warn_at:
            return None
        self._warned = True
        remaining = max(0, int(self._deadline_s - elapsed))
        return {
            "messages": [
                HumanMessage(content=(
                    f"Time budget almost spent: ~{remaining}s of {int(self._deadline_s)}s left. "
                    "Stop exploring now — finalize your script, run it once to confirm it works, "
                    "and submit your best proposal immediately."
                ))
            ]
        }


class StepBudgetExceeded(RuntimeError):
    """The agent burned its per-cycle model-step budget without converging on a proposal."""


class StepBudgetMiddleware(AgentMiddleware):
    """A per-cycle model-step budget (ROADMAP 5.1): nudge to wrap up, then hard-stop if ignored.

    Each ``before_model`` is one model step. At ``warn_fraction`` of ``max_steps`` it injects a
    one-time wrap-up nudge (like :class:`DeadlineMiddleware`, but counting steps not wall-clock);
    once the budget is spent it raises :class:`StepBudgetExceeded`, which the drivers turn into a
    cycle-level ``SandboxError`` so a non-converging agent yields a *recorded, fed-back* failed
    cycle with a clear reason instead of an opaque ``GraphRecursionError`` at the framework limit.
    """

    def __init__(self, max_steps: int, *, warn_fraction: float = _DEADLINE_WARN_FRACTION) -> None:
        super().__init__()
        self._max = max_steps
        self._warn_at = max(1, int(warn_fraction * max_steps))
        self._steps = 0
        self._warned = False

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self._steps += 1
        if self._steps > self._max:
            raise StepBudgetExceeded(
                f"model-step budget exhausted ({self._max} steps); the agent did not converge on a "
                "proposal within its budget"
            )
        if not self._warned and self._steps >= self._warn_at:
            self._warned = True
            return {
                "messages": [
                    HumanMessage(content=(
                        f"Step budget almost spent: {self._steps} of {self._max} model steps used. "
                        "Stop exploring now — finalize and submit your best proposal immediately."
                    ))
                ]
            }
        return None


def build_deepagents_agent(
    *,
    model: Any,
    operations: tuple[OperationSignature, ...],
    outbox: Path,
    system_prompt: str,
    backend: Any,
    extra_tools: Sequence[Any] = (),
    middleware: Sequence[Any] = (),
    deadline_s: float | None = None,
    step_budget: int | None = None,
) -> Any:
    """Build the YOLO coding agent: Deep Agents' native tools + a propose tool per operation.

    ``backend`` decides the coding capability: a shell-capable backend (``LocalShellBackend``) gives
    a working ``execute``; a plain ``FilesystemBackend`` gives file tools only. ``outbox`` is where
    the propose tools write the descriptor. ``extra_tools`` are domain/task tools the adapter binds
    (§8.2 seam, #6). ``middleware`` forwards to ``create_deep_agent``; when ``deadline_s`` is set a
    :class:`DeadlineMiddleware` is appended, and when ``step_budget`` is set a
    :class:`StepBudgetMiddleware` is appended (ROADMAP 5.1). (Deep Agents auto-injects
    **summarization/compaction** middleware into its base stack, so compaction is already on — 5.2.)
    """
    tools = [_make_propose_tool(sig, outbox) for sig in operations] + list(extra_tools)
    stack = list(middleware)
    if deadline_s is not None:
        stack.append(DeadlineMiddleware(deadline_s))
    if step_budget is not None:
        stack.append(StepBudgetMiddleware(step_budget))
    # The framework is opaque to us: treat the compiled agent as Any so its overloaded `invoke`
    # bridges cleanly through asyncio.to_thread / a direct call.
    agent: Any = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        backend=backend,
        middleware=stack,
    )
    return agent


def run_agent(
    agent: Any,
    user_message: str,
    *,
    recursion_limit: int = _DEFAULT_RECURSION_LIMIT,
    outbox: Path | None = None,
) -> None:
    """Run the agent loop to completion (synchronous).

    When ``outbox`` is given, the loop's telemetry (tokens / steps / tool-calls / model) is written
    to the reserved ``__telemetry__.json`` there, harvested back to the host like the proposal
    descriptor (ROADMAP 5.3b). Best-effort: a model without ``usage_metadata`` yields zeros / nulls.
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        {"recursion_limit": recursion_limit},
    )
    if outbox is not None:
        telemetry = _extract_telemetry(result)
        (outbox / RESERVED_TELEMETRY_NAME).write_bytes(json.dumps(telemetry).encode("utf-8"))


def _extract_telemetry(result: Any) -> dict[str, Any]:
    """Sum token usage + count steps / tool-calls across the run's ``AIMessage``s (5.3b)."""
    messages = result.get("messages", []) if isinstance(result, dict) else []
    input_tokens = output_tokens = model_steps = tool_calls = 0
    model_name: str | None = None
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        model_steps += 1
        tool_calls += len(message.tool_calls or [])
        usage = message.usage_metadata
        if usage:
            input_tokens += int(usage.get("input_tokens", 0) or 0)
            output_tokens += int(usage.get("output_tokens", 0) or 0)
        meta = message.response_metadata or {}
        model_name = meta.get("model_name") or meta.get("model") or model_name
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "model_steps": model_steps,
        "tool_calls": tool_calls,
        "model": model_name,
    }


class DeepAgentsInProcessDriver:
    """Runs the Deep Agents loop in this process (tests/dev). ``model`` is a provider string/model.

    YOLO: no ``interrupt_on`` / HITL. Uses a plain ``FilesystemBackend`` (file tools, **no**
    working ``execute``) — in-process must not run untrusted code on the host, so this driver is for
    the fake-model offline tests and the code-execution-free live smoke; arbitrary-code work belongs
    in the container driver.
    """

    def __init__(
        self,
        *,
        model: Any,
        recursion_limit: int = _DEFAULT_RECURSION_LIMIT,
        tool_names: tuple[str, ...] = (),
        step_budget: int | None = None,
    ) -> None:
        self._model = model
        self._recursion_limit = recursion_limit
        self._tool_names = tool_names
        self._step_budget = step_budget

    async def run(
        self,
        *,
        system_prompt: str,
        user_message: str,
        operations: tuple[OperationSignature, ...],
        workspace: ProvisionedWorkspace,
    ) -> None:
        backend = FilesystemBackend(root_dir=workspace.root, virtual_mode=False)
        agent = build_deepagents_agent(
            model=self._model,
            operations=operations,
            outbox=workspace.outbox(),
            system_prompt=system_prompt,
            backend=backend,
            extra_tools=resolve_tools(self._tool_names),
            step_budget=self._step_budget,
        )
        log.info("deepagents_run", ops=[s.name for s in operations], root=str(workspace.root))
        try:
            await asyncio.to_thread(
                run_agent, agent, user_message,
                recursion_limit=self._recursion_limit, outbox=workspace.outbox(),
            )
        except Exception as exc:
            # A runaway loop (GraphRecursionError) or any in-loop failure becomes a cycle-level
            # SandboxError, so the control plane can skip the cycle rather than abort the run (#21).
            raise SandboxError(
                f"in-process agent loop failed: {type(exc).__name__}: {exc}"
            ) from exc


def _make_propose_tool(signature: OperationSignature, outbox: Path) -> Callable[..., str]:
    """Bind one :class:`OperationSignature` to a propose tool that writes the outbox descriptor."""
    op_name = signature.name
    out_type = signature.output
    in_types = ", ".join(signature.inputs) or "(none)"

    def propose(parents: list[str], payload: dict[str, Any], rationale: str = "") -> str:
        if not isinstance(payload, dict) or not payload:
            return "rejected: payload must be a non-empty JSON object"
        if not parents:
            return "rejected: parents must include the input artifact id(s) from your context"
        descriptor = ProposalDescriptor(
            op_name=op_name, parents=tuple(parents), payload=payload, metadata=rationale
        )
        (outbox / RESERVED_PROPOSAL_NAME).write_bytes(descriptor.to_json())
        return f"proposal recorded ({op_name} -> {out_type}); the harness will harvest and gate it"

    propose.__name__ = op_name
    propose.__qualname__ = op_name
    propose.__doc__ = (
        f"Propose a {out_type} via {op_name} (inputs: {in_types}). Call once, last.\n\n"
        f"Args:\n"
        f"    parents: ids of the input artifact(s) of type {in_types}, taken from your context.\n"
        f"    payload: the {out_type} payload as a JSON object (see domain instructions / spec).\n"
        f"    rationale: brief note on how/why (audit trail; never shown to the verifier)."
    )
    return propose
