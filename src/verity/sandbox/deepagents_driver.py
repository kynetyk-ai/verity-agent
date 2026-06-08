"""Deep Agents agent build + the in-process driver (ROADMAP Phase 3).

The Deep Agents / LangChain import is quarantined to this module and
:mod:`~verity.sandbox.container_entry` (both behind the mypy override). The agent is a general
coding agent in **YOLO mode** — the full coding toolset plus a ``run_shell`` execution tool, and
**no** permission prompts / human-in-the-loop — whose only task-specific inputs come through the
control plane (the system prompt, the served context, and the operation tools generated from the
schema). Isolation, not in-loop gating, is the safety boundary.

:func:`build_deepagents_agent` + :func:`run_agent` are the shared build/run used by both the
**in-process** driver (here) and the **container** entrypoint, so the tool-binding and descriptor
contract are single-sourced. Each domain ``OperationSignature`` becomes one propose tool (name = op
name) that writes the descriptor to the outbox; the host core harvests and mints
(see :mod:`verity.sandbox.descriptor`).
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

from verity.control_plane.registries import OperationSignature
from verity.control_plane.workspace import ProvisionedWorkspace
from verity.logging import get_logger
from verity.sandbox.descriptor import RESERVED_PROPOSAL_NAME, ProposalDescriptor

__all__ = ["DeepAgentsInProcessDriver", "build_deepagents_agent", "run_agent"]

log = get_logger("verity.sandbox.deepagents_driver")

_RUN_SHELL_TIMEOUT_S = 1500
_MAX_TOOL_OUTPUT = 4000
_DEFAULT_RECURSION_LIMIT = 80


def build_deepagents_agent(
    *,
    model: Any,
    operations: tuple[OperationSignature, ...],
    root: Path,
    outbox: Path,
    system_prompt: str,
) -> Any:
    """Build the YOLO coding agent: propose-tool-per-operation + ``run_shell``, no HITL.

    Shared by the in-process driver and the container entrypoint. ``root`` is the agent's filesystem
    backend root; ``outbox`` is where the propose tools write the descriptor.
    """
    tools = [_make_propose_tool(sig, outbox) for sig in operations]
    tools.append(_make_run_shell_tool(root))
    # The framework is opaque to us: treat the compiled agent as Any so its overloaded `invoke`
    # bridges cleanly through asyncio.to_thread / a direct call.
    agent: Any = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        backend=FilesystemBackend(root_dir=root, virtual_mode=False),
    )
    return agent


def run_agent(
    agent: Any, user_message: str, *, recursion_limit: int = _DEFAULT_RECURSION_LIMIT
) -> None:
    """Run the agent loop to completion (synchronous)."""
    agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        {"recursion_limit": recursion_limit},
    )


class DeepAgentsInProcessDriver:
    """Runs the Deep Agents loop in this process (tests/dev). ``model`` is a provider string/model.

    YOLO: no ``interrupt_on`` / HITL middleware — the agent runs its tools freely. In-process runs
    untrusted code on the host, so this driver is for the fake-model offline tests and the
    code-execution-free live smoke; arbitrary-code work belongs in the container driver.
    """

    def __init__(self, *, model: Any, recursion_limit: int = _DEFAULT_RECURSION_LIMIT) -> None:
        self._model = model
        self._recursion_limit = recursion_limit

    async def run(
        self,
        *,
        system_prompt: str,
        user_message: str,
        operations: tuple[OperationSignature, ...],
        workspace: ProvisionedWorkspace,
    ) -> None:
        agent = build_deepagents_agent(
            model=self._model,
            operations=operations,
            root=workspace.root,
            outbox=workspace.outbox(),
            system_prompt=system_prompt,
        )
        log.info("deepagents_run", ops=[s.name for s in operations], root=str(workspace.root))
        await asyncio.to_thread(
            run_agent, agent, user_message, recursion_limit=self._recursion_limit
        )


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


def _make_run_shell_tool(root: Path) -> Callable[[str], str]:
    """The general-purpose code-execution tool (YOLO). Runs in the workspace; output truncated."""

    def run_shell(command: str) -> str:
        """Run a shell command in the workspace to inspect data or run code.

        Args:
            command: the shell command to execute.
        """
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=_RUN_SHELL_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return f"TIMEOUT after {_RUN_SHELL_TIMEOUT_S}s"
        out = (proc.stdout or "") + (f"\n[stderr]\n{proc.stderr}" if proc.stderr else "")
        if len(out) > _MAX_TOOL_OUTPUT:
            half = _MAX_TOOL_OUTPUT // 2
            out = out[:half] + "\n...[truncated]...\n" + out[-half:]
        return f"exit={proc.returncode}\n{out}"

    return run_shell
