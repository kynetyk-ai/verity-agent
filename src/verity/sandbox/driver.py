"""The framework/isolation seam (ROADMAP Phase 3).

A :class:`SandboxDriver` has one job: run an agent loop until it writes a proposal descriptor
(and any object attachments) into the workspace outbox. The host-side core
(:class:`~verity.sandbox.core.AgentSandbox`) then harvests and mints — so a driver returns nothing.

This is the single extension point for *additional sandbox configurations*: a different framework
(Deep Agents, pi-mono, a hand-rolled loop) or a different isolation strategy (in-process, container,
networked) is a new ``SandboxDriver`` honouring the descriptor contract — the core, the control
plane, and the registries are untouched.
"""

from __future__ import annotations

from typing import Protocol

from verity.control_plane.registries import OperationSignature
from verity.control_plane.workspace import ProvisionedWorkspace

__all__ = ["SandboxDriver"]


class SandboxDriver(Protocol):
    """Run the agent loop for one cycle; leave a proposal descriptor + attachments in the outbox."""

    async def run(
        self,
        *,
        system_prompt: str,
        user_message: str,
        operations: tuple[OperationSignature, ...],
        workspace: ProvisionedWorkspace,
    ) -> None: ...
