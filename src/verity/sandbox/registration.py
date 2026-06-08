"""Building + registering Deep Agents sandbox configurations (ROADMAP Phase 3).

:func:`build_deepagents_sandbox` assembles the framework-neutral core
(:class:`~verity.sandbox.core.AgentSandbox`) over a Deep Agents driver. **Additional sandbox
configurations are additive**: register more keys with a different ``model`` / ``driver`` (an open
model, a container driver, another framework) — each one closure; a task selects one by
``TaskConfig.sandbox_key``. Registration happens where a task is configured (it needs the domain's
schema, a workspace root, and a model), so the factory takes those explicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from verity.contracts import ProviderRegistry, SandboxPort
from verity.control_plane.registries import SchemaRegistry
from verity.sandbox.core import AgentSandbox
from verity.sandbox.deepagents_driver import DeepAgentsInProcessDriver

__all__ = ["build_deepagents_sandbox", "register_deepagents_sandbox"]


def build_deepagents_sandbox(
    *,
    schema: SchemaRegistry,
    root: Path,
    model: Any,
    proposer_identity: str,
    static_contents: Mapping[str, Mapping[str, bytes]] | None = None,
    clock: Callable[[], str] | None = None,
    id_source: Callable[[], str] | None = None,
) -> AgentSandbox:
    """Assemble an in-process Deep Agents :class:`AgentSandbox` for a domain's schema."""
    sandbox = AgentSandbox(
        root=root,
        driver=DeepAgentsInProcessDriver(model=model),
        schema=schema,
        proposer_identity=proposer_identity,
        static_contents=static_contents or {},
    )
    if clock is not None:
        sandbox.clock = clock
    if id_source is not None:
        sandbox.id_source = id_source
    return sandbox


def register_deepagents_sandbox(
    registry: ProviderRegistry[SandboxPort],
    key: str,
    **kwargs: Any,
) -> None:
    """Register a sandbox config under ``key`` (one line per additional configuration)."""
    registry.register(key, lambda: build_deepagents_sandbox(**kwargs))
