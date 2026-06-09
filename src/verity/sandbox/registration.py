"""Building + registering sandbox configurations (ROADMAP Phase 3).

:func:`build_deepagents_sandbox` (in-process) and :func:`build_container_sandbox` (isolated) build
the framework-neutral core (:class:`~verity.sandbox.core.AgentSandbox`) over a driver. **Additional
configurations are additive**: register more keys with a different ``model`` / ``driver`` (an open
model, the container driver, another framework) — each one closure; a task selects one by
``TaskConfig.sandbox_key``. Registration happens where a task is configured (it needs the domain's
schema and a workspace root), so the factories take those explicitly.

The in-process Deep Agents import is **lazy** (inside :func:`build_deepagents_sandbox`), so a
control-plane host that only orchestrates the *container* driver need not install the ``sandbox``
extra at all — the container image carries the framework.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from verity.contracts import ProviderRegistry, SandboxPort
from verity.control_plane.registries import SchemaRegistry
from verity.sandbox.container_driver import DeepAgentsContainerDriver
from verity.sandbox.core import AgentSandbox
from verity.sandbox.driver import SandboxDriver
from verity.sandbox.model_spec import ModelSpec

__all__ = [
    "build_sandbox",
    "build_deepagents_sandbox",
    "build_container_sandbox",
    "register_sandbox",
]


def build_sandbox(
    *,
    schema: SchemaRegistry,
    root: Path,
    driver: SandboxDriver,
    proposer_identity: str,
    static_contents: Mapping[str, Mapping[str, bytes]] | None = None,
    clock: Callable[[], str] | None = None,
    id_source: Callable[[], str] | None = None,
) -> AgentSandbox:
    """Assemble an :class:`AgentSandbox` over an explicit ``driver`` (the general factory)."""
    sandbox = AgentSandbox(
        root=root,
        driver=driver,
        schema=schema,
        proposer_identity=proposer_identity,
        static_contents=static_contents or {},
    )
    if clock is not None:
        sandbox.clock = clock
    if id_source is not None:
        sandbox.id_source = id_source
    return sandbox


def build_deepagents_sandbox(
    *,
    schema: SchemaRegistry,
    root: Path,
    model: Any,
    proposer_identity: str = "deepagents-inprocess",
    sandbox_tools: tuple[str, ...] = (),
    step_budget: int | None = None,
    **kwargs: Any,
) -> AgentSandbox:
    """An **in-process** Deep Agents sandbox (tests/dev). Requires the ``sandbox`` extra.

    ``sandbox_tools`` names extra (non-propose) tools the agent gets, resolved from the sandbox tool
    registry (5.4, #6). ``step_budget`` caps per-cycle model steps (5.1); None = framework limit.
    """
    from verity.sandbox.deepagents_driver import DeepAgentsInProcessDriver

    return build_sandbox(
        schema=schema,
        root=root,
        driver=DeepAgentsInProcessDriver(
            model=model, tool_names=sandbox_tools, step_budget=step_budget
        ),
        proposer_identity=proposer_identity,
        **kwargs,
    )


def build_container_sandbox(
    *,
    schema: SchemaRegistry,
    root: Path,
    model: str,
    model_spec: ModelSpec | None = None,
    image: str = "verity-sandbox:latest",
    data_sources: tuple[str, ...] = (),
    sandbox_tools: tuple[str, ...] = (),
    step_budget: int | None = None,
    proposer_identity: str | None = None,
    **kwargs: Any,
) -> AgentSandbox:
    """A **container-isolated** Deep Agents sandbox — safe YOLO arbitrary-code execution.

    ``model`` is the provider string; pass ``model_spec`` for a richer target (a local /
    OpenAI-compatible endpoint with a ``base_url``) — Phase 6. ``sandbox_tools`` names extra
    (non-propose) tools, resolved in-container from the tool registry (5.4, #6); the names ride
    ``CycleInput.tool_names``. ``step_budget`` caps per-cycle model steps (5.1; None = no cap).
    """
    spec = model_spec or ModelSpec.from_provider_string(model)
    driver = DeepAgentsContainerDriver(
        model=model, spec=model_spec, image=image, data_sources=data_sources,
        tool_names=sandbox_tools, step_budget=step_budget,
    )
    return build_sandbox(
        schema=schema,
        root=root,
        driver=driver,
        proposer_identity=proposer_identity or f"deepagents-container:{spec.provider_string()}",
        **kwargs,
    )


def register_sandbox(
    registry: ProviderRegistry[SandboxPort],
    key: str,
    factory: Callable[[], AgentSandbox],
) -> None:
    """Register a sandbox config under ``key`` (one line per additional configuration)."""
    registry.register(key, factory)
