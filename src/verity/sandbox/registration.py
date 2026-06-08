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
    **kwargs: Any,
) -> AgentSandbox:
    """An **in-process** Deep Agents sandbox (tests/dev). Requires the ``sandbox`` extra."""
    from verity.sandbox.deepagents_driver import DeepAgentsInProcessDriver

    return build_sandbox(
        schema=schema,
        root=root,
        driver=DeepAgentsInProcessDriver(model=model),
        proposer_identity=proposer_identity,
        **kwargs,
    )


def build_container_sandbox(
    *,
    schema: SchemaRegistry,
    root: Path,
    model: str,
    image: str = "verity-sandbox:latest",
    data_sources: tuple[str, ...] = (),
    proposer_identity: str | None = None,
    **kwargs: Any,
) -> AgentSandbox:
    """A **container-isolated** Deep Agents sandbox — safe YOLO arbitrary-code execution."""
    driver = DeepAgentsContainerDriver(model=model, image=image, data_sources=data_sources)
    return build_sandbox(
        schema=schema,
        root=root,
        driver=driver,
        proposer_identity=proposer_identity or f"deepagents-container:{model}",
        **kwargs,
    )


def register_sandbox(
    registry: ProviderRegistry[SandboxPort],
    key: str,
    factory: Callable[[], AgentSandbox],
) -> None:
    """Register a sandbox config under ``key`` (one line per additional configuration)."""
    registry.register(key, factory)
