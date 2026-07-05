"""Building + registering sandbox configurations (ROADMAP Phase 3).

:func:`build_sandbox` assembles the framework-neutral core
(:class:`~verity.sandbox.core.AgentSandbox`) over an explicit driver — production task types hand
it a :class:`~verity.sandbox.backend_driver.BackendSandboxDriver` (an isolated worker on a
``WorkerBackend``); :func:`build_deepagents_sandbox` is the in-process convenience for tests/dev.
**Additional configurations are additive**: register more keys with a different ``model`` /
``driver`` (an open model, another framework) — each one closure; a task selects one by
``TaskConfig.sandbox_key``. Registration happens where a task is configured (it needs the domain's
schema and a workspace root), so the factories take those explicitly.

The in-process Deep Agents import is **lazy** (inside :func:`build_deepagents_sandbox`), so a
control-plane host that only orchestrates isolated workers need not install the ``sandbox`` extra
at all — the worker image carries the framework. (The legacy per-cycle ``docker run`` driver and
its ``build_container_sandbox`` factory were removed in #129; ``BackendSandboxDriver`` is the one
container path.)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from verity.contracts import ProviderRegistry, SandboxPort
from verity.control_plane.registries import SchemaRegistry
from verity.sandbox.core import AgentSandbox
from verity.sandbox.driver import SandboxDriver

__all__ = [
    "build_sandbox",
    "build_deepagents_sandbox",
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


def register_sandbox(
    registry: ProviderRegistry[SandboxPort],
    key: str,
    factory: Callable[[], AgentSandbox],
) -> None:
    """Register a sandbox config under ``key`` (one line per additional configuration)."""
    registry.register(key, factory)
