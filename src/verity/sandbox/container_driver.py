"""The Deep Agents container driver (ROADMAP Phase 3, Sprint 2): safe YOLO code execution.

A ``SandboxDriver`` that runs the agent loop inside a fresh ``docker run`` per cycle, so the coding
agent can write *and run* code with **no permission prompts** while the **container is the safety
boundary**. The posture mirrors ``ContainerCodeRunner`` (non-root host-uid, all caps dropped,
``no-new-privileges``, read-only root + a ``tmpfs``, memory/CPU/PID limits, timeout-kill) — but with
**outbound network enabled** (the agent must reach the model API) and a writable workspace mount.

Gold-data isolation is physical: ``/work`` is mounted writable, with the read-only roles
(data/context/tools/spec) re-mounted read-only on top, and the task's ``data_sources`` mounted
read-only into ``data/``. The agent writes its descriptor + attachments to ``/work/outbox``, which
the host harvests after the container exits. This driver imports **no** Deep Agents — it only
orchestrates the container whose entrypoint (:mod:`verity.sandbox.container_entry`) does.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from verity.control_plane.registries import OperationSignature
from verity.control_plane.workspace import ProvisionedWorkspace
from verity.logging import get_logger
from verity.sandbox.container_io import (
    CONTAINER_INPUT_DIR,
    CONTAINER_WORKSPACE,
    CycleInput,
    effective_recursion_limit,
    effective_step_budget,
)
from verity.sandbox.errors import SandboxError
from verity.sandbox.model_spec import ModelSpec

__all__ = ["DeepAgentsContainerDriver"]

log = get_logger("verity.sandbox.container_driver")

_READ_ONLY_ROLES = ("data", "context", "tools", "spec")
_ENTRYPOINT = ("python", "-m", "verity.sandbox.container_entry")


@dataclass(frozen=True, slots=True)
class DeepAgentsContainerDriver:
    """Runs the Deep Agents loop in a fresh, network-enabled, non-root container per cycle (YOLO).

    ``model`` is the provider string passed to the container (e.g. ``anthropic:claude-sonnet-4-6``);
    for a richer target (a custom ``base_url`` for a local / OpenAI-compatible endpoint) pass a
    :class:`~verity.sandbox.model_spec.ModelSpec` as ``spec`` instead — when absent, ``spec`` is
    derived from ``model`` (Phase 6). ``data_sources`` are host paths mounted read-only into
    ``/work/data``. ``env_passthrough`` names env vars (the model key) forwarded into the container;
    the spec's own key env var is forwarded automatically.
    """

    model: str
    spec: ModelSpec | None = None  # richer model target; derived from `model` when None (Phase 6)
    image: str = "verity-sandbox:latest"
    docker_bin: str = "docker"
    data_sources: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()  # extra sandbox tools to bind, by registry name (5.4, #6)
    env_passthrough: tuple[str, ...] = ("ANTHROPIC_API_KEY",)
    extra_run_args: tuple[str, ...] = ()
    memory: str = "4g"
    cpus: str = "2"
    pids_limit: int = 512
    tmpfs_size: str = "256m"
    timeout_s: float = 1800.0
    recursion_limit: int = 80
    step_budget: int | None = None  # per-cycle model-step budget (5.1); None = framework limit only

    def _spec(self) -> ModelSpec:
        return self.spec or ModelSpec.from_provider_string(self.model)

    async def run(
        self,
        *,
        system_prompt: str,
        user_message: str,
        operations: tuple[OperationSignature, ...],
        workspace: ProvisionedWorkspace,
    ) -> None:
        inputs_dir = Path(tempfile.mkdtemp(prefix="verity-sbx-in-"))
        try:
            cycle = CycleInput(
                system_prompt=system_prompt,
                user_message=user_message,
                operations=operations,
                recursion_limit=effective_recursion_limit(self.recursion_limit, self.step_budget),
                # The soft wrap-up deadline is the container's hard timeout: the agent is nudged to
                # finalize before the kill (5.2). The hard timeout stays the backstop.
                deadline_s=self.timeout_s,
                tool_names=self.tool_names,
                step_budget=effective_step_budget(self.recursion_limit, self.step_budget),
            )
            (inputs_dir / "input.json").write_bytes(cycle.to_json())
            os.chmod(inputs_dir, 0o755)
            name = f"verity-sbx-{workspace.root.name}"
            cmd = self.docker_command(workspace, inputs_dir, name=name)
            log.info("sandbox_container_run", image=self.image, ops=[s.name for s in operations])
            await self._run_container(cmd, name=name)
        finally:
            shutil.rmtree(inputs_dir, ignore_errors=True)

    def docker_command(
        self, workspace: ProvisionedWorkspace, inputs_dir: Path, *, name: str
    ) -> list[str]:
        """Build the ``docker run`` argv — the hostile-input posture, with outbound network."""
        cmd: list[str] = [
            self.docker_bin, "run", "--rm", "--name", name,
            "--read-only",
            f"--tmpfs=/tmp:rw,size={self.tmpfs_size},mode=1777",
            f"--memory={self.memory}", f"--memory-swap={self.memory}",
            f"--cpus={self.cpus}", f"--pids-limit={self.pids_limit}",
            "--cap-drop=ALL", "--security-opt=no-new-privileges",
            f"--user={os.getuid()}:{os.getgid()}",
            # /work writable; read-only roles re-mounted ro on top (gold-data isolation, physical).
            "-v", f"{workspace.root}:{CONTAINER_WORKSPACE}:rw",
        ]
        for role in _READ_ONLY_ROLES:
            cmd += ["-v", f"{workspace.path_for(role)}:{CONTAINER_WORKSPACE}/{role}:ro"]
        for src in self.data_sources:
            host = Path(src)
            cmd += ["-v", f"{host}:{CONTAINER_WORKSPACE}/data/{host.name}:ro"]
        cmd += ["-v", f"{inputs_dir}:{CONTAINER_INPUT_DIR}:ro"]
        spec = self._spec()
        # Forward the spec's own API-key env var (by name; the value stays in the daemon env) on top
        # of the configured passthrough, so a new provider's key rides without extra config.
        key_env = spec.key_env()
        passthrough = list(self.env_passthrough)
        if key_env and key_env not in passthrough:
            passthrough.append(key_env)
        for var in passthrough:
            if os.environ.get(var):
                cmd += ["-e", var]
        # Reach a model server on the Docker host (a local OpenAI-compatible endpoint, or Docker
        # Model Runner's model-runner.docker.internal) — only when the spec's base_url is a
        # *.docker.internal host, so the Anthropic/hosted paths stay as-is.
        if (gateway := spec.gateway_host) is not None:
            cmd += [f"--add-host={gateway}:host-gateway"]
        for key, value in spec.to_env().items():  # VERITY_SANDBOX_MODEL (+ BASE_URL/KEY_ENV/EXTRA)
            cmd += ["-e", f"{key}={value}"]
        cmd += [
            "-e", "HOME=/tmp",
            "-e", "PYTHONUNBUFFERED=1",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-w", CONTAINER_WORKSPACE,
            *self.extra_run_args,
            self.image,
            *_ENTRYPOINT,
        ]
        return cmd

    async def _run_container(self, cmd: Sequence[str], *, name: str) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except OSError as exc:
            # A missing/unreachable docker binary would otherwise raise a raw OSError that bypasses
            # the control plane's degrade-don't-crash catch; type it as a recoverable cycle (5.1).
            raise SandboxError(
                f"could not launch the sandbox container ({self.docker_bin}): {exc}"
            ) from exc
        try:
            _out, err = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
        except TimeoutError:
            await self._kill(name)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.communicate(), timeout=5.0)
            raise SandboxError(f"sandbox container timed out after {self.timeout_s}s") from None
        if proc.returncode != 0:
            tail = err.decode(errors="replace")[-2000:]
            raise SandboxError(f"sandbox container exited {proc.returncode}: {tail}")
        log.info("sandbox_container_exit", code=proc.returncode)

    async def _kill(self, name: str) -> None:
        with contextlib.suppress(Exception):
            killer = await asyncio.create_subprocess_exec(
                self.docker_bin, "kill", name,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), timeout=10.0)
