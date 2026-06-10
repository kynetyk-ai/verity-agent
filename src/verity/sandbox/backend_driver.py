"""The backend-backed sandbox driver (ROADMAP 7.4.c / ADR 0003).

A `SandboxDriver` that runs one cycle as an ephemeral worker via a `WorkerBackend`, replacing the
``docker run``-baked `DeepAgentsContainerDriver`. It is the **substrate-agnostic** driver: it speaks
only the `WorkerBackend` seam, so the same driver runs on local Docker now and Kubernetes later.

It does the bytes file-transfer (`provisioning.backend`): the cycle input + the workspace's
read-only roles become ``readonly_inputs`` (so §3.5 gold-data isolation stays physical), ``/work``
is a fresh writable area, and the agent's outbox comes back as ``output_globs`` bytes — which the
driver **bridges into the host workspace outbox**, so `AgentSandbox`'s harvest/mint is untouched.
The in-worker entrypoint (`container_entry`) is reused unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from verity.contracts.run_context import RunContext
from verity.control_plane.registries import OperationSignature
from verity.control_plane.workspace import ProvisionedWorkspace
from verity.logging import get_logger
from verity.provisioning.backend import (
    CompletedWorker,
    Labels,
    ProvisioningError,
    ResourceLimits,
    Tmpfs,
    WorkerBackend,
    WorkerSpec,
)
from verity.sandbox.container_io import (
    CONTAINER_INPUT_PATH,
    CONTAINER_OUTBOX,
    CONTAINER_WORKSPACE,
    CycleInput,
)
from verity.sandbox.errors import SandboxError
from verity.sandbox.model_spec import ModelSpec

__all__ = ["BackendSandboxDriver"]

log = get_logger("verity.sandbox.backend_driver")

_READ_ONLY_ROLES = ("data", "context", "tools", "spec")
_ENTRYPOINT = ("python", "-m", "verity.sandbox.container_entry")


@dataclass(slots=True)
class BackendSandboxDriver:
    """Runs the agent loop for one cycle as a `WorkerBackend` worker (the agent writes inside the
    worker; this driver bridges the outbox bytes back to the host workspace).

    ``backend`` is the substrate (Docker now, k8s later). ``model``/``spec`` name the model target
    (forwarded to the worker via env, like the container driver). ``runtime`` selects the OCI
    runtime (e.g. ``runsc``) per worker. The control plane binds a `RunContext` (7.4.h /
    :class:`~verity.contracts.run_context.SupportsRunContext`), so every worker is **labelled** with
    its ``tenant``/``run``/``cycle`` for audit + reaping; before a run binds one, labels carry only
    ``role``/``config``.
    """

    backend: WorkerBackend
    model: str = "anthropic:claude-sonnet-4-6"
    spec: ModelSpec | None = None
    image: str = "verity-sandbox:latest"
    config: str = ""  # the registry config-key, for the worker label
    tool_names: tuple[str, ...] = ()
    env_passthrough: tuple[str, ...] = ("ANTHROPIC_API_KEY",)
    memory: str = "4g"
    cpus: str = "2"
    pids_limit: int = 512
    tmpfs_size: str = "256m"
    timeout_s: float = 1800.0
    recursion_limit: int = 80
    step_budget: int | None = None
    runtime: str | None = None
    run_context: RunContext | None = None

    def bind_run_context(self, ctx: RunContext) -> None:
        """Take the control plane's shared run-identity cell; we stamp it onto every worker."""
        self.run_context = ctx

    def _spec(self) -> ModelSpec:
        return self.spec or ModelSpec.from_provider_string(self.model)

    def _labels(self) -> Labels:
        ctx = self.run_context
        if ctx is None:
            return Labels(role="sandbox", config=self.config)
        return Labels(
            role="sandbox", config=self.config,
            tenant=ctx.tenant_id, run=ctx.run_id, cycle=str(ctx.cycle),
        )

    async def run(
        self,
        *,
        system_prompt: str,
        user_message: str,
        operations: tuple[OperationSignature, ...],
        workspace: ProvisionedWorkspace,
    ) -> None:
        cycle = CycleInput(
            system_prompt=system_prompt,
            user_message=user_message,
            operations=operations,
            recursion_limit=self.recursion_limit,
            deadline_s=self.timeout_s,  # the soft wrap-up budget is the worker's hard timeout (5.2)
            tool_names=self.tool_names,
            step_budget=self.step_budget,
        )
        worker_spec = self._worker_spec(cycle, workspace)
        log.info("sandbox_worker_run", image=self.image, ops=[s.name for s in operations])
        try:
            result = await self.backend.run_to_completion(worker_spec)
        except ProvisioningError as exc:
            # the worker could not be launched — infrastructure, not a verdict: a recoverable cycle.
            raise SandboxError(f"could not launch the sandbox worker: {exc}") from exc
        self._bridge_outbox(result, workspace)
        if result.timed_out:
            raise SandboxError(f"sandbox worker timed out after {self.timeout_s}s")
        if result.exit_code != 0:
            raise SandboxError(
                f"sandbox worker exited {result.exit_code}: {result.stderr[-2000:]}"
            )

    def _worker_spec(self, cycle: CycleInput, workspace: ProvisionedWorkspace) -> WorkerSpec:
        # The cycle input + the ro roles become physically-ro inputs (§3.5); /work is writable; the
        # agent's outbox comes back as bytes. Data rides via the `data` role (static_contents), not
        # host-path mounts -- so the spec carries no host paths (k8s-portable).
        readonly: dict[str, bytes] = {CONTAINER_INPUT_PATH: cycle.to_json()}
        for role in _READ_ONLY_ROLES:
            role_dir = workspace.path_for(role)
            for path in sorted(role_dir.rglob("*")):
                if path.is_file():
                    rel = path.relative_to(workspace.root).as_posix()
                    readonly[f"{CONTAINER_WORKSPACE}/{rel}"] = path.read_bytes()

        spec = self._spec()
        env = dict(spec.to_env())  # VERITY_SANDBOX_MODEL (+ BASE_URL/KEY_ENV/EXTRA)
        env["PYTHONUNBUFFERED"] = "1"
        passthrough = list(self.env_passthrough)
        key_env = spec.key_env()
        if key_env and key_env not in passthrough:
            passthrough.append(key_env)
        extra_hosts: dict[str, str] = {}
        if (gateway := spec.gateway_host) is not None:  # reach a host-local model server
            extra_hosts[gateway] = "host-gateway"

        return WorkerSpec(
            image=self.image,
            command=_ENTRYPOINT,
            labels=self._labels(),
            readonly_inputs=readonly,
            writable_dirs=(CONTAINER_WORKSPACE,),
            output_globs=(f"{CONTAINER_OUTBOX}/*",),
            network=True,  # the agent must reach the model API
            env=env,
            env_passthrough=tuple(passthrough),
            extra_hosts=extra_hosts,
            limits=ResourceLimits(memory=self.memory, cpus=self.cpus, pids=self.pids_limit),
            scratch=Tmpfs(size=self.tmpfs_size),
            workdir=CONTAINER_WORKSPACE,
            runtime=self.runtime,
            timeout_s=self.timeout_s,
        )

    @staticmethod
    def _bridge_outbox(result: CompletedWorker, workspace: ProvisionedWorkspace) -> None:
        """Write the worker's outbox bytes into the host workspace outbox, so `AgentSandbox`'s
        harvest finds them (the agent wrote inside the worker, not the host workspace)."""
        outbox = workspace.outbox()
        outbox.mkdir(parents=True, exist_ok=True)
        prefix = f"{CONTAINER_OUTBOX}/"
        for path, data in result.outputs.items():
            name = path[len(prefix):] if path.startswith(prefix) else os.path.basename(path)
            dest = outbox / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
