"""The backend-backed sandbox driver (ROADMAP 7.4.c / ADR 0003).

A `SandboxDriver` that runs one cycle as an ephemeral worker via a `WorkerBackend`, replacing the
``docker run``-baked `DeepAgentsContainerDriver`. It is the **substrate-agnostic** driver: it speaks
only the `WorkerBackend` seam, so the same driver runs on local Docker now and Kubernetes later.

It does the bytes file-transfer (`provisioning.backend`): the cycle input + the workspace's
read-only roles become ``readonly_inputs``, ``/work`` is a fresh writable area, and the agent's
outbox comes back as ``output_globs`` bytes — which the driver **bridges into the host workspace
outbox**, so `AgentSandbox`'s harvest/mint is untouched. The in-worker entrypoint
(`container_entry`) is reused unchanged. Gold-data isolation (§3.5) is by **ephemeral
regeneration**, not in-sandbox read-only: the worker is discarded each cycle and the verifier scores
against the control plane's own gold copy, so a corrupted input only makes a bad proposal.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

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
    effective_recursion_limit,
    effective_step_budget,
)
from verity.sandbox.descriptor import RESERVED_TELEMETRY_NAME, RESERVED_TRANSCRIPT_NAME
from verity.sandbox.errors import SandboxError
from verity.sandbox.log_transcript import parse_log_events, telemetry_from_turns
from verity.sandbox.model_spec import ModelSpec

__all__ = ["BackendSandboxDriver"]

log = get_logger("verity.sandbox.backend_driver")

# The sandbox worker's captured stderr carries the FULL agent transcript (logged, not file-written),
# so its output budget must clear a long run's turns — far above the 1 MB runaway-child default that
# bounds the verifier/code-runner. Generous; still a hard ceiling on host memory per cycle.
_SANDBOX_OUTPUT_CAP = 32_000_000

_READ_ONLY_ROLES = ("data", "context", "tools", "spec")
# Writable roles whose control-plane-provisioned content must also reach the worker. Before the
# worker runs, the writable `scratch` role holds only what the control plane materialized this turn:
# the durable objects from `ObjectProvisioningPolicy` (e.g. the prior accepted submission under
# `scratch/provided/` — the FE incumbent the agent builds on). The agent runs in the worker, so
# these must be carried in, else it never sees its incumbent and restarts from zero each cycle.
# Read-only is fine: it reads the prior script and writes a fresh one to the outbox, not editing it.
_PROVISIONED_ROLES = ("scratch",)
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
            recursion_limit=effective_recursion_limit(self.recursion_limit, self.step_budget),
            deadline_s=self.timeout_s,  # the soft wrap-up budget is the worker's hard timeout (5.2)
            tool_names=self.tool_names,
            step_budget=effective_step_budget(self.recursion_limit, self.step_budget),
        )
        worker_spec = self._worker_spec(cycle, workspace)
        log.info("sandbox_worker_run", image=self.image, ops=[s.name for s in operations])
        try:
            result = await self.backend.run_to_completion(worker_spec)
        except ProvisioningError as exc:
            # the worker could not be launched — infrastructure, not a verdict: a recoverable
            # cycle. No worker ever ran, so the host synthesizes the entire record (#125).
            raise SandboxError(
                f"could not launch the sandbox worker: {exc}",
                telemetry=_telemetry_bytes(
                    {"stop_reason": "launch_failure", "error": str(exc), "source": "host"}
                ),
            ) from exc
        transcript_bytes, telemetry_bytes, turns = self._bridge_outbox(result, workspace)
        if result.timed_out:
            # SIGKILLed mid-run: the worker never emitted its end-of-run telemetry event, so
            # nothing was there to salvage (#125). Rebuild the aggregate from the streamed turns
            # (real token/step sums where per-turn usage exists) and stamp WHY it stopped — the
            # host is the only party that knows. `source` says where the numbers came from.
            if telemetry_bytes is None:
                synthesized: dict[str, Any] = telemetry_from_turns(turns) or {"source": "host"}
                synthesized["stop_reason"] = "sandbox_timeout"
                synthesized["timeout_s"] = self.timeout_s
                telemetry_bytes = _telemetry_bytes(synthesized)
            raise SandboxError(
                f"sandbox worker timed out after {self.timeout_s}s",
                transcript=transcript_bytes, telemetry=telemetry_bytes,
            )
        if result.exit_code != 0:
            # A crash that reached run_agent's crash path already logged real telemetry (with a
            # `crash: …` stop_reason) — keep it. An abrupt death (OOM-kill, entrypoint failure)
            # left none: synthesize, as for the timeout.
            if telemetry_bytes is None:
                crash_synth: dict[str, Any] = telemetry_from_turns(turns) or {"source": "host"}
                crash_synth["stop_reason"] = "worker_crash"
                crash_synth["exit_code"] = result.exit_code
                telemetry_bytes = _telemetry_bytes(crash_synth)
            raise SandboxError(
                f"sandbox worker exited {result.exit_code}: {result.stderr[-2000:]}",
                transcript=transcript_bytes, telemetry=telemetry_bytes,
            )

    def _worker_spec(self, cycle: CycleInput, workspace: ProvisionedWorkspace) -> WorkerSpec:
        # The cycle input + the read-only roles ride in as `readonly_inputs`; /work is writable; the
        # agent's outbox comes back as bytes. Data rides via the `data` role (static_contents), not
        # host-path mounts -- so the spec carries no host paths (k8s-portable).
        readonly: dict[str, bytes] = {CONTAINER_INPUT_PATH: cycle.to_json()}
        for role in (*_READ_ONLY_ROLES, *_PROVISIONED_ROLES):
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
            output_cap=_SANDBOX_OUTPUT_CAP,
        )

    @staticmethod
    def _bridge_outbox(
        result: CompletedWorker, workspace: ProvisionedWorkspace
    ) -> tuple[bytes | None, bytes | None, list[dict[str, object]]]:
        """Bridge the worker's outputs into the host workspace outbox, so `AgentSandbox`'s harvest
        finds them (the agent wrote inside the now-gone worker).

        Two sources: (1) the agent's outbox *files* (the proposal descriptor + submission objects),
        returned as bytes; and (2) the **transcript + telemetry**, which the worker did NOT write as
        files (the agent shares its uid and could tamper) but **logged** to stderr — reconstructed
        here, host-side, after the container has exited, and written under the reserved names so the
        agent never had access to them. Reconstruction runs on every outcome (success, no-proposal,
        and a timed-out worker — its retained stderr still yields the turns streamed before the
        kill). Returns ``(transcript_bytes, worker_telemetry_bytes, turns)`` so the caller can
        synthesize telemetry for an abnormal outcome the worker never got to record (#125).
        """
        outbox = workspace.outbox()
        outbox.mkdir(parents=True, exist_ok=True)
        prefix = f"{CONTAINER_OUTBOX}/"
        for path, data in result.outputs.items():
            name = path[len(prefix):] if path.startswith(prefix) else os.path.basename(path)
            dest = outbox / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)

        turns, telemetry = parse_log_events(result.stderr)
        transcript_bytes = json.dumps(turns, default=str).encode("utf-8") if turns else None
        telemetry_bytes = (
            json.dumps(telemetry, default=str).encode("utf-8") if telemetry is not None else None
        )
        if transcript_bytes is not None:
            (outbox / RESERVED_TRANSCRIPT_NAME).write_bytes(transcript_bytes)
        if telemetry_bytes is not None:
            (outbox / RESERVED_TELEMETRY_NAME).write_bytes(telemetry_bytes)
        return transcript_bytes, telemetry_bytes, turns


def _telemetry_bytes(record: dict[str, Any]) -> bytes:
    """Serialize a host-side telemetry record to the ``__telemetry__.json`` byte shape."""
    return json.dumps(record, default=str).encode("utf-8")
