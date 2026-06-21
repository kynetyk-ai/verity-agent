"""The code-runner seam — executing object attachments in isolation (spec §3.6, §11, §12).

The auto-code-runner is the rung where a gate must *execute* material rather than merely read it
(§3.6): in the v1 domain it receives the submitted code and the gate's datasets, runs the pipeline,
and lets the gate score the result (§12). Running an agent-proposed program is the hostile-input
case, so the real runner is **container-isolated**, and the execution mechanism sits behind a narrow
:class:`CodeRunner` port so:

* the unit suite runs offline and deterministically against :class:`FakeCodeRunner`, and
* the real :class:`ContainerCodeRunner` (Docker) is exercised only by a Docker-marked integration
  test (auto-skipped when Docker is absent), so CI needs no Docker-in-CI.

The runner shuttles **bytes in, bytes out** and knows nothing about artifact types or scoring — the
gate's :func:`~verity.verifier.primitives.auto_code_runner` plugin interprets the result (§8.3).
This is also a concrete instance of the data plane: code + datasets in, a result file out, with no
standing container per request (the networked successor is tracked as issue #3).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from verity.contracts import GateUnavailable
from verity.contracts.run_context import RunContext
from verity.logging import get_logger
from verity.provisioning.backend import (
    Labels,
    ProvisioningError,
    ResourceLimits,
    Tmpfs,
    WorkerBackend,
    WorkerSpec,
)

__all__ = [
    "RunRequest",
    "RunResult",
    "RunScript",
    "CodeRunner",
    "FakeCodeRunner",
    "ContainerCodeRunner",
    "BackendCodeRunner",
    "docker_available",
]

log = get_logger("verity.verifier.code_runner")

# The file name a declared package list is written to inside the work dir (Phase 4 deps).
_REQUIREMENTS_NAME = "requirements.txt"


@dataclass(frozen=True, slots=True)
class RunRequest:
    """What to execute in isolation (spec §3.6).

    ``code`` is written as ``entrypoint`` into a read-only work dir and run with ``python``;
    ``inputs`` are read-only datasets mounted alongside (name → bytes); the script is expected to
    write its result to ``output_name`` in a writable output dir, whose bytes come back on the
    result. ``timeout_s`` bounds the run.

    ``requirements`` (a ``requirements.txt`` body) and ``network`` support the feature-engineering
    domain's deps decision (Phase 4): when ``requirements`` is given they are ``pip install``-ed
    into a writable tmpfs before the script runs, which needs ``network=True`` (outbound enabled).
    This is a deliberately weaker posture than the default no-network run — pinned versions keep it
    reproducible (§5.8). ``env`` are extra environment variables for the run (e.g. where the script
    finds its data). Defaults preserve the original no-network, no-deps behaviour exactly.
    """

    code: bytes
    entrypoint: str = "submission.py"
    inputs: Mapping[str, bytes] = field(default_factory=dict)
    output_name: str = "result.json"
    timeout_s: float = 30.0
    requirements: bytes | None = None
    network: bool = False
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunResult:
    """The outcome of one isolated run (spec §3.6).

    ``output`` is the bytes the script wrote to the declared output file, or ``None`` if it wrote
    none. The runner makes no judgment — interpreting this into a verdict is the gate's job (§8.3).
    """

    exit_code: int
    stdout: str
    stderr: str
    output: bytes | None = None
    timed_out: bool = False


# A fake runner's scripted behaviour: request -> result.
RunScript = Callable[[RunRequest], RunResult]


@runtime_checkable
class CodeRunner(Protocol):
    """The narrow port the auto-code-runner gate depends on (spec §3.6). Bytes in, bytes out."""

    async def run(self, request: RunRequest, /) -> RunResult: ...


def _ran_clean(_request: RunRequest) -> RunResult:
    return RunResult(exit_code=0, stdout="", stderr="")


@dataclass
class FakeCodeRunner:
    """A deterministic, in-process :class:`CodeRunner` for the suite (NON-PRODUCT).

    ``script`` is a pure rule from :class:`RunRequest` to :class:`RunResult`, so a test drives any
    outcome (clean run, non-zero exit, timeout, a written output file) without a container;
    ``calls`` records each request for assertions.
    """

    script: RunScript = _ran_clean
    calls: list[RunRequest] = field(default_factory=list)

    async def run(self, request: RunRequest, /) -> RunResult:
        self.calls.append(request)
        return self.script(request)


@dataclass(frozen=True, slots=True)
class ContainerCodeRunner:
    """A container-isolated :class:`CodeRunner` (Docker), for running agent-proposed code (§3.6).

    Each run is a fresh ``docker run`` with a hostile-input posture: no network, a read-only root
    filesystem, a small ``tmpfs`` for scratch, memory / CPU / PID limits, all capabilities dropped,
    ``no-new-privileges``, and a non-root user (the host uid, so mounted files round-trip cleanly).
    Code and datasets are mounted **read-only**; only the output dir is writable. The timeout is
    enforced by killing the container. Integration-only — **not exercised by the unit suite**.
    """

    image: str = "python:3.12-slim"
    docker_bin: str = "docker"
    memory: str = "512m"
    cpus: str = "1"
    pids_limit: int = 128
    tmpfs_size: str = "64m"

    async def run(self, request: RunRequest, /) -> RunResult:
        host = Path(tempfile.mkdtemp(prefix="verity-run-"))
        name = host.name
        try:
            code_dir, data_dir, out_dir = host / "code", host / "data", host / "out"
            for d in (code_dir, data_dir, out_dir):
                d.mkdir()
            (code_dir / request.entrypoint).write_bytes(request.code)
            if request.requirements is not None:
                (code_dir / _REQUIREMENTS_NAME).write_bytes(request.requirements)
            for input_name, blob in request.inputs.items():
                (data_dir / input_name).write_bytes(blob)
            out_dir.chmod(0o777)  # the non-root container user must be able to write here

            cmd = self._docker_cmd(name, code_dir, data_dir, out_dir, request)
            result = await self._run_container(cmd, name, request.timeout_s)

            output_path = out_dir / request.output_name
            output = output_path.read_bytes() if output_path.is_file() else None
            log.info(
                "code_run",
                entrypoint=request.entrypoint,
                exit_code=result.exit_code,
                timed_out=result.timed_out,
                produced_output=output is not None,
            )
            return RunResult(
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                output=output,
                timed_out=result.timed_out,
            )
        finally:
            shutil.rmtree(host, ignore_errors=True)

    def _docker_cmd(
        self, name: str, code_dir: Path, data_dir: Path, out_dir: Path, request: RunRequest
    ) -> list[str]:
        cmd = [self.docker_bin, "run", "--rm", "--name", name]
        if not request.network:
            cmd.append("--network=none")  # the default hostile-input posture (§3.6)
        # The deps path installs native wheels (e.g. numpy) into the tmpfs and must mmap their
        # ``.so`` files, so /tmp needs ``exec``; the strict no-deps path keeps the hardened default.
        tmpfs_opts = f"rw,size={self.tmpfs_size},mode=1777"
        if request.requirements is not None:
            tmpfs_opts += ",exec"
        cmd += [
            "--read-only",
            f"--tmpfs=/tmp:{tmpfs_opts}",
            f"--memory={self.memory}", f"--memory-swap={self.memory}",
            f"--cpus={self.cpus}", f"--pids-limit={self.pids_limit}",
            "--cap-drop=ALL", "--security-opt=no-new-privileges",
            f"--user={os.getuid()}:{os.getgid()}",
            "-v", f"{code_dir}:/work:ro",
            "-v", f"{data_dir}:/data:ro",
            "-v", f"{out_dir}:/out:rw",
            # CWD is the writable (ephemeral, tmpfs) /tmp, NOT the read-only /work mount: libraries
            # that scribble scratch relative to the CWD (CatBoost's `catboost_info/`, matplotlib,
            # joblib memmaps) then just work. The script is still run by absolute path and reads
            # /data / writes /out by absolute path, so the CWD does not affect its real I/O.
            "-w", "/tmp",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp",
        ]
        for key, value in request.env.items():
            cmd += ["-e", f"{key}={value}"]
        cmd.append(self.image)
        cmd += self._run_argv(request.entrypoint, with_deps=request.requirements is not None)
        return cmd

    @staticmethod
    def _run_argv(entrypoint: str, *, with_deps: bool) -> list[str]:
        """The in-container command: a direct ``python`` run, or pip-install-then-run with deps."""
        if not with_deps:
            return ["python", f"/work/{entrypoint}"]
        # Install the declared deps into a writable tmpfs (root is read-only) and run with them on
        # the path. Needs network=True (set by the caller). Pinned versions keep it reproducible.
        install = (
            f"pip install --no-cache-dir --target=/tmp/site -r /work/{_REQUIREMENTS_NAME} "
            f"&& PYTHONPATH=/tmp/site python /work/{entrypoint}"
        )
        return ["sh", "-c", install]

    async def _run_container(self, cmd: list[str], name: str, timeout_s: float) -> RunResult:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except OSError as exc:
            # A missing/unreachable docker binary means the gate cannot run the code at all — that
            # is infrastructure, not a verdict. Surface it as a recoverable GateUnavailable (5.1) so
            # the control plane degrades the cycle instead of aborting on a raw OSError.
            raise GateUnavailable(
                f"could not launch the code runner ({self.docker_bin}): {exc}"
            ) from exc
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError:
            await self._kill(name)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return RunResult(exit_code=-1, stdout="", stderr="", timed_out=True)
        return RunResult(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=out.decode(errors="replace"),
            stderr=err.decode(errors="replace"),
        )

    async def _kill(self, name: str) -> None:
        with contextlib.suppress(Exception):
            killer = await asyncio.create_subprocess_exec(
                self.docker_bin, "kill", name,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), timeout=10.0)


@dataclass(slots=True)
class BackendCodeRunner:
    """A :class:`CodeRunner` that runs the submitted code as an isolated **worker** via a
    `WorkerBackend` (ROADMAP 7.4.d / ADR 0003) — the clean resolution of the dropped in-process
    subprocess idea: untrusted code runs in a launched, labelled, isolated worker (a fresh container
    on Docker; a Pod on k8s later), while the gate logic stays trusted in-process.

    Maps the bytes-in/out `RunRequest` onto a `WorkerSpec`: the submission + ``requirements.txt`` +
    datasets are physically-ro inputs, ``/out`` is the writable output dir, the declared output file
    is harvested. With ``requirements`` it pip-installs into an exec tmpfs first (needs
    ``network=True``, set by the gate). Only ``request.inputs`` reach the worker — a gate's private
    answer key (e.g. the FE reserved labels) never does, by construction.

    The control plane binds a `RunContext` (7.4.h, via the verifier that holds this runner), so each
    code-runner worker is **labelled** with its ``tenant``/``run``/``cycle`` for audit + reaping.
    """

    backend: WorkerBackend
    image: str = "python:3.12-slim"
    memory: str = "512m"
    cpus: str = "1"
    pids_limit: int = 128
    tmpfs_size: str = "64m"
    runtime: str | None = None
    config: str = ""
    run_context: RunContext | None = None

    def bind_run_context(self, ctx: RunContext) -> None:
        """Take the run-identity cell the verifier forwards; we stamp it onto every worker."""
        self.run_context = ctx

    def _labels(self) -> Labels:
        ctx = self.run_context
        if ctx is None:
            return Labels(role="code-runner", config=self.config)
        return Labels(
            role="code-runner", config=self.config,
            tenant=ctx.tenant_id, run=ctx.run_id, cycle=str(ctx.cycle),
        )

    async def run(self, request: RunRequest, /) -> RunResult:
        spec = self._worker_spec(request)
        try:
            result = await self.backend.run_to_completion(spec)
        except ProvisioningError as exc:
            # the worker could not be launched — infrastructure, not a verdict: a recoverable gate.
            raise GateUnavailable(f"could not run the submitted code: {exc}") from exc
        output = result.outputs.get(f"/out/{request.output_name}")
        log.info(
            "backend_code_run",
            entrypoint=request.entrypoint, exit_code=result.exit_code,
            timed_out=result.timed_out, produced_output=output is not None,
        )
        return RunResult(
            exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr,
            output=output, timed_out=result.timed_out,
        )

    def _worker_spec(self, request: RunRequest) -> WorkerSpec:
        with_deps = request.requirements is not None
        readonly: dict[str, bytes] = {f"/work/{request.entrypoint}": request.code}
        if request.requirements is not None:
            readonly[f"/work/{_REQUIREMENTS_NAME}"] = request.requirements
        for name, blob in request.inputs.items():
            readonly[f"/data/{name}"] = blob
        if with_deps:
            command: tuple[str, ...] = (
                "sh", "-c",
                f"pip install --no-cache-dir --target=/tmp/site -r /work/{_REQUIREMENTS_NAME} "
                f"&& PYTHONPATH=/tmp/site python /work/{request.entrypoint}",
            )
        else:
            command = ("python", f"/work/{request.entrypoint}")
        return WorkerSpec(
            image=self.image,
            command=command,
            labels=self._labels(),
            readonly_inputs=readonly,
            writable_dirs=("/out",),
            output_globs=(f"/out/{request.output_name}",),
            network=request.network,
            env=dict(request.env),
            limits=ResourceLimits(memory=self.memory, cpus=self.cpus, pids=self.pids_limit),
            scratch=Tmpfs(size=self.tmpfs_size, allow_exec=with_deps),
            # CWD is the writable tmpfs (/tmp), NOT the read-only /work mount, so libraries that
            # write scratch relative to the CWD (CatBoost's `catboost_info/`, matplotlib, joblib)
            # work. The script runs by absolute path and uses /data / /out, so CWD is irrelevant.
            workdir="/tmp",
            runtime=self.runtime,
            timeout_s=request.timeout_s,
        )


def docker_available(docker_bin: str = "docker") -> bool:
    """True if a Docker daemon is reachable — gates the integration test (skips otherwise)."""
    try:
        completed = subprocess.run(
            [docker_bin, "version", "--format", "{{.Server.Version}}"],
            capture_output=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0
