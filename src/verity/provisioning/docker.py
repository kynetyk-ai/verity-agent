"""The Docker `WorkerBackend` (ROADMAP 7.4.b / ADR 0003) — the local substrate.

Launches each worker as one ``docker run`` against the host daemon (subprocess, no docker SDK — same
posture as today's two drivers). It is the **single** owner of the hostile-input ``docker run``
argv, unifying what ``container_driver.docker_command`` and ``code_runner._docker_cmd`` build today:
the fixed hardened baseline (``--rm``, ``--read-only``, ``--cap-drop=ALL``, ``no-new-privileges``,
non-root host uid, no swap) applied here, the workload knobs from the `WorkerSpec`.

I/O is the bytes file-transfer protocol (`backend.py`): ``readonly_inputs`` are written to a staging
dir and bind-mounted ``:ro`` at their absolute paths (overlaid on a writable mount where they nest);
``writable_dirs`` are fresh ``:rw`` mounts; ``output_globs`` are collected from those writable host
dirs after exit. Nothing the worker reads comes from the durable store and nothing is written to
it — the control plane harvests the returned bytes (sole-mutator, Principle 9).

Posture caveats (this holds a credential to the daemon — ADR f): the daemon should run **rootless**
and behind a narrowing **docker-socket-proxy** in a real deployment; those are deployment config,
not this module's concern. Every launch/destroy is structured-logged with the worker's labels — the
one audited choke point.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from verity.logging import get_logger
from verity.provisioning.backend import (
    CompletedWorker,
    Labels,
    WorkerHandle,
    WorkerSpec,
    WorkerStatus,
)

__all__ = ["DockerBackend", "docker_available"]

log = get_logger("verity.provisioning.docker")

_Mount = tuple[Path, str, str]  # (host, container_target, "ro"|"rw")


class DockerBackend:
    """A `WorkerBackend` over the local Docker daemon."""

    def __init__(self, docker_bin: str = "docker") -> None:
        self.docker_bin = docker_bin

    # -- the batch path consumers use ---------------------------------------------

    async def run_to_completion(self, spec: WorkerSpec) -> CompletedWorker:
        name = self._name(spec.labels)
        staging = Path(tempfile.mkdtemp(prefix="verity-worker-"))
        try:
            mounts, writable_hosts = self._materialize(spec, staging)
            argv = self.build_argv(spec, name=name, mounts=mounts)
            log.info("worker_launched", name=name, image=spec.image, **spec.labels.as_dict())
            result = await self._run(argv, name=name, timeout_s=spec.timeout_s)
            outputs = self._collect(spec, writable_hosts)
            log.info("worker_destroyed", name=name, exit_code=result.exit_code,
                     timed_out=result.timed_out, **spec.labels.as_dict())
            return CompletedWorker(
                exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr,
                outputs=outputs, timed_out=result.timed_out,
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    # -- the argv builder (pure; the security-critical surface, pinned offline) ---

    def build_argv(self, spec: WorkerSpec, *, name: str, mounts: Sequence[_Mount]) -> list[str]:
        """The full ``docker run`` argv: the fixed hardened baseline + the spec's workload knobs.

        Mount order is load-bearing — writable mounts precede the ``:ro`` overlays so a read-only
        input nested under a writable dir (a gold role in ``/work``) shadows it physically (§3.5).
        """
        tmpfs = f"rw,size={spec.scratch.size},mode=1777"
        if spec.scratch.allow_exec:
            tmpfs += ",exec"
        argv = [self.docker_bin, "run", "--rm", "--name", name]
        if spec.runtime is not None:
            argv.append(f"--runtime={spec.runtime}")  # gVisor (runsc) etc. (ADR i)
        argv += [
            "--read-only",
            f"--tmpfs=/tmp:{tmpfs}",
            f"--memory={spec.limits.memory}", f"--memory-swap={spec.limits.memory}",
            f"--cpus={spec.limits.cpus}", f"--pids-limit={spec.limits.pids}",
            "--cap-drop=ALL", "--security-opt=no-new-privileges",
            f"--user={os.getuid()}:{os.getgid()}",
        ]
        if not spec.network:
            argv.append("--network=none")
        for host, target, mode in mounts:  # writable first, then ro overlays (see docstring)
            argv += ["-v", f"{host}:{target}:{mode}"]
        for key, value in spec.labels.as_dict().items():
            argv += ["--label", f"{key}={value}"]
        for key, value in spec.env.items():
            argv += ["-e", f"{key}={value}"]
        for var in spec.env_passthrough:  # forwarded by NAME; the value stays in the daemon env
            if os.environ.get(var):
                argv += ["-e", var]
        for hostname, ip in spec.extra_hosts.items():
            argv.append(f"--add-host={hostname}:{ip}")
        argv += ["-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1", "-w", spec.workdir]
        argv.append(spec.image)
        argv += list(spec.command)
        return argv

    # -- materialize / collect (the file-transfer mechanism — Docker's private detail) --

    def _materialize(
        self, spec: WorkerSpec, staging: Path
    ) -> tuple[list[_Mount], dict[str, Path]]:
        mounts: list[_Mount] = []
        writable_hosts: dict[str, Path] = {}
        for cdir in spec.writable_dirs:  # writable mounts first
            host = staging / "rw" / cdir.lstrip("/")
            host.mkdir(parents=True, exist_ok=True)
            host.chmod(0o777)  # the non-root container user must be able to write
            mounts.append((host, cdir, "rw"))
            writable_hosts[cdir] = host
        for path, data in spec.readonly_inputs.items():  # ro overlays on top
            host = staging / "ro" / path.lstrip("/")
            host.parent.mkdir(parents=True, exist_ok=True)
            host.write_bytes(data)
            mounts.append((host, path, "ro"))
        return mounts, writable_hosts

    def _collect(self, spec: WorkerSpec, writable_hosts: Mapping[str, Path]) -> dict[str, bytes]:
        outputs: dict[str, bytes] = {}
        for pattern in spec.output_globs:  # absolute, e.g. /work/outbox/*
            for cdir, host in writable_hosts.items():
                root = cdir.rstrip("/")
                if pattern == root or pattern.startswith(root + "/"):
                    rel = pattern[len(root) + 1:]
                    for found in sorted(host.glob(rel)):
                        if found.is_file():
                            outputs[f"{root}/{found.relative_to(host)}"] = found.read_bytes()
                    break
        return outputs

    # -- fleet management (label-keyed; powers GC, 7.4.h) --------------------------

    async def list(self, selector: Mapping[str, str]) -> list[WorkerHandle]:
        args = [self.docker_bin, "ps", "--quiet", "--no-trunc"]
        for key, value in selector.items():
            args += ["--filter", f"label={key}={value}"]
        args += ["--filter", "label=harness=verity"]
        out = await self._capture(args)
        return [WorkerHandle(id=cid, labels=Labels(role="")) for cid in out.split()]

    async def reap(self, selector: Mapping[str, str]) -> int:
        handles = await self.list(selector)
        for handle in handles:
            await self.destroy(handle)
        return len(handles)

    async def destroy(self, handle: WorkerHandle) -> None:
        with contextlib.suppress(Exception):
            await self._capture([self.docker_bin, "rm", "-f", handle.id])

    async def status(self, handle: WorkerHandle) -> WorkerStatus:
        out = await self._capture(
            [self.docker_bin, "inspect", "-f", "{{.State.Status}}", handle.id]
        )
        state = out.strip()
        if not state:
            return WorkerStatus.GONE
        return {"running": WorkerStatus.RUNNING, "exited": WorkerStatus.SUCCEEDED}.get(
            state, WorkerStatus.RUNNING
        )

    async def logs(self, handle: WorkerHandle) -> bytes:
        return (await self._capture([self.docker_bin, "logs", handle.id])).encode(errors="replace")

    # launch / wait (detached) underpin a future fleet loop; the batch path is run_to_completion.
    async def launch(self, spec: WorkerSpec) -> WorkerHandle:  # pragma: no cover - fleet path
        raise NotImplementedError("the detached launch path lands with the fleet loop (7.4.h+)")

    async def wait(  # pragma: no cover - fleet path
        self, handle: WorkerHandle, *, timeout_s: float
    ) -> CompletedWorker:
        raise NotImplementedError("the detached wait path lands with the fleet loop (7.4.h+)")

    # -- internals ----------------------------------------------------------------

    def _name(self, labels: Labels) -> str:
        role = labels.role or "worker"
        return f"verity-{role}-{uuid.uuid4().hex[:10]}"

    async def _run(self, argv: Sequence[str], *, name: str, timeout_s: float) -> CompletedWorker:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except OSError as exc:
            # A missing/unreachable docker binary is infrastructure, not a verdict — surface it so
            # the consumer maps it to its own recoverable boundary error (SandboxError/GateUnavail).
            raise OSError(f"could not launch worker via {self.docker_bin!r}: {exc}") from exc
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError:
            await self._kill(name)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return CompletedWorker(exit_code=-1, stdout="", stderr="", timed_out=True)
        return CompletedWorker(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=out.decode(errors="replace"), stderr=err.decode(errors="replace"),
        )

    async def _kill(self, name: str) -> None:
        with contextlib.suppress(Exception):
            killer = await asyncio.create_subprocess_exec(
                self.docker_bin, "kill", name,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), timeout=10.0)

    async def _capture(self, args: Sequence[str]) -> str:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        out, _ = await proc.communicate()
        return out.decode(errors="replace")


def docker_available(docker_bin: str = "docker") -> bool:
    """True if a Docker daemon is reachable — gates the integration tests."""
    try:
        done = subprocess.run(
            [docker_bin, "version", "--format", "{{.Server.Version}}"],
            capture_output=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0
