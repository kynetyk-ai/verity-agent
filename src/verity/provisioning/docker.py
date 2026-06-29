"""The Docker `WorkerBackend` (ROADMAP 7.4.b / ADR 0003) — the local substrate.

Launches each worker as one ``docker run`` against the host daemon (subprocess, no docker SDK — same
posture as today's two drivers). It is the **single** owner of the hostile-input ``docker run``
argv, unifying what ``container_driver.docker_command`` and ``code_runner._docker_cmd`` build today:
the fixed hardened baseline (``--rm``, ``--read-only``, ``--cap-drop=ALL``, ``no-new-privileges``,
non-root host uid, no swap) applied here, the workload knobs from the `WorkerSpec`.

I/O is the bytes file-transfer protocol (`backend.py`): ``writable_dirs`` are fresh ``:rw`` mounts;
``readonly_inputs`` that nest under one are written *into* that writable mount (no read-only
protection inside the sandbox by design — a corrupted input just makes a bad proposal, and gold
regenerates the environment next cycle), and the rest are plain ``:ro`` mounts; ``output_globs`` are
collected from the writable host dirs after exit. Nothing the worker reads comes from the durable
store and nothing is written to it — the control plane harvests the returned bytes (sole-mutator,
Principle 9).

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
from verity.proc import communicate_capped
from verity.provisioning.backend import (
    CompletedWorker,
    Labels,
    ProvisioningError,
    WorkerHandle,
    WorkerSpec,
    WorkerStatus,
)

__all__ = ["DockerBackend", "docker_available"]

log = get_logger("verity.provisioning.docker")

_Mount = tuple[Path, str, str]  # (host, container_target, "ro"|"rw")
_Argv = list[str]  # aliased so annotations after the `list` method don't resolve to it (mypy)


class DockerBackend:
    """A `WorkerBackend` over the local Docker daemon.

    ``staging_root`` (or the ``VERITY_WORKER_STAGING`` env var) is the directory the per-worker
    staging dirs are created under — the host paths that become worker bind mounts. The default
    (``None``) uses the system temp dir, correct when the backend runs **on the host**. When the
    control plane runs **inside a container** and launches *sibling* workers via the mounted host
    socket, set this to a directory bind-mounted from the host at an **identical path**, so the
    worker ``-v`` sources resolve on the host daemon (otherwise the mount fails — the bind target
    would be a path that only exists inside the CP container).
    """

    def __init__(
        self, docker_bin: str = "docker", *, staging_root: str | Path | None = None
    ) -> None:
        self.docker_bin = docker_bin
        root = staging_root if staging_root is not None else os.environ.get("VERITY_WORKER_STAGING")
        self._staging_root = Path(root) if root else None

    # -- the batch path consumers use ---------------------------------------------

    def _make_staging(self) -> Path:
        """A fresh per-worker staging dir under ``staging_root`` (system temp if unset)."""
        if self._staging_root is not None:
            self._staging_root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="verity-worker-", dir=self._staging_root))

    async def run_to_completion(self, spec: WorkerSpec) -> CompletedWorker:
        name = self._name(spec.labels)
        staging = self._make_staging()
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

        Mounts are the writable ``:rw`` dirs plus the non-nested ``:ro`` inputs (inputs that nest
        under a writable dir are materialized into it by ``_materialize``, not bind-mounted).
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
        for cdir in spec.writable_dirs:
            host = staging / "rw" / cdir.lstrip("/")
            host.mkdir(parents=True, exist_ok=True)
            host.chmod(0o777)  # the non-root container user must be able to write
            mounts.append((host, cdir, "rw"))
            writable_hosts[cdir] = host
        for path, data in spec.readonly_inputs.items():
            enclosing = self._enclosing_writable(path, writable_hosts)
            if enclosing is not None:
                # Nested under a writable dir: write the bytes *into* that writable mount rather
                # than bind-mounting the file on top. A nested file bind-mount fails on Docker
                # Desktop (virtiofs) when the mountpoint doesn't pre-exist, and we don't need it:
                # there is no read-only protection inside the sandbox by design — a corrupted input
                # just makes a bad proposal, and gold regenerates the environment next cycle.
                cdir, host_root = enclosing
                dest = host_root / path[len(cdir.rstrip("/")) + 1:]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            else:
                # Not nested under a writable mount (e.g. /sandbox/input.json, the code-runner's
                # /data and /work): a plain ``:ro`` mount, which never conflicts.
                host = staging / "ro" / path.lstrip("/")
                host.parent.mkdir(parents=True, exist_ok=True)
                host.write_bytes(data)
                mounts.append((host, path, "ro"))
        return mounts, writable_hosts

    @staticmethod
    def _enclosing_writable(
        path: str, writable_hosts: Mapping[str, Path]
    ) -> tuple[str, Path] | None:
        """The writable dir that ``path`` nests inside (container-path prefix), or ``None``."""
        for cdir, host_root in writable_hosts.items():
            if path.startswith(cdir.rstrip("/") + "/"):
                return cdir, host_root
        return None

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

    # -- trusted long-lived service workers (§9.1: the verifier sibling) ----------

    def build_service_argv(self, spec: WorkerSpec, *, name: str) -> _Argv:
        """``docker run -d`` argv for a **trusted** long-lived service worker (the verifier).

        Deliberately a *separate* builder from the hostile :meth:`build_argv`: a service is trusted
        infrastructure that must mount the docker socket (to launch its own code-runner children)
        and join a user-defined network (so the control plane reaches it by name) — the opposite of
        cap-dropped, read-only, socket-less posture untrusted code runs under. Keeping the socket
        mount on this path *only* means an untrusted worker can never acquire it (it goes through
        :meth:`build_argv`, which has no socket branch). Resource limits still apply; the container
        is removed on stop (``--rm``) and explicitly on :meth:`destroy`.
        """
        argv = [self.docker_bin, "run", "-d", "--rm", "--name", name]
        if spec.runtime is not None:
            argv.append(f"--runtime={spec.runtime}")
        argv += [
            f"--memory={spec.limits.memory}", f"--memory-swap={spec.limits.memory}",
            f"--cpus={spec.limits.cpus}", f"--pids-limit={spec.limits.pids}",
        ]
        if spec.network_name is not None:
            argv += ["--network", spec.network_name]
        if spec.mount_docker_socket:
            argv += ["-v", "/var/run/docker.sock:/var/run/docker.sock"]
        for host, target, mode in spec.host_mounts:
            argv += ["-v", f"{host}:{target}:{mode}"]
        for key, value in spec.labels.as_dict().items():
            argv += ["--label", f"{key}={value}"]
        for key, value in spec.env.items():
            argv += ["-e", f"{key}={value}"]
        for var in spec.env_passthrough:  # forwarded by NAME; the value stays in the daemon env
            if os.environ.get(var):
                argv += ["-e", var]
        argv += ["-w", spec.workdir, spec.image]
        argv += list(spec.command)
        return argv

    async def launch(self, spec: WorkerSpec) -> WorkerHandle:
        """Launch a detached **trusted service** worker; returns a handle whose ``id`` is the
        container name (its DNS name on ``network_name``). Reached over the wire, destroyed via
        :meth:`destroy` — not harvested like a batch worker. (The batch path is
        :meth:`run_to_completion`; ``wait`` stays a fleet-loop concern.)"""
        name = spec.service_name or self._name(spec.labels)
        argv = self.build_service_argv(spec, name=name)
        await self._capture(argv)  # detached: returns the container id, then docker run exits
        log.info("service_launched", name=name, image=spec.image, **spec.labels.as_dict())
        return WorkerHandle(id=name, labels=spec.labels)

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
            # A missing/unreachable docker binary is infrastructure, not a verdict — raise the
            # neutral ProvisioningError; the consumer maps it to its own recoverable boundary error.
            raise ProvisioningError(
                f"could not launch worker via {self.docker_bin!r}: {exc}"
            ) from exc
        try:
            # Capped capture (V3): a runaway submission can't OOM the host via an unbounded pipe.
            out, err = await asyncio.wait_for(communicate_capped(proc), timeout=timeout_s)
        except TimeoutError:
            await self._kill(name)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(communicate_capped(proc), timeout=5.0)
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
