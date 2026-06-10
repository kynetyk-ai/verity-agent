"""The `WorkerBackend` provisioning seam (ROADMAP 7.4 / ADR 0003).

A backend launches a **labelled, isolated, ephemeral worker** — a batch unit of work: bytes in,
run, bytes out, gone. The control plane never touches this; sandbox/verifier *implementations* (the
driver, the code-runner) drive a backend behind their ports. ``DockerBackend`` exists now (7.4.b); a
``K8sBackend`` is a second impl of the *same* port for a cluster (designed-for, not built).

**Worker I/O is a bytes file-transfer protocol, not host paths** — so the contract forecloses no
substrate (a shared host filesystem is a Docker detail, absent on k8s). Three buckets, because plain
"files in / files out" would break the sandbox's *physical* gold-data isolation (§3.5):

* ``readonly_inputs`` — ``{absolute container path: bytes}``, materialized **physically read-only**
  (gold roles, the cycle input, the submitted code + datasets); the agent/script reads but never
  mutates them (a ro input under a writable dir is overlaid ro on top).
* ``writable_dirs`` — absolute container paths of fresh writable-ephemeral areas the backend gives
  (the sandbox scratch+outbox, the code-runner output dir); discarded with the worker.
* ``output_globs`` — absolute glob patterns (under a ``writable_dir``) collected after exit, keyed
  by absolute container path; absence matches nothing (the consumer decides what empty means).

Two principles fix the *mechanism*, not just the surface (ADR 0003 §f, and `contracts/wire.py`):
**backend-mediated** — inputs handed in, outputs collected out, the worker never touches the durable
store (sole-mutator, Principle 9; isolation — a hostile worker holds no store credential); and
**inline bytes now, by-reference later** — the inline-vs-content-addressed choice `wire.py` already
made for object bytes, so a future ``bytes | ObjectRef`` union (size-gated, reusing the object
store's content-addressing) is an isolated optimization, not a new transport.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

__all__ = [
    "WorkerBackend",
    "WorkerSpec",
    "WorkerHandle",
    "WorkerStatus",
    "CompletedWorker",
    "Labels",
    "ResourceLimits",
    "Tmpfs",
    "ProvisioningError",
]


class ProvisioningError(RuntimeError):
    """A worker could not be launched/managed — *infrastructure*, not a result (a missing daemon, an
    unreachable backend). **Neutral**: the backend raises this; each consumer remaps it to its own
    recoverable boundary error (the sandbox → ``SandboxError``, a gate → ``GateUnavailable``), so
    degrade-don't-crash holds without the backend knowing those types."""


@dataclass(frozen=True, slots=True)
class Labels:
    """The composite identity stamped on every worker — the universal key for naming, selection, and
    GC. ``run`` is the worker's *identity* scope; ``cycle`` its *lifetime* scope (a fresh worker per
    cycle), so consecutive cycles of one run never alias (ADR 0003 §b)."""

    role: str  # "sandbox" | "verifier" | "code-runner"
    tenant: str = "default"
    job: str = ""
    run: str = ""
    cycle: str = ""
    config: str = ""
    harness: str = "verity"

    def as_dict(self) -> dict[str, str]:
        """The non-empty labels, for stamping onto a worker."""
        pairs = {
            "harness": self.harness, "tenant": self.tenant, "job": self.job,
            "run": self.run, "cycle": self.cycle, "role": self.role, "config": self.config,
        }
        return {k: v for k, v in pairs.items() if v}

    def matches(self, selector: Mapping[str, str]) -> bool:
        """True if every key in ``selector`` equals this worker's label — a partial-match query, the
        substrate-neutral basis for ``list``/``reap`` (a backend renders it to its filter form)."""
        mine = self.as_dict()
        return all(mine.get(key) == value for key, value in selector.items())


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Per-worker resource caps (the `--memory`/`--cpus`/`--pids-limit` posture)."""

    memory: str = "512m"
    cpus: str = "1"
    pids: int = 128


@dataclass(frozen=True, slots=True)
class Tmpfs:
    """The worker's writable ``/tmp`` scratch. ``allow_exec`` is needed only by the deps path
    (native wheels mmap their ``.so``), kept off the strict default."""

    size: str = "64m"
    allow_exec: bool = False


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """Everything that *varies* per worker. The backend applies the fixed hardened baseline itself
    (``--rm``, ``--read-only``, ``--cap-drop=ALL``, ``no-new-privileges``, non-root host uid, no
    swap); the spec carries only the workload knobs. Every field traces to a real argv flag the two
    existing builders produce (`container_driver.docker_command`, `code_runner._docker_cmd`)."""

    image: str
    command: tuple[str, ...]
    labels: Labels
    readonly_inputs: Mapping[str, bytes] = field(default_factory=dict)  # abs path -> bytes, ro
    writable_dirs: tuple[str, ...] = ()
    output_globs: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    env_passthrough: tuple[str, ...] = ()  # forwarded by NAME (value stays in the daemon env)
    network: bool = False
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    scratch: Tmpfs = field(default_factory=Tmpfs)
    extra_hosts: Mapping[str, str] = field(default_factory=dict)  # host -> ip (e.g. host-gateway)
    workdir: str = "/work"
    runtime: str | None = None  # OCI runtime, e.g. "runsc" (gVisor); None = daemon default (ADR i)
    timeout_s: float = 600.0


@dataclass(frozen=True, slots=True)
class WorkerHandle:
    """A reference to a launched worker. Carries the labels, **not a network address**: batch
    workers are never reached over the wire (input mounted, output harvested), which is why ADR
    0003's address-by-handle concern dissolves under the batch model."""

    id: str
    labels: Labels


class WorkerStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    GONE = "gone"  # destroyed, or no such worker


@dataclass(frozen=True, slots=True)
class CompletedWorker:
    """The outcome of one batch worker run. ``outputs`` are the collected ``output_globs`` files,
    keyed by their workspace-relative path; the backend makes no judgment about them."""

    exit_code: int
    stdout: str
    stderr: str
    outputs: Mapping[str, bytes] = field(default_factory=dict)
    timed_out: bool = False


@runtime_checkable
class WorkerBackend(Protocol):
    """Launch / observe / destroy labelled isolated workers. Substrate-agnostic; knows nothing of
    gates, provenance, domains, or policy (those stay in the control plane / consumers)."""

    async def run_to_completion(self, spec: WorkerSpec) -> CompletedWorker:
        """The batch path consumers use: launch -> wait -> collect outputs -> destroy."""
        ...

    async def launch(self, spec: WorkerSpec) -> WorkerHandle: ...
    async def wait(self, handle: WorkerHandle, *, timeout_s: float) -> CompletedWorker: ...
    async def status(self, handle: WorkerHandle) -> WorkerStatus: ...
    async def logs(self, handle: WorkerHandle) -> bytes: ...
    async def destroy(self, handle: WorkerHandle) -> None: ...  # idempotent
    async def list(self, selector: Mapping[str, str]) -> list[WorkerHandle]: ...  # powers GC
    async def reap(self, selector: Mapping[str, str]) -> int: ...  # delete all matching; count
