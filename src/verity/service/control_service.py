"""The transport-agnostic control-service core (ROADMAP 8.1, ADR 0004).

`ControlService` is the standing service's brain, minus any transport: it multiplexes a **generic
`ControlPlane` + its own `SqliteStore` per task instance**, holds a `TaskCatalog` of the pre-built
task types, persists task definitions through a `TaskIndex`, and wires the `JobQueue` + daemon-level
`RunRecord` seams around runs. The daemon (8.2) and the future HTTP API (8.4) are thin adapters
over *this* object; here it runs in-process, offline, over any `WorkerBackend`.

**Why a store per task** (ADR 0004 (a)): a single shared store would let same-typed task instances
bleed — task B's `INCUMBENTS` slice and context manifest would surface task A's accepted artifacts
of the same type, contaminating B's scoring. A store per task scopes that by construction, at no
kernel cost — the one-CP-per-run shape `fe_run` already uses, with the service as multiplexer.

**Run identity** is unified: each per-task `ControlPlane` is given the *daemon-level*
`RunRecordStore` and a mutable run-id cell the service sets per run, so the kernel records the run
under the service's own ``run_id`` — one id, one record, no kernel change. Execution is **serial**
in v1 (the background executor + run lock is 8.2); a run still runs synchronously here, as today.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from verity.composition.catalog import TaskCatalog, default_catalog
from verity.composition.task_request import TaskRequest
from verity.contracts import Artifact
from verity.contracts.jobqueue import InProcessJobQueue, Job, JobQueue, JobStatus
from verity.control_plane.api import ControlPlane
from verity.control_plane.run_record import (
    InMemoryRunRecordStore,
    RunRecord,
    RunRecordStore,
)
from verity.control_plane.run_report import RunReport
from verity.control_plane.store import SqliteStore
from verity.logging import get_logger
from verity.provisioning.backend import WorkerBackend
from verity.service.task_index import TaskIndex

__all__ = ["ControlService", "UnknownTask"]

log = get_logger("verity.service.control_service")


class UnknownTask(KeyError):
    """Raised when a ``task_id`` was never created (or its definition is unreadable)."""


@dataclass(slots=True)
class _RunIdCell:
    """A mutable run-id holder the service sets per run; the CP reads it as ``run_id_source``."""

    value: str = ""

    def __call__(self) -> str:
        return self.value


@dataclass(slots=True)
class _ResidentTask:
    """A configured, live task: its definition, its control plane, and its run-id cell.

    ``in_cp_task_id`` is the id the builder used *inside* the CP (`"fe"`/`"code"`) — distinct from
    the daemon-level ``task_id`` (the index key); each task instance owns its own control plane.
    """

    request: TaskRequest
    cp: ControlPlane
    in_cp_task_id: str
    run_id_cell: _RunIdCell


def _uuid() -> str:
    return uuid.uuid4().hex


class ControlService:
    """Multiplex a control plane + store per task; create, run, and read back results."""

    def __init__(
        self,
        *,
        backend: WorkerBackend,
        root: Path | None = None,
        catalog: TaskCatalog | None = None,
        run_records: RunRecordStore | None = None,
        queue: JobQueue | None = None,
        task_id_source: Callable[[], str] | None = None,
        run_id_source: Callable[[], str] | None = None,
        job_id_source: Callable[[], str] | None = None,
    ) -> None:
        self._backend = backend
        self._root = root
        self._catalog = catalog if catalog is not None else default_catalog()
        # The daemon-level run-results view. Injected into every per-task CP so the kernel records
        # each run here under the service's own run_id — one store, one record (no kernel change).
        self._run_records = run_records if run_records is not None else InMemoryRunRecordStore()
        self._queue = queue if queue is not None else InProcessJobQueue()
        self._index = TaskIndex(root)
        self._task_id_source = task_id_source if task_id_source is not None else _uuid
        self._run_id_source = run_id_source if run_id_source is not None else _uuid
        self._job_id_source = job_id_source if job_id_source is not None else _uuid
        # Lazily-built, then cached: a configured CP stays resident across runs (no re-configure).
        self._resident: dict[str, _ResidentTask] = {}
        # Per-task stores: created at create_task (so in-memory tasks survive), reopened from disk
        # on rehydration. Keyed by daemon task_id.
        self._stores: dict[str, SqliteStore] = {}
        self._run_jobs: dict[str, str] = {}  # run_id -> job_id (for status by run_id)

    # -- catalog ------------------------------------------------------------------

    def catalog_types(self) -> list[str]:
        """The task types this service can instantiate (rendered by ``verity catalog`` in 8.2)."""
        return self._catalog.types()

    # -- task lifecycle -----------------------------------------------------------

    def create_task(self, request: TaskRequest, *, data: bytes | None = None) -> str:
        """Create a durable task instance from a declarative request; returns its ``task_id``.

        Mints a ``task_id``, creates the task's own store, optionally ingests ``data`` into it as an
        immutable content-addressed root input (stamping ``request.data.data_ref``), and persists
        the request via the index. The control plane is **not** configured yet — that is lazy, on
        ``run`` (so create-without-run and the restart-then-run path share one code path).
        """
        if not self._catalog.has(request.type_name):
            raise UnknownTask(
                f"unknown task type {request.type_name!r}; known: {self._catalog.types()}"
            )
        task_id = self._task_id_source()
        store = self._open_store(task_id)
        if data is not None:
            ref = store.put_object(data)
            request = request.with_data_ref(ref.content_hash)
        self._index.put(task_id, request)
        log.info("task_created", task_id=task_id, type=request.type_name, has_data=data is not None)
        return task_id

    def list_tasks(self) -> list[str]:
        return self._index.list()

    def get_request(self, task_id: str) -> TaskRequest | None:
        return self._index.get(task_id)

    # -- run ----------------------------------------------------------------------

    async def run(self, task_id: str, *, goal: str | None = None) -> str:
        """Run a task once (serial in v1); returns the ``run_id``.

        Mints a job through the `JobQueue`, executes the run body synchronously, and records a
        daemon-level `RunRecord` keyed by ``(tenant_id, run_id)`` (the kernel writes it into the
        injected store under our run id). A run that aborts is recorded ``"aborted"`` and the job is
        marked failed — degrade-don't-crash; the client reads *why* via ``status``/``results``.
        """
        resident = await self._resident_cp(task_id)  # config errors propagate (misuse, fail fast)
        request = resident.request
        run_id = self._run_id_source()
        resident.run_id_cell.value = run_id  # the CP mints THIS run_id at run() start
        job = Job(
            job_id=self._job_id_source(),
            tenant_id=request.tenant_id,
            run_id=run_id,
            payload={"task_id": task_id},
        )
        await self._queue.enqueue(job)
        await self._queue.claim()
        self._run_jobs[run_id] = job.job_id
        try:
            await resident.cp.run(resident.in_cp_task_id, goal=goal or request.goal)
        except Exception as exc:  # degrade-don't-crash: record + mark failed, never escape
            # On a bounded-failure abort the kernel already recorded an "aborted" RunRecord; for any
            # other failure, record one defensively from the store-derived report.
            if self._run_records.get(request.tenant_id, run_id) is None:
                self._run_records.put(
                    RunRecord.from_report(
                        resident.cp.run_report(resident.in_cp_task_id),
                        run_id=run_id,
                        status="aborted",
                    )
                )
            await self._queue.fail(job.job_id, str(exc))
            log.warning("run_failed", task_id=task_id, run_id=run_id, error=str(exc))
            return run_id
        await self._queue.complete(job.job_id, {"run_id": run_id})
        log.info("run_complete", task_id=task_id, run_id=run_id)
        return run_id

    # -- results / status ---------------------------------------------------------

    async def status(self, run_id: str) -> JobStatus:
        """The run's scheduling status (`QUEUED`/`CLAIMED`/`DONE`/`FAILED`)."""
        job_id = self._run_jobs.get(run_id)
        if job_id is None:
            raise UnknownTask(f"unknown run: {run_id}")
        return await self._queue.status(job_id)

    def results(self, run_id: str, *, tenant_id: str = "default") -> RunRecord | None:
        """The run's full operational record (status + `RunReport` + accepted artifact ids)."""
        return self._run_records.get(tenant_id, run_id)

    async def run_report(self, task_id: str) -> RunReport:
        """The latest store-derived `RunReport` for a task (requires the task to be resident)."""
        resident = await self._resident_cp(task_id)
        return resident.cp.run_report(resident.in_cp_task_id)

    async def accepted_artifacts(self, task_id: str, *, type: str | None = None) -> list[Artifact]:
        resident = await self._resident_cp(task_id)
        return resident.cp.accepted_artifacts(type=type)

    def get_object(self, task_id: str, content_hash: str) -> bytes:
        """Egress: read durable artifact/object bytes out of a task's store (code, a built DB)."""
        return self._open_store(task_id).get_object(content_hash)

    # -- internals ----------------------------------------------------------------

    def _open_store(self, task_id: str) -> SqliteStore:
        """Get-or-open this task's store: in-memory stores are held, disk stores reopened on use."""
        if task_id in self._stores:
            return self._stores[task_id]
        if self._root is None:
            store = SqliteStore()
        else:
            task_dir = self._index.task_dir(task_id)
            task_dir.mkdir(parents=True, exist_ok=True)
            store = SqliteStore(path=str(task_dir / "store.db"), object_dir=task_dir / "objects")
        self._stores[task_id] = store
        return store

    async def _resident_cp(self, task_id: str) -> _ResidentTask:
        """Build the task's control plane on first use (or after a restart) and cache it.

        The **rehydration path**: a task created earlier — even in a prior process — is rebuilt here
        from its persisted `TaskRequest` and its on-disk store, via the catalog builder. The kernel
        is given our run-id cell + the daemon run-records store so runs record under the service id.
        """
        resident = self._resident.get(task_id)
        if resident is not None:
            return resident
        request = self._index.get(task_id)
        if request is None:
            raise UnknownTask(f"unknown task: {task_id}")
        store = self._open_store(task_id)
        cell = _RunIdCell()
        cp = ControlPlane(
            store,
            policy=request.policy.to_orchestration_policy(),
            run_id_source=cell,
            run_records=self._run_records,
        )
        in_cp_task_id = await self._catalog.build(
            request.type_name, cp, backend=self._backend, request=request
        )
        resident = _ResidentTask(
            request=request, cp=cp, in_cp_task_id=in_cp_task_id, run_id_cell=cell
        )
        self._resident[task_id] = resident
        log.info("task_resident", task_id=task_id, type=request.type_name)
        return resident
