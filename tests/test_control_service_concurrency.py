"""Background execution + the run lock + graceful failure (ROADMAP 8.2, ADR 0004 (a)/(c)).

`submit_run` returns a run_id immediately and runs the body as a background task behind a single run
lock, so the daemon serves `status`/`results` reads **concurrently** with an in-flight run. These
test that at the `ControlService` level (no socket / no `service` extra needed): a gated backend
holds a run in-flight while reads return, then releasing it completes the run; and a task the agent
cannot satisfy is recorded as a failed run whose reason is readable — graceful, never a crash.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from tools.harness.dataset import stratified_split, subsample

from tests._fe_offline import FeWorkers, offline_catalog, role_files_from_raw
from verity.composition.task_request import DataRequest, PolicyRequest, TaskRequest
from verity.contracts.jobqueue import JobStatus
from verity.provisioning import FakeBackend
from verity.provisioning.backend import CompletedWorker, WorkerSpec
from verity.service import ControlService

_RAW = b"id,a,class\n0,0,X\n1,2,X\n2,4,X\n3,6,Y\n4,8,Y\n5,10,Y\n"
_REAL_TEST = b"id\n100\n101\n102\n103\n"


def _reserved() -> dict[str, str]:
    return stratified_split(
        subsample(_RAW, per_class=100, target="class"),
        target="class", id_column="id", reserved_fraction=0.5,
    ).reserved_labels


def _fe_request() -> TaskRequest:
    return TaskRequest(type_name="fe-kaggle", policy=PolicyRequest(stop_on_accept=True),
                       data=DataRequest())


def _fe_files() -> dict[str, dict[str, bytes]]:
    return role_files_from_raw(_RAW, _REAL_TEST, per_class=100, reserved_fraction=0.5)


@dataclass
class _GatedBackend:
    """Wraps a `FakeBackend`, blocking every worker on a test-controlled event — so a run stays
    in-flight (the run lock held) until the test releases it. Delegates the rest of the port."""

    inner: FakeBackend
    gate: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def launched(self) -> list[WorkerSpec]:
        return self.inner.launched

    async def run_to_completion(self, spec: WorkerSpec) -> CompletedWorker:
        await self.gate.wait()
        return await self.inner.run_to_completion(spec)

    async def launch(self, spec: WorkerSpec):  # pragma: no cover - FE uses run_to_completion
        return await self.inner.launch(spec)

    async def wait(self, handle, *, timeout_s):  # pragma: no cover
        return await self.inner.wait(handle, timeout_s=timeout_s)

    async def status(self, handle):  # pragma: no cover
        return await self.inner.status(handle)

    async def logs(self, handle):  # pragma: no cover
        return await self.inner.logs(handle)

    async def destroy(self, handle) -> None:  # pragma: no cover
        await self.inner.destroy(handle)

    async def list(self, selector):  # pragma: no cover
        return await self.inner.list(selector)

    async def reap(self, selector) -> int:  # pragma: no cover
        return await self.inner.reap(selector)


def test_submit_run_is_background_and_reads_are_concurrent() -> None:
    async def scenario() -> None:
        workers = FeWorkers(steps=[("submit", ("ds",), b"good", ["feat0"])],
                            by_marker={b"good": _reserved()})
        backend = _GatedBackend(inner=FakeBackend(script=workers))
        service = ControlService(backend=backend, root=None, catalog=offline_catalog())
        task_id = service.create_task(_fe_request(), files=_fe_files())

        run_id = await service.submit_run(task_id)  # returns immediately, run still gated
        # The run is in-flight: status reads CLAIMED and results returns (None) WITHOUT blocking.
        assert await service.status(run_id) is JobStatus.CLAIMED
        assert service.results(run_id) is None

        backend.gate.set()  # release the workers
        await service.await_run(run_id)

        assert await service.status(run_id) is JobStatus.DONE
        record = service.results(run_id)
        assert record is not None and record.status == "complete"

    asyncio.run(scenario())


def test_unsatisfiable_run_fails_gracefully_and_informatively() -> None:
    async def scenario() -> None:
        # The agent proposes nothing each cycle -> repeated sandbox failure -> abort.
        backend = FakeBackend(script=FeWorkers(steps=[], by_marker={}))
        service = ControlService(backend=backend, root=None, catalog=offline_catalog())
        task_id = service.create_task(_fe_request(), files=_fe_files())

        run_id = await service.run(task_id)  # never raises — degrade-don't-crash

        assert await service.status(run_id) is JobStatus.FAILED
        record = service.results(run_id)
        assert record is not None and record.status == "aborted"
        # The reason is readable: a sandbox failure is counted and surfaced in the report.
        assert record.report.summary.outcomes.get("sandbox_failed", 0) >= 1
        assert any(c.sandbox_error for c in record.report.cycles)

    asyncio.run(scenario())
