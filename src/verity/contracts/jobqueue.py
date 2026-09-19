"""The job-queue seam (ROADMAP Phase 7 — multi-tenancy & run-control, seams-first).

The control plane grows into a multi-tenant **job server**: a run is enqueued, claimed by a worker,
and completed. This is the *seam* — a :class:`JobQueue` port plus a trivial in-process default that
behaves like today (a run executes synchronously; the queue just records its state). The real engine
— a networked queue + worker model, commits still serialized per tenant — is a later adapter swap
behind this port, not a reshape (spec §16, "designed-for, not built now").

It lives in :mod:`verity.contracts` because, like the other ports, it is a cross-service contract
owned by none of them. A job has its own ``job_id`` and *carries* the run it schedules
(``tenant_id`` + ``run_id``), so tenant scoping is keyed in from the outset. See
``docs/glossary.md`` for task vs. run vs. job.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

__all__ = ["JobStatus", "Job", "JobQueue", "InProcessJobQueue"]


class JobStatus(StrEnum):
    """The lifecycle of a queued run."""

    QUEUED = "queued"
    CLAIMED = "claimed"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Job:
    """A unit of run-work: identified by ``job_id``, scheduling the run ``(tenant_id, run_id)``.

    ``payload`` is opaque to the queue (the run inputs a worker needs). A job *is not* the run — it
    is the run enqueued for a worker; ``JobStatus`` is its scheduling lifecycle, distinct from the
    artifact lifecycle (see ``docs/glossary.md``).
    """

    job_id: str
    tenant_id: str
    run_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class JobQueue(Protocol):
    """Enqueue / claim / complete / status — the run-control surface (engine-agnostic)."""

    async def enqueue(self, job: Job) -> str: ...
    async def claim(self) -> Job | None: ...
    async def complete(self, job_id: str, result: Mapping[str, Any]) -> None: ...
    async def fail(self, job_id: str, error: str) -> None: ...
    async def status(self, job_id: str) -> JobStatus: ...


@dataclass(frozen=True, slots=True)
class _JobState:
    job: Job
    status: JobStatus
    result: Mapping[str, Any] | None = None
    error: str | None = None


class InProcessJobQueue:
    """The trivial in-process default: a FIFO of jobs + their states, no worker, no network.

    Behaves like today — a caller enqueues, claims, runs the work itself, completes — so swapping it
    in changes nothing. A Redis/SQS-backed queue is a drop-in :class:`JobQueue` later.
    """

    def __init__(self) -> None:
        self._pending: deque[str] = deque()
        self._states: dict[str, _JobState] = {}

    async def enqueue(self, job: Job) -> str:
        self._states[job.job_id] = _JobState(job=job, status=JobStatus.QUEUED)
        self._pending.append(job.job_id)
        return job.job_id

    async def claim(self) -> Job | None:
        while self._pending:
            job_id = self._pending.popleft()
            state = self._states.get(job_id)
            if state is not None and state.status is JobStatus.QUEUED:
                self._states[job_id] = _JobState(job=state.job, status=JobStatus.CLAIMED)
                return state.job
        return None

    async def complete(self, job_id: str, result: Mapping[str, Any]) -> None:
        state = self._require(job_id)
        self._states[job_id] = _JobState(job=state.job, status=JobStatus.DONE, result=dict(result))

    async def fail(self, job_id: str, error: str) -> None:
        state = self._require(job_id)
        self._states[job_id] = _JobState(job=state.job, status=JobStatus.FAILED, error=error)

    async def status(self, job_id: str) -> JobStatus:
        return self._require(job_id).status

    def _require(self, job_id: str) -> _JobState:
        state = self._states.get(job_id)
        if state is None:
            raise KeyError(f"unknown job: {job_id}")
        return state
