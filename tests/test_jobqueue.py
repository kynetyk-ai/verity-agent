"""The job-queue seam (ROADMAP Phase 7 multi-tenancy). The in-process default; offline."""

from __future__ import annotations

import asyncio

import pytest

from verity.contracts.jobqueue import InProcessJobQueue, Job, JobStatus


def _job(job_id: str, tenant: str = "default") -> Job:
    return Job(job_id=job_id, tenant_id=tenant, run_id=f"run-{job_id}", payload={"goal": "x"})


def test_enqueue_claim_complete_lifecycle() -> None:
    async def go() -> tuple[JobStatus, JobStatus, JobStatus]:
        q = InProcessJobQueue()
        await q.enqueue(_job("j1"))
        queued = await q.status("j1")
        claimed_job = await q.claim()
        claimed = await q.status("j1")
        await q.complete("j1", {"ok": True})
        done = await q.status("j1")
        assert claimed_job is not None and claimed_job.job_id == "j1"
        return queued, claimed, done

    queued, claimed, done = asyncio.run(go())
    assert (queued, claimed, done) == (JobStatus.QUEUED, JobStatus.CLAIMED, JobStatus.DONE)


def test_claim_is_fifo_and_empty_returns_none() -> None:
    async def go() -> list[str | None]:
        q = InProcessJobQueue()
        await q.enqueue(_job("a"))
        await q.enqueue(_job("b"))
        first = await q.claim()
        second = await q.claim()
        empty = await q.claim()
        return [j.job_id if j else None for j in (first, second, empty)]

    assert asyncio.run(go()) == ["a", "b", None]


def test_fail_marks_failed() -> None:
    async def go() -> JobStatus:
        q = InProcessJobQueue()
        await q.enqueue(_job("j1"))
        await q.claim()
        await q.fail("j1", "boom")
        return await q.status("j1")

    assert asyncio.run(go()) is JobStatus.FAILED


def test_unknown_job_raises() -> None:
    async def go() -> None:
        await InProcessJobQueue().status("nope")

    with pytest.raises(KeyError, match="unknown job"):
        asyncio.run(go())
