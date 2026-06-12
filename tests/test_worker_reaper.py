"""The label-reaper (ROADMAP 7.4.h / ADR 0003 §e). Offline — `FakeBackend`, no Docker.

Workers are stamped with run identity (7.4.h), which makes GC a label query: the reaper selects by
``{harness,tenant,run,role}`` and destroys the match, knowing nothing about the run that made them.
This asserts the selector floor (always scoped to this harness) and that reaping by run removes only
that run's workers — the death-switch for crashed control planes and aborted runs.
"""

from __future__ import annotations

import asyncio

from verity.provisioning import FakeBackend, build_selector, reap
from verity.provisioning.backend import Labels, WorkerSpec


def _spec(*, run: str, role: str = "sandbox", tenant: str = "default") -> WorkerSpec:
    return WorkerSpec(
        image="x", command=("true",),
        labels=Labels(role=role, tenant=tenant, run=run, harness="verity"),
    )


def test_build_selector_always_carries_the_harness_floor() -> None:
    assert build_selector() == {"harness": "verity"}
    full = build_selector(tenant="acme", run="r1", role="sandbox")
    assert full == {"harness": "verity", "tenant": "acme", "run": "r1", "role": "sandbox"}


def test_reap_by_run_removes_only_that_runs_workers() -> None:
    backend = FakeBackend()

    async def go() -> tuple[int, list[str]]:
        await backend.launch(_spec(run="r1"))
        await backend.launch(_spec(run="r1", role="code-runner"))
        await backend.launch(_spec(run="r2"))
        reaped = await reap(backend, build_selector(run="r1"))
        survivors = await backend.list(build_selector())
        return reaped, [h.labels.run for h in survivors]

    reaped, survivors = asyncio.run(go())
    assert reaped == 2  # both r1 workers, regardless of role
    assert survivors == ["r2"]  # r2 untouched


def test_reap_matches_nothing_returns_zero() -> None:
    backend = FakeBackend()

    async def go() -> int:
        await backend.launch(_spec(run="r1"))
        return await reap(backend, build_selector(run="absent"))

    assert asyncio.run(go()) == 0


def test_reap_floor_scopes_to_this_harness() -> None:
    backend = FakeBackend()

    async def go() -> int:
        await backend.launch(_spec(run="r1"))
        # a foreign worker (another harness) is never matched by the verity floor
        await backend.launch(
            WorkerSpec(image="x", command=("true",), labels=Labels(role="sandbox", harness="other"))
        )
        return await reap(backend, build_selector())

    assert asyncio.run(go()) == 1  # only the harness=verity worker
