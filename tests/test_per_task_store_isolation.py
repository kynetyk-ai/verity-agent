"""Per-task store isolation (ROADMAP 8.1, ADR 0004 (a)): no same-typed `INCUMBENTS` bleed.

The control service multiplexes a `ControlPlane` + `SqliteStore` **per task instance**, precisely so
two instances of the *same* task type cannot contaminate each other: if they shared one store, task
B's `INCUMBENTS` slice (and its context manifest) would surface task A's accepted artifacts of the
same type, corrupting B's selection scoring. This test creates two FE tasks on one service, drives
task A to an accepted `Submission`, and asserts task B sees **none** of it — and that the two tasks
own distinct stores by construction.
"""

from __future__ import annotations

import asyncio

from tests._fe_offline import FeWorkers
from verity.composition.dataset import stratified_split, subsample
from verity.composition.task_request import DataRequest, PolicyRequest, TaskRequest
from verity.domains.feature_engineering import SUBMISSION
from verity.provisioning import FakeBackend
from verity.service import ControlService

_RAW = b"id,a,class\n0,0,X\n1,2,X\n2,4,X\n3,6,Y\n4,8,Y\n5,10,Y\n"


def _fe_request() -> TaskRequest:
    return TaskRequest(
        type_name="fe",
        policy=PolicyRequest(stop_on_accept=True),
        data=DataRequest(per_class=100, reserved_fraction=0.5),
    )


def test_two_fe_instances_do_not_share_incumbents() -> None:
    reserved = stratified_split(
        subsample(_RAW, per_class=100, target="class"),
        target="class", id_column="id", reserved_fraction=0.5,
    ).reserved_labels

    # One backend, one FeWorkers — but only task A is ever run, so the script drives A alone.
    workers = FeWorkers(
        steps=[("submit", ("ds",), b"good", ["feat0"])],
        by_marker={b"good": dict(reserved)},
    )
    service = ControlService(backend=FakeBackend(script=workers), root=None)

    task_a = service.create_task(_fe_request(), data=_RAW)
    task_b = service.create_task(_fe_request(), data=_RAW)
    assert task_a != task_b

    asyncio.run(service.run(task_a))

    accepted_a = asyncio.run(service.accepted_artifacts(task_a, type=SUBMISSION))
    accepted_b = asyncio.run(service.accepted_artifacts(task_b, type=SUBMISSION))
    assert accepted_a, "task A should have an accepted Submission"
    assert not accepted_b, "task B must not see task A's accepted Submission (INCUMBENTS bleed)"

    # Structural proof: the two task instances own distinct control planes and distinct stores.
    res_a = asyncio.run(service._resident_cp(task_a))
    res_b = asyncio.run(service._resident_cp(task_b))
    assert res_a.cp is not res_b.cp
    assert res_a.cp.store is not res_b.cp.store
