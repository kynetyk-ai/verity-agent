"""The run-record store + tenant_id threading (ROADMAP Phase 7 multi-tenancy). Offline."""

from __future__ import annotations

import pytest

from verity.control_plane.run_record import InMemoryRunRecordStore, RunRecord
from verity.control_plane.run_report import RunReport, RunSummary


def _report(*, task_id: str = "t1", tenant_id: str = "default") -> RunReport:
    summary = RunSummary(
        cycles_run=1, outcomes={"accepted": 1},
        accepted=[{"id": "sub-1", "type": "Note"}], superseded=[],
        trial_count=0, refine_count=0, object_count=0, wall_time_ms=12.0,
    )
    return RunReport(
        report_schema_version=1, task_id=task_id, generated_at="t", policy={},
        cycles=[], summary=summary, agent_telemetry=None, tenant_id=tenant_id,
    )


def test_report_carries_tenant_id_in_dict() -> None:
    assert _report(tenant_id="acme").to_dict()["tenant_id"] == "acme"


def test_from_report_records_task_run_and_accepted_ids() -> None:
    record = RunRecord.from_report(
        _report(task_id="t9", tenant_id="acme"), run_id="run-1", status="complete"
    )
    assert record.tenant_id == "acme"
    assert record.task_id == "t9"  # which task (configuration) this run executed
    assert record.run_id == "run-1"  # the run's own identity — distinct from the task
    assert record.status == "complete"
    assert record.accepted_artifact_ids == ("sub-1",)  # a pointer into the store, not a copy


def test_run_id_is_required_so_task_and_run_stay_distinct() -> None:
    # No default: a run's identity must be minted by the caller, never silently reuse the task_id.
    with pytest.raises(TypeError):
        RunRecord.from_report(_report(), status="complete")  # type: ignore[call-arg]


def test_store_put_get_and_tenant_namespacing() -> None:
    store = InMemoryRunRecordStore()
    acme = RunRecord.from_report(_report(task_id="t1", tenant_id="acme"), run_id="r1", status="ok")
    globex = RunRecord.from_report(
        _report(task_id="t2", tenant_id="globex"), run_id="r1", status="ok"
    )
    store.put(acme)
    store.put(globex)

    assert store.get("acme", "r1") is acme
    assert store.get("globex", "r1") is globex  # same run_id, different tenant namespace
    assert store.list("acme") == [acme]  # no cross-tenant reads
    assert {r.tenant_id for r in store.list("globex")} == {"globex"}


def test_list_filters_runs_by_task() -> None:
    store = InMemoryRunRecordStore()
    r1 = RunRecord.from_report(_report(task_id="taskA"), run_id="run-1", status="ok")
    r2 = RunRecord.from_report(_report(task_id="taskA"), run_id="run-2", status="ok")
    r3 = RunRecord.from_report(_report(task_id="taskB"), run_id="run-3", status="ok")
    for r in (r1, r2, r3):
        store.put(r)

    # task : run is 1 : many — both runs of taskA, none of taskB
    assert {r.run_id for r in store.list("default", task_id="taskA")} == {"run-1", "run-2"}
    assert store.list("default", task_id="taskB") == [r3]


def test_run_record_to_dict_shape() -> None:
    doc = RunRecord.from_report(
        _report(task_id="t1", tenant_id="acme"), run_id="run-1", status="complete"
    ).to_dict()
    assert doc["tenant_id"] == "acme" and doc["task_id"] == "t1" and doc["run_id"] == "run-1"
    assert doc["status"] == "complete" and doc["accepted_artifact_ids"] == ["sub-1"]
    assert doc["report"]["task_id"] == "t1"  # the embedded report names its task
