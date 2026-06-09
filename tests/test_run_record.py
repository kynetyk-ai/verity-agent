"""The run-record store + tenant_id threading (ROADMAP Phase 7 multi-tenancy). Offline."""

from __future__ import annotations

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


def test_from_report_pulls_tenant_run_and_accepted_ids() -> None:
    record = RunRecord.from_report(_report(task_id="t9", tenant_id="acme"), status="complete")
    assert record.tenant_id == "acme"
    assert record.run_id == "t9"  # defaults to the report's task_id
    assert record.status == "complete"
    assert record.accepted_artifact_ids == ("sub-1",)  # a pointer into the store, not a copy


def test_from_report_accepts_an_explicit_run_id() -> None:
    record = RunRecord.from_report(_report(), status="complete", run_id="run-42")
    assert record.run_id == "run-42"


def test_store_put_get_and_tenant_namespacing() -> None:
    store = InMemoryRunRecordStore()
    acme = RunRecord.from_report(_report(task_id="r1", tenant_id="acme"), status="complete")
    globex = RunRecord.from_report(_report(task_id="r2", tenant_id="globex"), status="complete")
    store.put(acme)
    store.put(globex)

    assert store.get("acme", "r1") is acme
    assert store.get("globex", "r1") is None  # not in globex's namespace
    assert store.list("acme") == [acme]  # no cross-tenant reads
    assert {r.tenant_id for r in store.list("globex")} == {"globex"}


def test_run_record_to_dict_shape() -> None:
    doc = RunRecord.from_report(_report(tenant_id="acme"), status="complete").to_dict()
    assert doc["tenant_id"] == "acme" and doc["status"] == "complete"
    assert doc["accepted_artifact_ids"] == ["sub-1"]
    assert doc["report"]["tenant_id"] == "acme"  # the embedded report
