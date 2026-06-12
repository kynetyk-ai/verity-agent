"""Telemetry over the RunReport (ROADMAP Phase 7.3). Offline; Prometheus behind importorskip."""

from __future__ import annotations

import pytest

from verity.control_plane.run_report import (
    CommitReport,
    CycleReport,
    DecisionReport,
    RunReport,
    RunSummary,
    Timings,
)
from verity.telemetry import (
    NullMetricsSink,
    RecordingMetricsSink,
    export_run_report,
    metrics_processor,
)


def _report() -> RunReport:
    commit = CommitReport(
        outcome="accepted", status="accepted", supersedes=None, defects=[],
        decisions=[
            DecisionReport("runs-clean", "accept", "ok", [], None),
            DecisionReport("selection", "accept", "beats", [], 0.9),
        ],
    )
    cycle = CycleReport(
        index=0, goal="g", entered_protocol=True, shape_error=None, sandbox_error=None,
        gate_error=None, proposal=None, commit=commit, rationale=None,
        timings=Timings(cycle_ms=120.0, sandbox_ms=80.0, commit_ms=30.0),
        agent_telemetry=None,
    )
    summary = RunSummary(
        cycles_run=1, outcomes={"accepted": 1, "rejected": 2},
        accepted=[{"id": "sub-1", "type": "Note"}], superseded=[],
        trial_count=2, refine_count=0, object_count=3, wall_time_ms=120.0,
    )
    return RunReport(
        report_schema_version=1, task_id="t1", generated_at="t", policy={},
        cycles=[cycle], summary=summary,
        agent_telemetry={"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050,
                         "model_steps": 4, "tool_calls": 3, "model": "claude"},
        tenant_id="acme",
    )


def test_export_emits_gauges_counters_and_histograms() -> None:
    sink = RecordingMetricsSink()
    export_run_report(_report(), sink)

    gauges = {name: value for name, value, _ in sink.gauges}
    assert gauges["verity_cycles_run"] == 1.0
    assert gauges["verity_accepted"] == 1.0
    assert gauges["verity_object_count"] == 3.0
    assert gauges["verity_total_tokens"] == 1050.0  # from agent_telemetry

    counters = {(name, tuple(sorted(lbls.items()))) for name, _v, lbls in sink.counters}
    # outcomes -> verity_outcomes_total{outcome=...}; both outcomes present
    outcome_labels = {
        lbls["outcome"] for name, _v, lbls in sink.counters if name == "verity_outcomes_total"
    }
    assert outcome_labels == {"accepted", "rejected"}
    # one gate-decision counter per decision (2 in the cycle's commit)
    gate = [c for c in sink.counters if c[0] == "verity_gate_decisions_total"]
    assert len(gate) == 2 and {c[2]["verdict"] for c in gate} == {"accept"}
    assert counters  # non-empty

    observed = {name for name, _v, _l in sink.observations}
    assert observed == {"verity_cycle_ms", "verity_sandbox_ms", "verity_commit_ms"}


def test_every_metric_is_tenant_and_task_labelled() -> None:
    sink = RecordingMetricsSink()
    export_run_report(_report(), sink)
    for _name, _value, labels in sink.gauges + sink.counters + sink.observations:
        assert labels.get("tenant_id") == "acme" and labels.get("task_id") == "t1"


def test_null_sink_is_a_noop() -> None:
    export_run_report(_report(), NullMetricsSink())  # must not raise


def test_metrics_processor_counts_known_events() -> None:
    sink = RecordingMetricsSink()
    processor = metrics_processor(sink)
    out = processor(None, "info", {"event": "proposal_committed", "task_id": "t1"})
    processor(None, "warning", {"event": "gate_cycle_failed"})
    processor(None, "info", {"event": "something_else"})  # ignored

    names = [name for name, _v, _l in sink.counters]
    assert "verity_commits_total" in names and "verity_gate_failures_total" in names
    assert len(sink.counters) == 2  # the unknown event emitted nothing
    assert out["event"] == "proposal_committed"  # the event dict passes through


def test_prometheus_sink_records_into_a_registry() -> None:
    pytest.importorskip("prometheus_client")
    from prometheus_client import generate_latest

    from verity.telemetry import PrometheusMetricsSink

    sink = PrometheusMetricsSink()
    export_run_report(_report(), sink)
    scrape = generate_latest(sink.registry).decode()

    assert "verity_cycles_run" in scrape
    assert "verity_outcomes_total" in scrape
    assert 'tenant_id="acme"' in scrape
