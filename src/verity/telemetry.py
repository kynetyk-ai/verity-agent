"""Metrics emission over the RunReport (ROADMAP Phase 7.3) — an adapter, not woven into the loop.

The control plane *emits* a ``RunReport`` (a faithful projection of the store); telemetry is a
**consumer** of that signal, exactly as the report intends ("parsing it is the receiving service's
job"). :func:`export_run_report` walks a report into a :class:`MetricsSink` — counters for outcomes
and gate decisions, histograms for the per-cycle latencies, gauges for the run totals (cycles,
accepted, tokens, object growth). The control loop is untouched.

The default :class:`NullMetricsSink` is a no-op, so the lean core stays metrics-free.
:class:`PrometheusMetricsSink` (a lazy ``prometheus_client`` import, the ``telemetry`` extra) backs
a ``/metrics`` scrape — a Grafana board over these series gives outcome rate, p50/p95 latencies,
refine-loop depth, and tokens per accepted artifact. An OpenTelemetry sink is the same shape, later.
:func:`metrics_processor` mirrors a few structlog events to a sink so live counters track the loop's
own events without new code in the loop.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from verity.control_plane.run_report import RunReport

__all__ = [
    "MetricsSink",
    "NullMetricsSink",
    "RecordingMetricsSink",
    "PrometheusMetricsSink",
    "export_run_report",
    "metrics_processor",
]


@runtime_checkable
class MetricsSink(Protocol):
    """A minimal metrics surface: monotonic counters, distributions, and point-in-time gauges."""

    def incr(self, name: str, amount: float = 1.0, **labels: str) -> None: ...
    def observe(self, name: str, value: float, **labels: str) -> None: ...
    def gauge(self, name: str, value: float, **labels: str) -> None: ...


class NullMetricsSink:
    """The default — discards everything, so telemetry is opt-in and the core stays metrics-free."""

    def incr(self, name: str, amount: float = 1.0, **labels: str) -> None: ...
    def observe(self, name: str, value: float, **labels: str) -> None: ...
    def gauge(self, name: str, value: float, **labels: str) -> None: ...


class RecordingMetricsSink:
    """Records every call (name, value, labels) — for tests and a quick in-process inspection."""

    def __init__(self) -> None:
        self.counters: list[tuple[str, float, dict[str, str]]] = []
        self.observations: list[tuple[str, float, dict[str, str]]] = []
        self.gauges: list[tuple[str, float, dict[str, str]]] = []

    def incr(self, name: str, amount: float = 1.0, **labels: str) -> None:
        self.counters.append((name, amount, labels))

    def observe(self, name: str, value: float, **labels: str) -> None:
        self.observations.append((name, value, labels))

    def gauge(self, name: str, value: float, **labels: str) -> None:
        self.gauges.append((name, value, labels))


def export_run_report(report: RunReport, sink: MetricsSink) -> None:
    """Walk a :class:`RunReport` into ``sink`` — outcomes, gate decisions, latencies, run totals."""
    labels = {"tenant_id": report.tenant_id, "task_id": report.task_id}
    summary = report.summary

    sink.gauge("verity_cycles_run", float(summary.cycles_run), **labels)
    sink.gauge("verity_accepted", float(len(summary.accepted)), **labels)
    sink.gauge("verity_superseded", float(len(summary.superseded)), **labels)
    sink.gauge("verity_trial_count", float(summary.trial_count), **labels)
    sink.gauge("verity_refine_count", float(summary.refine_count), **labels)
    sink.gauge("verity_object_count", float(summary.object_count), **labels)

    for outcome, count in summary.outcomes.items():
        sink.incr("verity_outcomes_total", float(count), outcome=outcome, **labels)

    telemetry = report.agent_telemetry or {}
    for key in ("input_tokens", "output_tokens", "total_tokens", "model_steps", "tool_calls"):
        value = telemetry.get(key)
        if value is not None:
            sink.gauge(f"verity_{key}", float(value), **labels)

    for cycle in report.cycles:
        timings = cycle.timings
        if timings.cycle_ms is not None:
            sink.observe("verity_cycle_ms", timings.cycle_ms, **labels)
        if timings.sandbox_ms is not None:
            sink.observe("verity_sandbox_ms", timings.sandbox_ms, **labels)
        if timings.commit_ms is not None:
            sink.observe("verity_commit_ms", timings.commit_ms, **labels)
        if cycle.commit is not None:
            for decision in cycle.commit.decisions:
                sink.incr(
                    "verity_gate_decisions_total", 1.0,
                    verdict=decision.verdict, gate=decision.gate, **labels,
                )


# structlog event -> counter, so live counters track the loop's own events without loop changes.
_EVENT_COUNTERS = {
    "proposal_committed": "verity_commits_total",
    "gate_cycle_failed": "verity_gate_failures_total",
    "sandbox_cycle_failed": "verity_sandbox_failures_total",
}


def metrics_processor(sink: MetricsSink) -> Any:
    """A structlog processor that increments a counter for known events, then passes them on."""

    def processor(_logger: Any, _method: str, event_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        name = _EVENT_COUNTERS.get(str(event_dict.get("event", "")))
        if name is not None:
            sink.incr(name)
        return event_dict

    return processor


class PrometheusMetricsSink:
    """A :class:`MetricsSink` over ``prometheus_client`` (lazy import — the ``telemetry`` extra).

    Metrics are created on first use with the label *names* of that call; serve the registry at
    ``/metrics``. ``start_http_server(port, registry=sink.registry)`` exposes the scrape.
    """

    def __init__(self, registry: Any = None) -> None:
        from prometheus_client import CollectorRegistry

        self.registry: Any = registry if registry is not None else CollectorRegistry()
        self._metrics: dict[str, Any] = {}

    def _metric(self, kind: str, name: str, label_names: tuple[str, ...]) -> Any:
        existing = self._metrics.get(name)
        if existing is not None:
            return existing
        from prometheus_client import Counter, Gauge, Histogram

        factory = {"counter": Counter, "histogram": Histogram, "gauge": Gauge}[kind]
        metric = factory(name, name, list(label_names), registry=self.registry)
        self._metrics[name] = metric
        return metric

    def incr(self, name: str, amount: float = 1.0, **labels: str) -> None:
        self._metric("counter", name, tuple(sorted(labels))).labels(**labels).inc(amount)

    def observe(self, name: str, value: float, **labels: str) -> None:
        self._metric("histogram", name, tuple(sorted(labels))).labels(**labels).observe(value)

    def gauge(self, name: str, value: float, **labels: str) -> None:
        self._metric("gauge", name, tuple(sorted(labels))).labels(**labels).set(value)
