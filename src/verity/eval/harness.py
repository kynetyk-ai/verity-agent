"""The cross-model benchmark harness (ROADMAP Phase 6.3).

Run the **same task** through several model configs (arms) and project a side-by-side comparison of
proposal quality + cost + latency. It reads only the existing :class:`RunReport` surface and the
:mod:`verity.eval.pricing` cost layer — it adds **no** new control-plane concept. Each arm gives a
``build`` that configures a :class:`~verity.control_plane.api.ControlPlane` for one model (exactly
what the test helpers already do); the harness runs it, pulls the report, and rows it up.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from verity.control_plane.api import ControlPlane
from verity.control_plane.run_report import RunReport
from verity.control_plane.store import default_clock
from verity.eval.pricing import DEFAULT_PRICING, ModelPrice, PricingTable, cost_usd

__all__ = [
    "ConfiguredTask",
    "BenchmarkArm",
    "ComparisonRow",
    "Comparison",
    "run_benchmark",
    "COMPARISON_SCHEMA_VERSION",
]

# Bump when the comparison JSON shape changes.
COMPARISON_SCHEMA_VERSION = 1

JsonDict = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ConfiguredTask:
    """A control plane with one task configured, ready to run (what an arm's ``build`` returns).

    ``model_name`` is the arm's known model identity, used as the pricing key when the run telemetry
    reports a null/arbitrary model name (common for local / OpenAI-compatible servers).
    """

    control_plane: ControlPlane
    task_id: str
    goal: str
    model_name: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkArm:
    """One model under test. ``build`` is async because ``ControlPlane.configure`` is async.

    ``price`` overrides the pricing table for this arm (e.g. a local model at ``ModelPrice(0, 0)``,
    or a hosted rate keyed off the arm rather than the reported model name).
    """

    label: str
    build: Callable[[], Awaitable[ConfiguredTask]]
    price: ModelPrice | None = None


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """One arm's results, projected from its :class:`RunReport` (+ the cost layer)."""

    label: str
    model: str | None
    cycles_run: int
    outcomes: dict[str, int]
    accepted_count: int
    superseded_count: int
    wall_time_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    model_steps: int | None
    tool_calls: int | None
    cost_usd: float | None
    best_score: float | None

    def to_dict(self) -> JsonDict:
        return {
            "label": self.label, "model": self.model, "cycles_run": self.cycles_run,
            "outcomes": self.outcomes, "accepted_count": self.accepted_count,
            "superseded_count": self.superseded_count, "wall_time_ms": self.wall_time_ms,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens, "model_steps": self.model_steps,
            "tool_calls": self.tool_calls, "cost_usd": self.cost_usd,
            "best_score": self.best_score,
        }


@dataclass(frozen=True, slots=True)
class Comparison:
    """The benchmark result — one row per arm, plus metadata."""

    rows: list[ComparisonRow]
    generated_at: str
    schema_version: int = COMPARISON_SCHEMA_VERSION

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version, "generated_at": self.generated_at,
            "rows": [r.to_dict() for r in self.rows],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def render_text(self) -> str:
        """A compact fixed-width table (a convenience, not the consumed surface)."""
        header = f"{'arm':<16} {'accepted':>8} {'cycles':>6} {'tokens':>8} {'cost$':>9} {'best':>6}"
        lines = [header, "-" * len(header)]
        for r in self.rows:
            cost = "—" if r.cost_usd is None else f"{r.cost_usd:.4f}"
            tokens = "—" if r.total_tokens is None else str(r.total_tokens)
            best = "—" if r.best_score is None else f"{r.best_score:.3f}"
            lines.append(
                f"{r.label:<16} {r.accepted_count:>8} {r.cycles_run:>6} "
                f"{tokens:>8} {cost:>9} {best:>6}"
            )
        return "\n".join(lines)


async def run_benchmark(
    arms: Sequence[BenchmarkArm],
    *,
    pricing: PricingTable = DEFAULT_PRICING,
    clock: Callable[[], str] = default_clock,
) -> Comparison:
    """Run each arm's task to completion and project a :class:`Comparison` (one row per arm)."""
    rows: list[ComparisonRow] = []
    for arm in arms:
        task = await arm.build()
        await task.control_plane.run(task.task_id, goal=task.goal)
        report = task.control_plane.run_report(task.task_id)
        rows.append(_row_from_report(arm, task, report, pricing))
    return Comparison(rows=rows, generated_at=clock())


def _row_from_report(
    arm: BenchmarkArm, task: ConfiguredTask, report: RunReport, pricing: PricingTable
) -> ComparisonRow:
    tel = report.agent_telemetry or {}
    summary = report.summary
    return ComparisonRow(
        label=arm.label,
        model=tel.get("model") or task.model_name,
        cycles_run=summary.cycles_run,
        outcomes=dict(summary.outcomes),
        accepted_count=len(summary.accepted),
        superseded_count=len(summary.superseded),
        wall_time_ms=summary.wall_time_ms,
        input_tokens=tel.get("input_tokens"),
        output_tokens=tel.get("output_tokens"),
        total_tokens=tel.get("total_tokens"),
        model_steps=tel.get("model_steps"),
        tool_calls=tel.get("tool_calls"),
        cost_usd=cost_usd(
            report.agent_telemetry, pricing, model_key=task.model_name, price=arm.price
        ),
        best_score=_best_score(report),
    )


def _best_score(report: RunReport) -> float | None:
    """The max non-null gate score across all cycles' decisions (the harness extracts it, per the

    RunReport's contract that score is one optional decision field — never a first-class concept).
    """
    scores = [
        d.score
        for c in report.cycles
        if c.commit is not None
        for d in c.commit.decisions
        if d.score is not None
    ]
    return max(scores) if scores else None
