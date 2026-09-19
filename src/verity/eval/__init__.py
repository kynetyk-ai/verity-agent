"""The eval / benchmarking harness (ROADMAP Phase 6.3).

Runs the **same task across multiple model configs** and projects proposal quality + cost + latency
from the existing :class:`~verity.control_plane.run_report.RunReport`. **Cost (dollars) lives here,
never in the control plane** — the CP stays dollar-free; this module is the consumer that prices the
token counts the report already carries (mirroring the report's "all-nullable, consumer computes"
discipline).
"""

from __future__ import annotations

from verity.eval.harness import (
    BenchmarkArm,
    Comparison,
    ComparisonRow,
    ConfiguredTask,
    run_benchmark,
)
from verity.eval.pricing import DEFAULT_PRICING, ModelPrice, cost_usd

__all__ = [
    "BenchmarkArm",
    "Comparison",
    "ComparisonRow",
    "ConfiguredTask",
    "run_benchmark",
    "ModelPrice",
    "DEFAULT_PRICING",
    "cost_usd",
]
