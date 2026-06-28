"""Project ablation sweep results into metrics + a summary (analysis core, experimental-design §4).

Pure standard library — no matplotlib, no pandas — so the projection is importable and tested
without the plotting extra (the figures live in :mod:`experiments.ablation.figures`, used only by
the CLI). It reads the per-cell `RunReport` JSONs + the ``manifest.json`` that
:func:`experiments.ablation.sweep.run_sweep` writes, and derives the signals the ladder is built to
measure: final held-out balanced accuracy, best-so-far trajectories, served-context growth (F4), the
agent's self-vs-independent divergence (F5), gate catch-rate, regression incidence, and the
pre-registered Claim A/B/B′ deltas with a nonparametric (Cliff's delta) effect size.

The held-out score for a cycle is the **max non-null gate decision score** (the §12 scorer puts it
on ``selection`` / ``score-and-accept``) — the same projection ``verity.eval.harness`` uses.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "CycleData",
    "CellData",
    "load_results",
    "cell_from_report",
    "group_cells",
    "summary",
    "cliffs_delta",
]

JsonDict = dict[str, Any]

# The agent reports its held-out estimate as `ESTIMATED_BALANCED_ACCURACY: <float>` in the proposal
# rationale (segregated from the gate); parse it back for the fudge metric (F5).
_ESTIMATE_RE = re.compile(r"ESTIMATED_BALANCED_ACCURACY:\s*([0-9]*\.?[0-9]+)")

# The condition order of the ladder + the pre-registered comparisons (experimental-design §1.2).
_LADDER = ("exp1", "exp2", "exp3", "exp4", "exp4b")
_DELTAS = (
    ("2-1", "exp2", "exp1"),   # value of naive looping
    ("A (3-2)", "exp3", "exp2"),   # Claim A — the independent gate
    ("B (4-3)", "exp4", "exp3"),   # Claim B — context hygiene
    ("B' (4b-4)", "exp4b", "exp4"),  # Claim B′ — best-vs-all selection
)


# --------------------------------------------------------------------------- per-cycle / per-cell


@dataclass(frozen=True, slots=True)
class CycleData:
    """One cycle's projected facts (kernel-typed; everything optional that a cycle may lack)."""

    index: int
    score: float | None  # independent held-out balanced accuracy (max non-null decision score)
    self_estimate: float | None  # the agent's reported estimate (F5), or None if absent/garbled
    served_context_chars: int | None
    provisioned_object_bytes: int | None
    outcome: str | None  # commit outcome ("accepted"/"rejected"/…) or None for a failed cycle
    tokens: int | None  # this cycle's total agent tokens


@dataclass(frozen=True, slots=True)
class CellData:
    """One ``(condition, model, seed)`` run, projected to its per-cycle series + derived helpers."""

    condition: str
    model: str
    seed: int
    cycles: tuple[CycleData, ...]

    @property
    def scores(self) -> list[float | None]:
        return [c.score for c in self.cycles]

    def best_so_far(self) -> list[float | None]:
        """Cumulative max of the held-out score (None carried until the first scored cycle)."""
        out: list[float | None] = []
        running: float | None = None
        for c in self.cycles:
            if c.score is not None:
                running = c.score if running is None else max(running, c.score)
            out.append(running)
        return out

    @property
    def final_best(self) -> float | None:
        scored = [c.score for c in self.cycles if c.score is not None]
        return max(scored) if scored else None

    def regressed(self) -> bool:
        """True if any **accepted** cycle scored below the best held-out seen in prior cycles.

        Only possible under always-accept (a reject-gate cannot accept a worse-than-best attempt);
        the direct measure of the unguarded loop going backward (experimental-design §4).
        """
        running: float | None = None
        for c in self.cycles:
            if (
                c.outcome == "accepted"
                and c.score is not None
                and running is not None
                and c.score < running
            ):
                return True
            if c.score is not None:
                running = c.score if running is None else max(running, c.score)
        return False

    def tokens_to_threshold(self, threshold: float) -> int | None:
        """Cumulative agent tokens until best-so-far first reaches ``threshold`` (None if never)."""
        cumulative = 0
        for c in self.cycles:
            cumulative += c.tokens or 0
            if c.score is not None and c.score >= threshold:
                return cumulative
        return None


# --------------------------------------------------------------------------- report -> CellData


def _cycle_score(cycle: JsonDict) -> float | None:
    commit = cycle.get("commit")
    if not commit:
        return None
    scores = [d["score"] for d in commit.get("decisions", []) if d.get("score") is not None]
    return max(scores) if scores else None


def _parse_self_estimate(rationale: str | None) -> float | None:
    if not rationale:
        return None
    m = _ESTIMATE_RE.search(rationale)
    return float(m.group(1)) if m else None


def _cycle_tokens(cycle: JsonDict) -> int | None:
    tel = cycle.get("agent_telemetry")
    return tel.get("total_tokens") if tel else None


def cycle_from_dict(cycle: JsonDict) -> CycleData:
    commit = cycle.get("commit")
    return CycleData(
        index=int(cycle.get("index", 0)),
        score=_cycle_score(cycle),
        self_estimate=_parse_self_estimate(cycle.get("rationale")),
        served_context_chars=cycle.get("served_context_chars"),
        provisioned_object_bytes=cycle.get("provisioned_object_bytes"),
        outcome=commit.get("outcome") if commit else None,
        tokens=_cycle_tokens(cycle),
    )


def cell_from_report(report: JsonDict, *, condition: str, model: str, seed: int) -> CellData:
    """Project a `RunReport` dict into a :class:`CellData` for the given cell coordinates."""
    return CellData(
        condition=condition, model=model, seed=seed,
        cycles=tuple(cycle_from_dict(c) for c in report.get("cycles", [])),
    )


def load_results(results_dir: str | Path) -> list[CellData]:
    """Read ``manifest.json`` + each cell's `RunReport` JSON from ``results_dir`` (sweep output).

    The manifest carries the cell coordinates, so the reader never parses the hyphenated filenames.
    A manifest entry whose report file is missing is skipped (logged-by-absence in the count).
    """
    root = Path(results_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cells: list[CellData] = []
    for entry in manifest:
        report_path = root / entry["report"]
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cells.append(
            cell_from_report(
                report, condition=entry["condition"], model=entry["model"],
                seed=int(entry["seed"]),
            )
        )
    return cells


# --------------------------------------------------------------------------- aggregation


def group_cells(cells: Iterable[CellData]) -> dict[tuple[str, str], list[CellData]]:
    """Group cells by ``(condition, model)`` (the replicate set whose seeds we aggregate over)."""
    groups: dict[tuple[str, str], list[CellData]] = {}
    for cell in cells:
        groups.setdefault((cell.condition, cell.model), []).append(cell)
    return groups


def _quartiles(values: Sequence[float]) -> tuple[float, float, float]:
    """(q1, median, q3); degrades gracefully below the 2 points ``statistics.quantiles`` needs."""
    if not values:
        return (0.0, 0.0, 0.0)
    med = statistics.median(values)
    if len(values) < 2:
        v = values[0]
        return (v, v, v)
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return (q1, med, q3)


def _final_quality(group: Sequence[CellData]) -> list[float]:
    return [c.final_best for c in group if c.final_best is not None]


def _trajectory(group: Sequence[CellData]) -> list[JsonDict]:
    """Per-cycle best-so-far across seeds: mean + IQR band (robust at small N)."""
    per_cell = [c.best_so_far() for c in group]
    max_len = max((len(b) for b in per_cell), default=0)
    out: list[JsonDict] = []
    for k in range(max_len):
        vals: list[float] = []
        for b in per_cell:
            if k < len(b):
                v = b[k]
                if v is not None:
                    vals.append(v)
        if not vals:
            continue
        q1, med, q3 = _quartiles(vals)
        out.append({
            "cycle": k, "n": len(vals), "mean": statistics.fmean(vals),
            "median": med, "q1": q1, "q3": q3,
        })
    return out


def _context_trajectory(group: Sequence[CellData]) -> list[JsonDict]:
    """Per-cycle mean provisioned-object bytes + served-context chars (F4)."""
    max_len = max((len(c.cycles) for c in group), default=0)
    out: list[JsonDict] = []
    for k in range(max_len):
        bytes_: list[int] = []
        chars: list[int] = []
        for cell in group:
            if k < len(cell.cycles):
                cd = cell.cycles[k]
                if cd.provisioned_object_bytes is not None:
                    bytes_.append(cd.provisioned_object_bytes)
                if cd.served_context_chars is not None:
                    chars.append(cd.served_context_chars)
        out.append({
            "cycle": k,
            "provisioned_object_bytes": statistics.fmean(bytes_) if bytes_ else None,
            "served_context_chars": statistics.fmean(chars) if chars else None,
        })
    return out


def _fudge(group: Sequence[CellData]) -> list[float]:
    """Pooled ``self_estimate − independent score`` over cycles where both are present (F5)."""
    return [
        c.self_estimate - c.score
        for cell in group for c in cell.cycles
        if c.self_estimate is not None and c.score is not None
    ]


def _self_estimate_coverage(group: Sequence[CellData]) -> float | None:
    """Fraction of scored cycles that carried a parseable self-estimate (a compliance watch)."""
    scored = [c for cell in group for c in cell.cycles if c.score is not None]
    if not scored:
        return None
    return sum(1 for c in scored if c.self_estimate is not None) / len(scored)


def _gate_catch_rate(group: Sequence[CellData]) -> JsonDict:
    """Among self-claimed improvements, the fraction that were not real improvements (F5).

    A self-claimed improvement: a cycle whose ``self_estimate`` exceeds the run's best independent
    score so far. "Not real": the cycle's own independent score is missing or ≤ that prior best —
    the agent thought it improved but didn't. Under the unguarded loop (Exp 2) these commit anyway;
    under the gate (Exp 3) they are rejected — this counts how often the agent fooled itself.
    """
    claimed = caught = 0
    for cell in group:
        prior_best: float | None = None
        for c in cell.cycles:
            if c.self_estimate is not None and (prior_best is None or c.self_estimate > prior_best):
                claimed += 1
                if c.score is None or (prior_best is not None and c.score <= prior_best):
                    caught += 1
            if c.score is not None:
                prior_best = c.score if prior_best is None else max(prior_best, c.score)
    return {"claimed": claimed, "caught": caught, "rate": (caught / claimed) if claimed else None}


def cliffs_delta(treatment: Sequence[float], baseline: Sequence[float]) -> float | None:
    """Cliff's delta effect size in [-1, 1] (nonparametric; no scipy). None if either side is empty.

    Positive → ``treatment`` tends to exceed ``baseline``. Magnitude guide (Romano): |δ|<0.147
    negligible, <0.33 small, <0.474 medium, else large.
    """
    if not treatment or not baseline:
        return None
    gt = sum(1 for a in treatment for b in baseline if a > b)
    lt = sum(1 for a in treatment for b in baseline if a < b)
    return (gt - lt) / (len(treatment) * len(baseline))


def summary(cells: Iterable[CellData], *, threshold: float | None = None) -> JsonDict:
    """The machine-readable analysis: per-group metrics + the pre-registered ladder deltas.

    ``threshold`` (optional) adds a tokens-to-threshold readout per cell group. The deltas compare
    the final-quality distributions of adjacent ladder rungs **within each model** (the §1.3
    equal-compute comparison), each with a mean difference + Cliff's delta + the two sample sizes.
    """
    cell_list = list(cells)
    groups = group_cells(cell_list)
    by_group: JsonDict = {}
    for (condition, model), group in sorted(groups.items()):
        fq = _final_quality(group)
        entry: JsonDict = {
            "condition": condition, "model": model, "n_seeds": len(group),
            "final_quality": fq,
            "final_quality_mean": statistics.fmean(fq) if fq else None,
            "trajectory": _trajectory(group),
            "context_trajectory": _context_trajectory(group),
            "fudge": _fudge(group),
            "self_estimate_coverage": _self_estimate_coverage(group),
            "gate_catch_rate": _gate_catch_rate(group),
            "regression_incidence": (
                sum(1 for c in group if c.regressed()) / len(group) if group else None
            ),
        }
        if threshold is not None:
            reached = [c.tokens_to_threshold(threshold) for c in group]
            entry["tokens_to_threshold"] = {
                "threshold": threshold,
                "reached": [t for t in reached if t is not None],
                "n_reached": sum(1 for t in reached if t is not None),
            }
        by_group[f"{condition}|{model}"] = entry

    models = sorted({m for _, m in groups})
    deltas: JsonDict = {}
    for model in models:
        fq_by_cond = {
            cond: _final_quality(groups.get((cond, model), [])) for cond in _LADDER
        }
        model_deltas: JsonDict = {}
        for name, hi, lo in _DELTAS:
            a, b = fq_by_cond.get(hi, []), fq_by_cond.get(lo, [])
            if not a or not b:
                continue
            model_deltas[name] = {
                "mean_diff": statistics.fmean(a) - statistics.fmean(b),
                "cliffs_delta": cliffs_delta(a, b),
                "n_hi": len(a), "n_lo": len(b),
            }
        if model_deltas:
            deltas[model] = model_deltas

    return {"cells": by_group, "deltas": deltas}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m experiments.ablation.analyze RESULTS_DIR --out FIG_DIR [--threshold T]``.

    Writes ``summary.json`` (pure stdlib) then renders F1–F5 (needs the ``analysis`` group:
    ``uv run --group analysis``). The figures import is lazy, so a summary-only run needs no extra.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Analyze ablation sweep results into figures.")
    parser.add_argument("results_dir", help="sweep output dir (per-cell reports + manifest.json)")
    parser.add_argument("--out", required=True, help="dir for summary.json + F1-F5.png")
    parser.add_argument("--threshold", type=float, default=None,
                        help="held-out score for the tokens-to-threshold readout")
    parser.add_argument("--reference", type=float, default=None,
                        help="F1 headroom reference (default: best observed in the sweep)")
    args = parser.parse_args(argv)

    cells = load_results(args.results_dir)
    result = summary(cells, threshold=args.threshold)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    from experiments.ablation import figures  # lazy: only rendering needs matplotlib

    paths = figures.render_all(result, out, reference=args.reference)
    print(f"analyzed {len(cells)} cells → summary.json + {len(paths)} figures in {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper over the tested projection
    raise SystemExit(main())
