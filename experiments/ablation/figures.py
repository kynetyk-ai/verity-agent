"""Render the ablation figure catalogue F1–F5 (``docs/experimental-design.md`` §4) with matplotlib.

Consumes the dict :func:`experiments.ablation.analyze.summary` produces — no re-derivation, just
drawing. Imported **only** by the analyze CLI (``analyze.main``), so the analysis projection stays
matplotlib-free and testable without the ``analysis`` dependency group.

Each ``figure_fN`` writes one PNG and returns its path (or ``None`` when the data for that figure is
absent — e.g. F1 needs the ``exp1`` rung). :func:`render_all` runs them all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # headless: render to files, never a display (safe after import in mpl >=3.3)

__all__ = [
    "figure_f1",
    "figure_f2",
    "figure_f3",
    "figure_f4",
    "figure_f5",
    "render_all",
]

JsonDict = dict[str, Any]

# The ladder order + the loop rungs (Exp 1 is the single-shot baseline, excluded from trajectories).
CONDITION_ORDER = ("exp1", "exp2", "exp3", "exp4", "exp4b")
LOOP_CONDITIONS = ("exp2", "exp3", "exp4", "exp4b")
_DPI = 140


def _by_coords(summary: JsonDict) -> dict[tuple[str, str], JsonDict]:
    """Re-key ``summary['cells']`` ("condition|model") to ``(condition, model)`` entries."""
    out: dict[tuple[str, str], JsonDict] = {}
    for key, entry in summary.get("cells", {}).items():
        cond, _, model = key.partition("|")
        out[(cond, model)] = entry
    return out


def _models(cells: dict[tuple[str, str], JsonDict]) -> list[str]:
    return sorted({m for _, m in cells})


def _conditions(cells: dict[tuple[str, str], JsonDict]) -> list[str]:
    present = {c for c, _ in cells}
    return [c for c in CONDITION_ORDER if c in present]


def figure_f1(summary: JsonDict, out_dir: Path, *, reference: float | None = None) -> Path | None:
    """F1 — Exp 1 one-shot final-score distribution by model, with a headroom reference line."""
    cells = _by_coords(summary)
    models = [m for m in _models(cells) if ("exp1", m) in cells]
    data = [cells[("exp1", m)]["final_quality"] for m in models]
    data = [d for d in data if d]
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(1.6 * len(data) + 3, 4))
    ax.boxplot(data, tick_labels=models, showmeans=True)
    for i, d in enumerate(data, start=1):
        ax.scatter([i] * len(d), d, alpha=0.5, color="tab:blue", zorder=3)
    ref = reference if reference is not None else max(v for d in data for v in d)
    ax.axhline(ref, linestyle="--", color="grey")
    ax.annotate(f"reference best {ref:.3f}", xy=(1, ref), xytext=(4, 4),
                textcoords="offset points", fontsize=8, color="grey")
    ax.set_title("F1 — one-shot final score by model (Exp 1)")
    ax.set_ylabel("held-out balanced accuracy")
    return _save(fig, out_dir / "F1.png")


def figure_f2(summary: JsonDict, out_dir: Path) -> Path | None:
    """F2 — best-so-far trajectory (mean + IQR band), faceted by model, one line per loop rung."""
    cells = _by_coords(summary)
    models = _models(cells)
    if not models:
        return None
    fig, axes = plt.subplots(
        1, len(models), sharey=True, squeeze=False, figsize=(4.5 * len(models) + 1, 4)
    )
    for ax, model in zip(axes[0], models, strict=True):
        for cond in LOOP_CONDITIONS:
            entry = cells.get((cond, model))
            traj = entry["trajectory"] if entry else []
            if not traj:
                continue
            xs = [t["cycle"] for t in traj]
            ax.plot(xs, [t["mean"] for t in traj], marker="o", label=cond)
            ax.fill_between(xs, [t["q1"] for t in traj], [t["q3"] for t in traj], alpha=0.15)
        ax.set_title(model)
        ax.set_xlabel("cycle")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("best-so-far held-out BA")
    fig.suptitle("F2 — best-so-far trajectory (mean ± IQR)")
    return _save(fig, out_dir / "F2.png")


def figure_f3(summary: JsonDict, out_dir: Path) -> Path | None:
    """F3 (money plot) — final score by condition × model, annotated with the ladder deltas."""
    cells = _by_coords(summary)
    conditions = _conditions(cells)
    models = _models(cells)
    if not conditions or not models:
        return None
    fig, ax = plt.subplots(figsize=(1.4 * len(conditions) + 3, 4.5))
    width = 0.8 / len(models)
    for j, model in enumerate(models):
        heights = [(cells.get((c, model)) or {}).get("final_quality_mean") or 0.0
                   for c in conditions]
        ax.bar([i + j * width for i in range(len(conditions))], heights, width=width, label=model)
    ax.set_xticks([i + width * (len(models) - 1) / 2 for i in range(len(conditions))])
    ax.set_xticklabels(conditions)
    ax.set_ylabel("final held-out BA (mean)")
    ax.set_title("F3 — final score by condition × model (Claims A / B / B′)")
    ax.legend(fontsize=8)
    delta_lines = [
        f"{model}: " + ", ".join(
            f"{name} Δ={d['mean_diff']:+.3f} (δ={(d['cliffs_delta'] or 0):+.2f})"
            for name, d in md.items()
        )
        for model, md in summary.get("deltas", {}).items()
    ]
    if delta_lines:
        ax.text(0.0, -0.22, "\n".join(delta_lines), transform=ax.transAxes, fontsize=7, va="top")
    return _save(fig, out_dir / "F3.png", extra_bottom=bool(delta_lines))


def figure_f4(summary: JsonDict, out_dir: Path) -> Path | None:
    """F4 — provisioned-context bytes per cycle, one line per condition (bounded vs ballooning)."""
    cells = _by_coords(summary)
    conditions = _conditions(cells)
    models = _models(cells)
    if not conditions:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    drew = False
    for cond in conditions:
        trajs = [cells[(cond, m)]["context_trajectory"] for m in models if (cond, m) in cells]
        xs, ys = _pool_context(trajs)
        if xs:
            ax.plot(xs, ys, marker="o", label=cond)
            drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.set_xlabel("cycle")
    ax.set_ylabel("provisioned-object bytes (mean over models)")
    ax.set_title("F4 — context size per cycle")
    ax.legend(fontsize=8)
    return _save(fig, out_dir / "F4.png")


def figure_f5(summary: JsonDict, out_dir: Path) -> Path | None:
    """F5 — self-vs-independent divergence (fudge) by condition + the gate catch-rate bar."""
    cells = _by_coords(summary)
    conditions = _conditions(cells)
    models = _models(cells)
    if not conditions:
        return None
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4))

    fudge_data, fudge_labels = [], []
    for cond in conditions:
        pooled = [v for m in models for v in (cells.get((cond, m)) or {}).get("fudge", [])]
        if pooled:
            fudge_data.append(pooled)
            fudge_labels.append(cond)
    if fudge_data:
        axl.boxplot(fudge_data, tick_labels=fudge_labels, showmeans=True)
    axl.axhline(0, linestyle="--", color="grey")
    axl.set_title("F5a — self-estimate − independent (fudge)")
    axl.set_ylabel("estimate − independent score")

    rates, rate_labels = [], []
    for cond in conditions:
        claimed = caught = 0
        for m in models:
            gc = (cells.get((cond, m)) or {}).get("gate_catch_rate", {})
            claimed += gc.get("claimed", 0)
            caught += gc.get("caught", 0)
        if claimed:
            rates.append(caught / claimed)
            rate_labels.append(cond)
    if rates:
        axr.bar(rate_labels, rates, color="tab:orange")
    axr.set_ylim(0, 1)
    axr.set_title("F5b — gate catch-rate")
    axr.set_ylabel("fraction of self-claimed improvements caught")
    fig.suptitle("F5 — self-vs-independent divergence + gate catch-rate")
    return _save(fig, out_dir / "F5.png")


def render_all(summary: JsonDict, out_dir: Path, *, reference: float | None = None) -> list[Path]:
    """Render every figure whose data is present; return the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        figure_f1(summary, out_dir, reference=reference),
        figure_f2(summary, out_dir),
        figure_f3(summary, out_dir),
        figure_f4(summary, out_dir),
        figure_f5(summary, out_dir),
    ]
    return [p for p in candidates if p is not None]


def _pool_context(trajs: list[list[JsonDict]]) -> tuple[list[int], list[float]]:
    """Per-cycle mean provisioned-object bytes across models (cycles any model has a value)."""
    xs: list[int] = []
    ys: list[float] = []
    max_len = max((len(t) for t in trajs), default=0)
    for k in range(max_len):
        vals = [
            t[k]["provisioned_object_bytes"] for t in trajs
            if k < len(t) and t[k]["provisioned_object_bytes"] is not None
        ]
        if vals:
            xs.append(k)
            ys.append(sum(vals) / len(vals))
    return xs, ys


def _save(fig: Any, path: Path, *, extra_bottom: bool = False) -> Path:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", pad_inches=0.4 if extra_bottom else 0.1)
    plt.close(fig)
    return path
