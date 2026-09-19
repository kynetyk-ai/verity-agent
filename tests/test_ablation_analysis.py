"""The ablation analysis projection (experiments/ablation/analyze) — pure stdlib, no matplotlib.

Builds minimal synthetic `RunReport` cycle dicts in-memory and pins every derived signal: the
held-out score extraction, the self-estimate parse, best-so-far, the regression flag, gate
catch-rate, final-quality aggregation, and the pre-registered Claim A/B/B′ deltas + Cliff's delta.

`experiments/` is no longer tracked in this repo (it is operator-local sweep machinery), so this
module skips wholesale when it is absent rather than failing a fresh clone's suite.
"""

from __future__ import annotations

import pytest

analyze = pytest.importorskip(
    "experiments.ablation.analyze", reason="experiments/ not present in this checkout"
)

CellData = analyze.CellData
cell_from_report = analyze.cell_from_report
cliffs_delta = analyze.cliffs_delta
summary = analyze.summary


def _cycle(
    index: int, *, score: float | None = None, estimate: float | None = None,
    outcome: str | None = "accepted", served: int | None = 100, provisioned: int | None = 0,
    tokens: int | None = 10,
) -> dict:
    """A `RunReport`-shaped cycle dict: score on a gate decision, estimate in the rationale."""
    decisions = [{"gate": "score-and-accept", "score": score}] if score is not None else []
    commit = {"outcome": outcome, "decisions": decisions} if outcome is not None else None
    rationale = (
        f"reasoned. ESTIMATED_BALANCED_ACCURACY: {estimate}" if estimate is not None else "x"
    )
    return {
        "index": index, "commit": commit, "rationale": rationale,
        "served_context_chars": served, "provisioned_object_bytes": provisioned,
        "agent_telemetry": {"total_tokens": tokens} if tokens is not None else None,
    }


def _cell(condition: str, model: str, seed: int, cycles: list[dict]) -> CellData:
    return cell_from_report({"cycles": cycles}, condition=condition, model=model, seed=seed)


# --------------------------------------------------------------------------- per-cycle projection


def test_cycle_score_is_the_max_non_null_decision_score() -> None:
    cell = _cell("exp3", "m", 0, [
        {"index": 0, "commit": {"outcome": "accepted", "decisions": [
            {"gate": "runs-clean", "score": None}, {"gate": "selection", "score": 0.91},
        ]}, "rationale": "x"},
    ])
    assert cell.cycles[0].score == 0.91


def test_crashed_cycle_has_no_score() -> None:
    # runs-clean rejected before scoring → no decision score → None (not 0.0).
    cell = _cell("exp3", "m", 0, [
        {"index": 0, "commit": {"outcome": "rejected", "decisions": [
            {"gate": "runs-clean", "score": None},
        ]}, "rationale": "x"},
    ])
    assert cell.cycles[0].score is None
    assert cell.final_best is None


def test_self_estimate_parse_handles_present_and_absent() -> None:
    cell = _cell("exp2", "m", 0, [
        _cycle(0, score=0.9, estimate=0.95),
        {"index": 1, "commit": None, "rationale": "no estimate here"},
        {"index": 2, "commit": None, "rationale": "ESTIMATED_BALANCED_ACCURACY: garbled"},
    ])
    assert cell.cycles[0].self_estimate == 0.95
    assert cell.cycles[1].self_estimate is None
    assert cell.cycles[2].self_estimate is None  # non-numeric → not matched


# --------------------------------------------------------------------------- per-cell derived


def test_best_so_far_is_cumulative_max_with_none_carry() -> None:
    cell = _cell("exp2", "m", 0, [
        _cycle(0, score=0.80), _cycle(1, score=None, outcome=None),
        _cycle(2, score=0.85), _cycle(3, score=0.70),
    ])
    assert cell.best_so_far() == [0.80, 0.80, 0.85, 0.85]
    assert cell.final_best == 0.85


def test_regression_flag_fires_only_on_accept_below_prior_best() -> None:
    # always-accept: cycle 2 accepts a 0.70 after a 0.85 best → regression.
    worse = _cell("exp2", "m", 0, [
        _cycle(0, score=0.85, outcome="accepted"), _cycle(1, score=0.70, outcome="accepted"),
    ])
    assert worse.regressed() is True
    # improve-gate: the 0.70 is rejected, not accepted → no regression.
    guarded = _cell("exp3", "m", 0, [
        _cycle(0, score=0.85, outcome="accepted"), _cycle(1, score=0.70, outcome="rejected"),
    ])
    assert guarded.regressed() is False


def test_tokens_to_threshold_accumulates_until_reached() -> None:
    cell = _cell("exp2", "m", 0, [
        _cycle(0, score=0.80, tokens=10), _cycle(1, score=0.90, tokens=15),
    ])
    assert cell.tokens_to_threshold(0.85) == 25  # 10 + 15, reached at cycle 1
    assert cell.tokens_to_threshold(0.99) is None


# --------------------------------------------------------------------------- aggregation / summary


def test_gate_catch_rate_counts_fooled_self_claims() -> None:
    # cycle 1 claims 0.99 (> prior best 0.80) but actually scores 0.70 (≤ prior best) → caught.
    fooled = _cell("exp3", "m", 0, [
        _cycle(0, score=0.80, estimate=0.80), _cycle(1, score=0.70, estimate=0.99),
    ])
    rate = summary([fooled])["cells"]["exp3|m"]["gate_catch_rate"]
    assert rate["claimed"] == 2  # cycle 0 (0.80 > none) and cycle 1 (0.99 > 0.80)
    assert rate["caught"] == 1   # only cycle 1's claim was not a real improvement
    assert rate["rate"] == 0.5


def test_cliffs_delta_signs_and_bounds() -> None:
    assert cliffs_delta([0.9, 0.95], [0.1, 0.2]) == 1.0
    assert cliffs_delta([0.1], [0.9]) == -1.0
    assert cliffs_delta([], [0.5]) is None


def test_summary_reports_pre_registered_deltas_per_model() -> None:
    # exp2 (baseline) vs exp3 (gate) for one model, 2 seeds each; exp3 strictly better → Claim A.
    cells = [
        _cell("exp2", "sonnet", 0, [_cycle(0, score=0.80)]),
        _cell("exp2", "sonnet", 1, [_cycle(0, score=0.82)]),
        _cell("exp3", "sonnet", 0, [_cycle(0, score=0.90)]),
        _cell("exp3", "sonnet", 1, [_cycle(0, score=0.92)]),
    ]
    out = summary(cells, threshold=0.85)
    claim_a = out["deltas"]["sonnet"]["A (3-2)"]
    assert claim_a["mean_diff"] > 0 and claim_a["cliffs_delta"] == 1.0
    assert claim_a["n_hi"] == 2 and claim_a["n_lo"] == 2
    # final-quality + tokens-to-threshold land per group
    exp3 = out["cells"]["exp3|sonnet"]
    assert exp3["final_quality_mean"] == 0.91
    assert exp3["tokens_to_threshold"]["n_reached"] == 2


# --------------------------------------------------------------------------- figures (smoke)


def _full_ladder_summary() -> dict:
    """A summary over all five rungs × 1 model × 2 seeds, so every figure has data to render."""
    cells: list[CellData] = []
    base = {"exp1": 0.80, "exp2": 0.82, "exp3": 0.90, "exp4": 0.91, "exp4b": 0.93}
    for cond, score in base.items():
        n_cycles = 1 if cond == "exp1" else 3
        for seed in range(2):
            cycles = [
                _cycle(i, score=score + 0.01 * i + 0.005 * seed, estimate=score + 0.05,
                       provisioned=100 * (i + 1) if cond in ("exp2", "exp3") else 100)
                for i in range(n_cycles)
            ]
            cells.append(_cell(cond, "sonnet", seed, cycles))
    return summary(cells)


def _write_results_dir(path, cells_meta) -> None:  # type: ignore[no-untyped-def]
    """Write a sweep-style results dir: a manifest.json + one report per cell (score on a gate)."""
    import json

    path.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cond, model, seed, score in cells_meta:
        key = f"{cond}-{model}-seed{seed}"
        report = {"cycles": [{"index": 0, "commit": {"outcome": "accepted", "decisions": [
            {"gate": "score-and-accept", "score": score}]}, "rationale": "x"}]}
        (path / f"{key}.json").write_text(json.dumps(report), encoding="utf-8")
        manifest.append({"condition": cond, "model": model, "seed": seed,
                         "run_id": key, "status": "complete", "report": f"{key}.json"})
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_load_results_merges_separately_run_rungs(tmp_path) -> None:
    # The bottom rung (exp1) run on its own + the loop (exp3) run separately combine into one
    # analysis — the §1 split: calibrate first, then run the guarded loop.
    load_results = analyze.load_results

    exp1_dir = tmp_path / "exp1"
    loop_dir = tmp_path / "loop"
    _write_results_dir(exp1_dir, [("exp1", "m", 0, 0.80), ("exp1", "m", 1, 0.82)])
    _write_results_dir(loop_dir, [("exp3", "m", 0, 0.90), ("exp3", "m", 1, 0.92)])

    cells = [c for d in (exp1_dir, loop_dir) for c in load_results(d)]
    out = summary(cells)
    assert set(out["cells"]) == {"exp1|m", "exp3|m"}  # both rungs present from the two dirs
    assert out["cells"]["exp1|m"]["final_quality_mean"] == 0.81
    assert out["cells"]["exp3|m"]["final_quality_mean"] == 0.91


def test_render_all_writes_five_figures(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    figures = pytest.importorskip("experiments.ablation.figures")

    paths = figures.render_all(_full_ladder_summary(), tmp_path)
    assert {p.name for p in paths} == {"F1.png", "F2.png", "F3.png", "F4.png", "F5.png"}
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
