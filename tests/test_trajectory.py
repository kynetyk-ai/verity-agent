"""The user-side fe-kaggle trajectory readout (``prototyping_datasci_test/trajectory.py``).

It's a standalone script (not an installed module), so we load it by path and drive both its parsing
helpers and its CLI entrypoint over a synthetic RunReport — the shape ``verity results`` emits.
Proves the climb readout extracts the proxy / competitive-estimate / public scores, best included.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "prototyping_datasci_test" / "trajectory.py"


def _load():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("fe_trajectory", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cycle(i: int, status: str, *, proxy=None, estimate=None, public=None):  # type: ignore[no-untyped-def]
    decisions = []
    if proxy is not None:
        decisions.append({"gate": "proxy-improves", "verdict": "accept", "score": proxy})
    if estimate is not None:
        kind = "refine" if status == "revised" else "accept"
        decisions.append({"gate": "competitive", "verdict": kind, "score": estimate})
    if public is not None:
        decisions.append({"gate": "kaggle", "verdict": "accept", "score": public})
    return {
        "index": i,
        "proposal": {"artifact_id": f"submission-{i}"},
        "commit": {"status": status, "decisions": decisions},
    }


def _report():  # type: ignore[no-untyped-def]
    return {
        "task_id": "fe-kaggle",
        "generated_at": "2026-06-21T10:00:00Z",
        "cycles": [
            _cycle(1, "revised", proxy=0.9120, estimate=0.9120),  # below the bar, no submission
            _cycle(2, "accepted", proxy=0.9540, estimate=0.9540, public=0.95314),  # the climb
        ],
    }


def test_decision_score_picks_the_named_gate() -> None:
    mod = _load()
    commit = _report()["cycles"][1]["commit"]
    assert mod._decision_score(commit, "kaggle") == 0.95314
    assert mod._decision_score(commit, "competitive") == 0.9540
    assert mod._decision_score(commit, "absent") is None
    assert mod._decision_score(None, "kaggle") is None


def test_rows_project_the_climb() -> None:
    mod = _load()
    rows = mod._rows(_report())
    assert [r["status"] for r in rows] == ["revised", "accepted"]
    assert rows[0]["public"] is None  # nothing submitted below the bar
    assert rows[1]["public"] == 0.95314


def test_cli_reports_the_best_public_score() -> None:
    out = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        input=json.dumps(_report()),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "best public score: 0.95314" in out
    assert "accepted submissions (leaderboard steps): 1" in out
