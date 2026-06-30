#!/usr/bin/env python3
"""Document the fe-kaggle climb across runs — a **user-side** readout over `verity results` JSON.

The control plane deliberately keeps no "best score" / "score trajectory" — a score is one optional
field a gate decision may carry, and reading it as a climb is domain knowledge (the receiving
service's job; see ``verity/control_plane/run_report.py``). So this lives here, not in the kernel:
it extracts the per-cycle proxy / competitive-estimate / public scores from one or more RunReport
JSON documents and prints the climb of accepted submissions toward the top-N% bar.

Usage:
    # one run
    docker exec verity-cp verity results <run_id> | python3 prototyping_datasci_test/trajectory.py
    # many runs (the cross-run climb): collect the JSONs, pass them as files
    python3 prototyping_datasci_test/trajectory.py run1.json run2.json ...

Each input is a RunReport dict (``verity results <run_id>``). Reads stdin when given no files.
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _decision_score(commit: dict[str, Any] | None, gate: str) -> float | None:
    if not commit:
        return None
    for d in commit.get("decisions", ()):
        if d.get("gate") == gate and d.get("score") is not None:
            return float(d["score"])
    return None


def _rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    run = str(report.get("generated_at", "?"))
    out: list[dict[str, Any]] = []
    for cycle in report.get("cycles", ()):
        commit = cycle.get("commit")
        proposal = cycle.get("proposal") or {}
        out.append(
            {
                "run": run,
                "cycle": cycle.get("index"),
                "artifact": proposal.get("artifact_id", "-"),
                "status": (commit or {}).get("status", "—"),
                "proxy": _decision_score(commit, "proxy-improves"),
                "estimate": _decision_score(commit, "competitive"),
                "public": _decision_score(commit, "kaggle"),
            }
        )
    return out


def _fmt(score: float | None) -> str:
    return f"{score:.5f}" if score is not None else "-"


def main() -> int:
    paths = sys.argv[1:]
    blobs = (
        [open(p, encoding="utf-8").read() for p in paths]
        if paths
        else [sys.stdin.read()]
    )
    reports: list[dict[str, Any]] = []
    for blob in blobs:
        blob = blob.strip()
        if blob:
            reports.append(json.loads(blob))
    if not reports:
        print("no RunReport JSON on input", file=sys.stderr)
        return 2

    rows = [r for report in reports for r in _rows(report)]
    rows.sort(key=lambda r: (r["run"], r["cycle"] if r["cycle"] is not None else -1))

    task = reports[0].get("task_id", "fe-kaggle")
    print(f"{task} score trajectory  ({len(reports)} run(s), {len(rows)} cycle(s))")
    print(f"{'run':<26} {'cyc':>3}  {'status':<10} {'proxy':>8} {'estimate':>8} {'public':>8}")
    for r in rows:
        run = r["run"][:25]
        print(
            f"{run:<26} {str(r['cycle']):>3}  {r['status']:<10} "
            f"{_fmt(r['proxy']):>8} {_fmt(r['estimate']):>8} {_fmt(r['public']):>8}"
        )

    publics = [(r["public"], r) for r in rows if r["public"] is not None]
    accepted = [r for r in rows if r["status"] == "accepted"]
    if publics:
        best, br = max(publics, key=lambda pr: pr[0])
        print(f"\nbest public score: {best:.5f}  (run {br['run'][:25]}, cycle {br['cycle']})")
    print(f"accepted submissions (leaderboard steps): {len(accepted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
