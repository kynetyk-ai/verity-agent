#!/usr/bin/env python3
"""Code-runner timeout calibration (issue #111).

Runs a curated subset of the heaviest *non-CatBoost* agent submission scripts collected across the
prototyping runs through the **exact** component the fe-kaggle runs-clean gate uses
(`verity.verifier.ContainerCodeRunner`, the `verity-code-runner` image), on the full-data kaggle-rung
workload, under the gate's real caps but with a deliberately generous, non-killing timeout — and times
each run's wall-clock. The point is to set a *permissive* `code_timeout_s` from measured data rather
than a guess (3x the slowest clean completion).

Minimum modifications: this is a thin host-side driver over the unchanged code-runner. No daemon, no
verifier service, no agent. Run from the repo root:  uv run python <thisfile>
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from verity.verifier import ContainerCodeRunner, RunRequest

REPO = Path(__file__).resolve().parents[2]
PROTO = REPO / "prototyping_datasci_test"
HERE = PROTO / "code-runner-calibration"
RESULTS = HERE / "results"

# Full-data kaggle-rung workload: the gate trains on the full training set and predicts the real test
# set. Mounted as the names the scripts read (train.csv / test.csv).
TRAIN = PROTO / "verifier" / "full_train.csv"
TEST = PROTO / "verifier" / "test.csv"

# Gate-parity caps (infra/compose.daemon.yml + fe_kaggle_service.py defaults). pids_limit is the
# ContainerCodeRunner default (128) the gate inherits. A generous, non-killing timeout so we capture
# the true runtime rather than clipping it.
RUNNER = ContainerCodeRunner(
    image="verity-code-runner:latest",
    memory="16g",
    cpus="16",
    tmpfs_size="2g",
    pids_limit=128,
)
TIMEOUT_S = 3600.0
HEADROOM = 3.0  # permissive: recommended budget = HEADROOM x slowest clean completion

# The curated subset: the heaviest non-CatBoost scripts we have, spanning model mixes (incl. the
# sklearn RF/ExtraTrees ensemble, likely the slowest) and a 268-line 4-config LightGBM ensemble known
# to have run clean within the 30-min gate. label -> path under prototyping-runs/.
RUNS = PROTO / "prototyping-runs"
CANDIDATES: list[tuple[str, Path]] = [
    ("fugu160-cycle8-662ln-lgbm+xgb", RUNS / "2026-06-27-fugu-fulldata-160min/submissions/cycle8-submission.py"),
    ("fugu160-cycle2-525ln-lgbm", RUNS / "2026-06-27-fugu-fulldata-160min/submissions/cycle2-submission.py"),
    ("fugu160-cycle5-464ln-lgbm", RUNS / "2026-06-27-fugu-fulldata-160min/submissions/cycle5-submission.py"),
    ("sonnet-nudge-cycle8-458ln-rf+et+lgbm+xgb", RUNS / "2026-06-22-sonnet-nudge-fulldata/submissions/cycle8-submission.py"),
    ("sonnet-nudge-cycle5-378ln-et+lgbm+xgb", RUNS / "2026-06-22-sonnet-nudge-fulldata/submissions/cycle5-submission.py"),
    ("gpt5.4-cycle8-365ln-lgbm+xgb", RUNS / "2026-06-21-gpt5.4-fulldata/submissions/cycle8-submission.py"),
    ("fugu-rerun-cycle9-343ln-lgbm+xgb", RUNS / "2026-06-26-fugu-fulldata-rerun/submissions/cycle9-submission.py"),
    ("qwen-cyc8-268ln-4config-lgbm", RUNS / "2026-06-28-qwen-top50-newsteers/submissions/cyc8_rejected_noimprove.py"),
]


async def main() -> None:
    requirements = (HERE / "requirements.txt").read_bytes()
    train, test = TRAIN.read_bytes(), TEST.read_bytes()
    inputs = {"train.csv": train, "test.csv": test}
    env = {"VERITY_DATA": "/data", "VERITY_OUT": "/out"}
    print(f"data: train={len(train)/1e6:.0f}MB test={len(test)/1e6:.0f}MB | "
          f"caps: mem=16g cpus=16 tmpfs=2g | timeout={TIMEOUT_S:.0f}s | n={len(CANDIDATES)}", flush=True)

    records = []
    for i, (label, path) in enumerate(CANDIDATES, 1):
        if not path.is_file():
            print(f"[{i}/{len(CANDIDATES)}] {label}: MISSING {path}", flush=True)
            records.append({"label": label, "status": "missing", "path": str(path)})
            continue
        req = RunRequest(
            code=path.read_bytes(),
            entrypoint="submission.py",
            inputs=inputs,
            output_name="predictions.csv",
            timeout_s=TIMEOUT_S,
            requirements=requirements,
            network=True,
            env=env,
        )
        print(f"[{i}/{len(CANDIDATES)}] {label}: running...", flush=True)
        t0 = time.monotonic()
        res = await RUNNER.run(req)
        wall = time.monotonic() - t0
        produced = res.output is not None
        clean = res.exit_code == 0 and produced and not res.timed_out
        status = ("clean" if clean else "timed_out" if res.timed_out
                  else "oom" if res.exit_code == 137 else "crashed")
        (RESULTS / f"{i:02d}_{label}.stderr.txt").write_text(res.stderr[-8000:])
        (RESULTS / f"{i:02d}_{label}.stdout.txt").write_text(res.stdout[-4000:])
        rec = {
            "label": label, "status": status, "wall_s": round(wall, 1),
            "exit_code": res.exit_code, "timed_out": res.timed_out, "produced_output": produced,
            "stderr_tail": res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "",
        }
        records.append(rec)
        print(f"    -> {status} wall={wall:.0f}s exit={res.exit_code} produced={produced}", flush=True)

    clean_walls = [r["wall_s"] for r in records if r.get("status") == "clean"]
    summary = {
        "caps": {"memory": "16g", "cpus": "16", "tmpfs": "2g", "pids": 128},
        "workload": "full_train.csv + real test.csv (kaggle-rung); wall = pip-install + script run",
        "timeout_cap_s": TIMEOUT_S,
        "n_candidates": len(CANDIDATES),
        "n_clean": len(clean_walls),
        "slowest_clean_s": max(clean_walls) if clean_walls else None,
        "headroom": HEADROOM,
        "recommended_code_timeout_s": round(max(clean_walls) * HEADROOM) if clean_walls else None,
        "records": records,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
