#!/usr/bin/env python3
"""Candidate-4 follow-up: measure the TRUE runtime of the one script the main run clipped.

`sonnet-nudge-cycle8` (a 5-fold stack of RF + ExtraTrees + LightGBM + XGBoost) hit the main run's
3600s cap, which is the *same* arbitrary 60-min budget under test — so "it timed out" tells us nothing
about its real cost. This re-runs ONLY that script through the identical gate-parity code-runner, but
with a deliberately huge, non-clipping cap, to get its true completion time. Run AFTER the main loop
finishes (so it gets the full 16 cores and the number is representative).

  uv run python prototyping_datasci_test/code-runner-calibration/followup_candidate4.py
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

SCRIPT = PROTO / "prototyping-runs/2026-06-22-sonnet-nudge-fulldata/submissions/cycle8-submission.py"
TRAIN = PROTO / "verifier" / "full_train.csv"
TEST = PROTO / "verifier" / "test.csv"

RUNNER = ContainerCodeRunner(image="verity-code-runner:latest", memory="16g", cpus="16",
                             tmpfs_size="2g", pids_limit=128)
TIMEOUT_S = 21600.0  # 6h non-clipping cap


async def main() -> None:
    req = RunRequest(
        code=SCRIPT.read_bytes(),
        entrypoint="submission.py",
        inputs={"train.csv": TRAIN.read_bytes(), "test.csv": TEST.read_bytes()},
        output_name="predictions.csv",
        timeout_s=TIMEOUT_S,
        requirements=(HERE / "requirements.txt").read_bytes(),
        network=True,
        env={"VERITY_DATA": "/data", "VERITY_OUT": "/out"},
    )
    print(f"candidate-4 follow-up: 5-fold RF+ET+LGBM+XGB stack | cap={TIMEOUT_S:.0f}s", flush=True)
    t0 = time.monotonic()
    res = await RUNNER.run(req)
    wall = time.monotonic() - t0
    clean = res.exit_code == 0 and res.output is not None and not res.timed_out
    status = "clean" if clean else "timed_out" if res.timed_out else "crashed"
    (RESULTS / "candidate4_followup.stderr.txt").write_text(res.stderr[-8000:])
    (RESULTS / "candidate4_followup.stdout.txt").write_text(res.stdout[-4000:])
    rec = {"label": "sonnet-nudge-cycle8-458ln-rf+et+lgbm+xgb (true runtime)",
           "status": status, "wall_s": round(wall, 1), "exit_code": res.exit_code,
           "timed_out": res.timed_out, "produced_output": res.output is not None}
    (RESULTS / "candidate4_followup.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
