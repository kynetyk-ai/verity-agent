"""Containerized FE run entrypoint (ROADMAP 7.4, Part C): ``python -m verity.composition.fe_run``.

Runs the §12 feature-engineering task on a **generic** control plane, configured through its API —
the form the containerized control plane uses. It builds a task-agnostic `ControlPlane`, applies the
FE task with `configure_fe_task` (no hand-wiring), and runs it; the `DockerBackend` launches the
sandbox + code-runner **worker containers** as siblings on the host daemon. When this process is the
control-plane container, the workers are siblings of *it* (see `infra/compose.fe.yml` for the socket
+ shared-staging wiring), so the same code runs the system on the host or fully containerized.

Configuration is by environment (the natural container surface):

* ``VERITY_FE_DATASET``      — path to the labelled CSV in the container (default ``/data/train``)
* ``VERITY_MODEL``           — provider string (default ``anthropic:claude-sonnet-4-6``)
* ``VERITY_LOCAL_BASE_URL``  — if set, ``VERITY_MODEL`` is a local model served here (OpenAI API)
* ``VERITY_MAX_CYCLES``      — refine cycles (default 4)
* ``VERITY_PER_CLASS``       — rows per class to subsample for speed (default 300)
* ``VERITY_RESERVED_FRACTION`` — held-out share (default 0.5)
* ``VERITY_SANDBOX_IMAGE``   — the agent worker image (default ``verity-sandbox:latest``)
* ``VERITY_WORKER_STAGING``  — shared host<->container staging dir (read by `DockerBackend`)
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from verity.composition.dataset import stratified_split, subsample
from verity.composition.fe import FE_GOAL, FE_TASK_ID, ProvisioningConfig, configure_fe_task
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.store import SqliteStore
from verity.domains.feature_engineering import SUBMISSION
from verity.logging import get_logger
from verity.provisioning import DockerBackend
from verity.sandbox.model_spec import ModelSpec
from verity.sandbox.providers import local_spec

log = get_logger("verity.composition.fe_run")


def _model_spec() -> ModelSpec:
    model = os.environ.get("VERITY_MODEL", "anthropic:claude-sonnet-4-6")
    base_url = os.environ.get("VERITY_LOCAL_BASE_URL")
    if base_url:
        return local_spec(model, base_url=base_url)
    return ModelSpec.from_provider_string(model)


async def _amain() -> int:
    dataset = Path(os.environ.get("VERITY_FE_DATASET", "/data/train.csv"))
    if not dataset.exists():
        log.error("dataset_missing", path=str(dataset))
        return 2
    per_class = int(os.environ.get("VERITY_PER_CLASS", "300"))
    max_cycles = int(os.environ.get("VERITY_MAX_CYCLES", "4"))
    reserved_fraction = float(os.environ.get("VERITY_RESERVED_FRACTION", "0.5"))
    spec = _model_spec()
    log.info("fe_run_start", model=spec.model, per_class=per_class, max_cycles=max_cycles)

    raw = subsample(dataset.read_bytes(), per_class=per_class)
    split = stratified_split(raw, target="class", id_column="id",
                             reserved_fraction=reserved_fraction)

    # A GENERIC control plane; the FE task applied through its API. The control plane is not built
    # around FE — `configure_fe_task` registers the FE providers + seeds the dataset + configures.
    cp = ControlPlane(SqliteStore(), policy=OrchestrationPolicy(max_cycles=max_cycles))
    await configure_fe_task(
        cp,
        backend=DockerBackend(),  # reads VERITY_WORKER_STAGING for the sibling-mount path
        split=split,
        provisioning=ProvisioningConfig(
            model_spec=spec,
            sandbox_image=os.environ.get("VERITY_SANDBOX_IMAGE", "verity-sandbox:latest"),
        ),
    )
    await cp.run(FE_TASK_ID, goal=FE_GOAL)

    report = cp.run_report(FE_TASK_ID)
    accepted = cp.accepted_artifacts(type=SUBMISSION)
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\naccepted submissions: {[a.id for a in accepted]}")
    log.info("fe_run_complete", accepted=len(accepted), cycles=report.summary.cycles_run)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
