"""The Phase 8 done-line, live (ROADMAP 8.3, ADR 0004 sprint 3): one standing container, many tasks.

This proves the headline property of the long-lived control-plane service: a **single running
container, configured at runtime**, runs more than one task type on more than one input, exports
durable artifacts to the exchange, and survives a restart with its task definitions intact — all
with **no image rebuild between tasks**. It drives the daemon exactly as an operator (or Claude
Code) would: `docker exec verity-cp verity …` over the in-container Unix socket.

``@docker`` + ``@live`` — manual / billed. Auto-skips without Docker, the control-plane + sandbox
images, a model (an ``ANTHROPIC_API_KEY`` or a ``VERITY_LOCAL_BASE_URL``), and the stellar dataset.
Build the images first: ``just cp-serve`` does it, or::

    docker build -f Dockerfile.sandbox      -t verity-sandbox:latest .
    docker build -f Dockerfile.controlplane -t verity-controlplane:latest .

The flow asserted end to end:
  1. ``verity catalog`` lists **both** ``fe`` and ``code`` task types (multi-type, one image);
  2. an **fe** task: ingest the dataset → create → run → results accepted → export to the exchange,
     with the exported bytes visible on the host;
  3. a **code** task: create → run → results accepted — *no rebuild* between (2) and (3);
  4. **restart** the container, then re-run the *pre-restart* fe task — proving durable definitions.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

_CP_IMAGE = "verity-controlplane:latest"
_SANDBOX_IMAGE = "verity-sandbox:latest"
_COMPOSE = "infra/compose.daemon.yml"
_CONTAINER = "verity-cp"
_DATASET = Path("feature-engineering-test/train.csv")


def _docker_ok() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _image_present(name: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", name], capture_output=True, timeout=10
    ).returncode == 0


def _model_args() -> list[str]:
    """Per-task model selection, derived from the environment (hosted key or a local endpoint)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ["--model", os.environ.get("VERITY_MODEL", "anthropic:claude-sonnet-4-6")]
    base = os.environ["VERITY_LOCAL_BASE_URL"]  # guaranteed by _SKIP otherwise
    model = os.environ.get("VERITY_MODEL", "qwen3.6:27b-coding-mxfp8")
    return ["--model", model, "--base-url", base]


_SKIP = not (
    _docker_ok()
    and _image_present(_CP_IMAGE)
    and _image_present(_SANDBOX_IMAGE)
    and _DATASET.exists()
    and (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("VERITY_LOCAL_BASE_URL"))
)


def _cli(*args: str, timeout: float = 1200.0) -> dict:
    """Run a `verity` client subcommand inside the standing container; parse its JSON stdout."""
    done = subprocess.run(
        ["docker", "exec", _CONTAINER, "verity", *args],
        capture_output=True, text=True, timeout=timeout,
    )
    label = " ".join(args)
    assert done.returncode == 0, f"`verity {label}` failed: {done.stderr or done.stdout}"
    return json.loads(done.stdout)


def _await_run(run_id: str, *, timeout_s: float = 1800.0) -> dict:
    """Poll `verity status` until the run reaches a terminal state; return its results."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = _cli("status", run_id)["status"]
        if status in ("done", "failed"):
            break
        time.sleep(5.0)
    else:  # pragma: no cover - timeout guard
        raise AssertionError(f"run {run_id} did not finish within {timeout_s}s")
    return _cli("results", run_id)


@pytest.fixture
def daemon(tmp_path: Path) -> Iterator[Path]:
    """Bring the daemon up over a fresh exchange + store on the host; tear it down after."""
    exchange = tmp_path / "exchange"
    store = tmp_path / "store"
    for sub in ("in", "out"):
        (exchange / sub).mkdir(parents=True)
    store.mkdir()
    env = {
        **os.environ,
        "VERITY_EXCHANGE_HOST": str(exchange),
        "VERITY_STORE_HOST": str(store),
        "VERITY_WORKER_STAGING": "/tmp/verity-staging",
    }
    Path("/tmp/verity-staging").mkdir(exist_ok=True)
    up = ["docker", "compose", "-f", _COMPOSE, "up", "-d"]
    subprocess.run(up, env=env, check=True, timeout=120)
    try:
        # Wait for the daemon to bind its socket inside the container.
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            probe = subprocess.run(
                ["docker", "exec", _CONTAINER, "test", "-S", "/run/verity.sock"],
                capture_output=True,
            )
            if probe.returncode == 0:
                break
            time.sleep(1.0)
        else:  # pragma: no cover - startup guard
            raise AssertionError("daemon socket never appeared")
        yield exchange
    finally:
        subprocess.run(["docker", "compose", "-f", _COMPOSE, "down"], env=env, timeout=120)


@pytest.mark.docker
@pytest.mark.live
@pytest.mark.skipif(_SKIP, reason="needs Docker, the CP+sandbox images, a model, and the dataset")
def test_one_container_two_task_types_durable_across_restart(daemon: Path) -> None:
    exchange = daemon

    # (1) the catalog describes more than one task type, from a single image.
    types = {t["type_name"] for t in _cli("catalog")["task_types"]}
    assert {"fe", "code"} <= types, f"expected fe + code in the catalog, got {types}"

    # (2) an fe task: ingest the dataset through the exchange, create, run, accept, export.
    (exchange / "in" / "train.csv").write_bytes(_DATASET.read_bytes())
    handle = _cli("ingest", "train.csv")["handle"]
    fe_task = _cli(
        "create", "--type", "fe", "--data", handle, "--max-cycles", "4", "--stop-on-accept",
        "--per-class", "300", "--reserved-fraction", "0.5", *_model_args(),
    )["task_id"]
    fe_results = _await_run(_cli("run", fe_task)["run_id"])
    assert fe_results["accepted_artifact_ids"], "the fe run produced no accepted artifact"
    export = _cli("export", fe_results["run_id"])
    assert export["exported"], "nothing was exported"
    out_dir = exchange / "out" / fe_results["run_id"]
    assert out_dir.is_dir() and any(out_dir.rglob("*")), "no exported bytes on the host exchange"

    # (3) a code task on the SAME running container — no rebuild between task types.
    code_task = _cli("create", "--type", "code", *_model_args())["task_id"]
    code_results = _await_run(_cli("run", code_task)["run_id"])
    assert code_results["accepted_artifact_ids"], "the code run produced no accepted artifact"

    # (4) restart the container (volumes retained), then re-run the pre-restart fe task.
    subprocess.run(["docker", "restart", _CONTAINER], check=True, timeout=60)
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if subprocess.run(
            ["docker", "exec", _CONTAINER, "test", "-S", "/run/verity.sock"], capture_output=True
        ).returncode == 0:
            break
        time.sleep(1.0)
    assert fe_task in _cli("tasks")["tasks"], "the task definition did not survive the restart"
    rerun = _await_run(_cli("run", fe_task)["run_id"])
    assert rerun["accepted_artifact_ids"], "the pre-restart fe task did not run after the restart"
