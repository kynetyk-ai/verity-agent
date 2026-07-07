"""Sandbox-image purity guards (#136 / #137).

The agent runs *inside* the sandbox image, so the image is part of the isolation boundary: it must
carry no datasets (the ablation answer key lives under ``data/*/verifier/``), no prior run results
(past accepted submissions for the same task), and its venv must satisfy the agent's reasonable
``python -m pip`` assumption. Two layers:

* an offline pin of the ``Dockerfile.sandbox.dockerignore`` entries — the cheap regression guard
  (the 2026-07-07 leak existed precisely because ``data``/``results`` were missing from this file);
* ``docker``-marked checks against the *built* image — the invariant asserted where it actually
  holds, by construction at the image boundary (sibling in spirit to
  ``test_cp_image_carries_no_data_prep``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from verity.verifier import docker_available

REPO = Path(__file__).resolve().parents[1]
SANDBOX_IMAGES = ("verity-sandbox:latest", "verity-fe-sandbox:latest")


def docker_test(fn):
    """Mark ``docker`` and skip when no Docker daemon is reachable (so CI needs no Docker)."""
    fn = pytest.mark.skipif(not docker_available(), reason="no Docker daemon reachable")(fn)
    return pytest.mark.docker(fn)


def _image_present(image: str) -> bool:
    probe = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, check=False
    )
    return probe.returncode == 0


def _run_in(image: str, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "sh", image, "-c", command],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_sandbox_dockerignore_excludes_data_and_results() -> None:
    """The leak-vector entries stay pinned in the per-Dockerfile ignore file."""
    entries = {
        line.strip()
        for line in (REPO / "Dockerfile.sandbox.dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    missing = {"data", "results", ".env", ".git"} - entries
    assert not missing, f"Dockerfile.sandbox.dockerignore lost isolation entries: {missing}"


@docker_test
@pytest.mark.parametrize("image", SANDBOX_IMAGES)
def test_sandbox_image_carries_no_datasets_or_results(image: str) -> None:
    """The built image has no /app/data and no /app/results (#137)."""
    if not _image_present(image):
        pytest.skip(f"{image} not built on this host")
    result = _run_in(image, "ls /app/data 2>/dev/null; ls /app/results 2>/dev/null; true")
    assert result.stdout.strip() == "", (
        f"{image} carries repo data/results inside /app: {result.stdout[:200]}"
    )


@docker_test
@pytest.mark.parametrize("image", SANDBOX_IMAGES)
def test_sandbox_venv_answers_python_dash_m_pip(image: str) -> None:
    """`python -m pip` works in the baked venv (#136) — the agent's reasonable assumption."""
    if not _image_present(image):
        pytest.skip(f"{image} not built on this host")
    result = _run_in(image, "python -m pip --version")
    assert result.returncode == 0, (
        f"python -m pip fails in {image}: {result.stderr.strip()[:200]}"
    )
