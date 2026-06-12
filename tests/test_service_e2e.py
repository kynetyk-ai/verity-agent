"""End-to-end service split (ROADMAP Phase 7.2): a control plane driving a *containerized* verifier
over real HTTP. ``@docker`` — auto-skipped without Docker, the ``verity-verifier`` image, or the
``service`` extra (build it: ``docker build -f Dockerfile.verifier -t verity-verifier:latest .``).

This is the milestone proof: the CP serializes a ``VerifierRequest``, POSTs it to a verifier running
in another container, decodes the ``VerdictBundle``, and commits — with **no control-plane change**
(it sees only the async ``VerifierPort``). The second test kills the verifier mid-run and asserts
the cycle degrades to a recorded gate failure rather than crashing the run (over the wire).
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import time
from collections.abc import Iterator

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx  # noqa: E402

from tests.test_gate_resilience import _build, _note  # noqa: E402
from verity.contracts import ArtifactStatus  # noqa: E402
from verity.control_plane.api import OrchestrationPolicy  # noqa: E402
from verity.transport.http import build_remote_verifier  # noqa: E402
from verity.verifier import docker_available  # noqa: E402

_IMAGE = "verity-verifier:latest"
_PORT = 8013


def _verifier_image_present() -> bool:
    try:
        done = subprocess.run(
            ["docker", "image", "inspect", _IMAGE], capture_output=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _wait_healthy(port: int, *, timeout_s: float = 25.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with contextlib.suppress(httpx.HTTPError):
            resp = httpx.post(f"http://localhost:{port}/health", timeout=2.0)
            if resp.status_code == 200 and resp.json().get("ok"):
                return
        time.sleep(0.5)
    raise AssertionError(f"verifier container did not become healthy on :{port}")


@contextlib.contextmanager
def _verifier_container(port: int) -> Iterator[str]:
    name = f"verity-verifier-e2e-{port}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name, "-p", f"{port}:8001", _IMAGE],
        capture_output=True, check=True,
    )
    try:
        _wait_healthy(port)
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


_SKIP = not (docker_available() and _verifier_image_present())


@pytest.mark.docker
@pytest.mark.skipif(_SKIP, reason="needs Docker and the verity-verifier image")
def test_control_plane_commits_via_a_containerized_verifier() -> None:
    with _verifier_container(_PORT):
        cp, _sandbox, store = _build(
            items=[_note("n1")],
            policy=OrchestrationPolicy(max_cycles=1),
            verifier=build_remote_verifier(f"http://localhost:{_PORT}"),
        )
        results = asyncio.run(cp.run("t1", goal="author a note"))

    assert results[0].commit is not None
    assert results[0].commit.status is ArtifactStatus.ACCEPTED  # committed via the remote verifier
    assert store.get_artifact("n1") is not None


@pytest.mark.docker
@pytest.mark.skipif(_SKIP, reason="needs Docker and the verity-verifier image")
def test_a_killed_verifier_degrades_the_cycle_not_the_run() -> None:
    with _verifier_container(_PORT) as name:
        cp, _sandbox, _store = _build(
            items=[_note("n1"), _note("n2")],
            policy=OrchestrationPolicy(max_cycles=2),
            verifier=build_remote_verifier(f"http://localhost:{_PORT}"),
        )
        first = asyncio.run(cp.run_cycle("t1", goal="author a note"))
        assert first.commit is not None and first.commit.status is ArtifactStatus.ACCEPTED

        subprocess.run(["docker", "kill", name], capture_output=True, check=False)  # verifier dies

        second = asyncio.run(cp.run_cycle("t1", goal="author a note"))
        assert second.gate_error is not None  # the unreachable verifier degraded the cycle
        assert second.sandbox_error is None  # it was the gate, not the sandbox
