"""The container sandbox driver (ROADMAP Phase 3, Sprint 2).

The command-construction + protocol tests need neither Docker nor Deep Agents (the driver only
orchestrates a container). The docker+live integration test runs the real isolated container against
a real model — auto-skipped without Docker, the image, or a key.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    Operation,
    OperationStatus,
    ProviderRegistry,
    SandboxPort,
    VerifierPort,
)
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.commit import CommitOutcome
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import DefaultRetrievalPolicy, OperationSignature
from verity.control_plane.store import SqliteStore
from verity.control_plane.workspace import DefaultLayout
from verity.domains.code import DATASET, build_code_domain
from verity.sandbox.container_driver import DeepAgentsContainerDriver
from verity.sandbox.container_io import CycleInput
from verity.sandbox.registration import build_container_sandbox
from verity.verifier import FakeCodeRunner, RunResult, docker_available

_SANDBOX_IMAGE = "verity-sandbox:latest"


def test_cycle_input_roundtrips() -> None:
    ci = CycleInput(
        system_prompt="SYS",
        user_message="do it",
        operations=(OperationSignature("submit", ("Dataset",), "Submission"),),
        recursion_limit=42,
        deadline_s=1500.0,  # soft wrap-up budget (5.2)
        tool_names=("read_pdf",),  # extra sandbox tools by name (5.4)
    )
    back = CycleInput.from_json(ci.to_json())
    assert back == ci
    assert back.deadline_s == 1500.0 and back.tool_names == ("read_pdf",)


def test_docker_command_has_hostile_posture_with_network(tmp_path: Path) -> None:
    workspace = DefaultLayout().provision(tmp_path / "ws", {})
    driver = DeepAgentsContainerDriver(
        model="anthropic:claude-sonnet-4-6",
        image=_SANDBOX_IMAGE,
        data_sources=("/data/train.csv",),
        env_passthrough=(),  # deterministic: don't depend on the ambient environment
    )
    cmd = driver.docker_command(workspace, tmp_path / "in", name="verity-sbx-ws")
    joined = " ".join(cmd)

    # hardening
    assert "--rm" in cmd and "--read-only" in cmd
    assert "--cap-drop=ALL" in cmd and "--security-opt=no-new-privileges" in cmd
    assert f"--user={os.getuid()}:{os.getgid()}" in cmd
    # outbound network ENABLED (unlike the code runner) so the agent can reach the model API
    assert "--network=none" not in cmd
    # writable workspace, with the read-only roles re-mounted ro on top (gold-data isolation)
    assert f"-v {workspace.root}:/work:rw" in joined
    assert f"{workspace.path_for('data')}:/work/data:ro" in joined
    assert f"{workspace.path_for('spec')}:/work/spec:ro" in joined
    # the task's data source is mounted read-only into data/
    assert "/data/train.csv:/work/data/train.csv:ro" in joined
    # the model travels via env; the entrypoint is the container module
    assert "VERITY_SANDBOX_MODEL=anthropic:claude-sonnet-4-6" in cmd
    assert cmd[-4:] == [_SANDBOX_IMAGE, "python", "-m", "verity.sandbox.container_entry"]


def test_env_passthrough_forwards_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    workspace = DefaultLayout().provision(tmp_path / "ws", {})
    driver = DeepAgentsContainerDriver(model="m", image=_SANDBOX_IMAGE)
    cmd = driver.docker_command(workspace, tmp_path / "in", name="n")
    # the key is forwarded by name (value stays in the daemon env, never on the argv)
    assert cmd.count("ANTHROPIC_API_KEY") == 1 and "sk-test" not in " ".join(cmd)


# ----------------------------------------------------------------- real container (Docker + live)


def _sandbox_image_present() -> bool:
    try:
        done = subprocess.run(
            ["docker", "image", "inspect", _SANDBOX_IMAGE],
            capture_output=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


@pytest.mark.docker
@pytest.mark.live
@pytest.mark.skipif(
    not (docker_available() and _sandbox_image_present() and os.environ.get("ANTHROPIC_API_KEY")),
    reason="needs Docker, the verity-sandbox image, and ANTHROPIC_API_KEY",
)
def test_container_sandbox_commits_a_submission_end_to_end(tmp_path: Path) -> None:
    domain = build_code_domain(FakeCodeRunner(script=lambda _r: RunResult(0, "", "")))
    store = SqliteStore()
    store.propose(
        Artifact("ds", DATASET, {"n": 10}, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation("op-ds", "load", (), "ds", OperationStatus.SUCCESS, "t0"),
    )
    sandbox = build_container_sandbox(
        schema=domain.schema,
        root=tmp_path / "ws",
        model="anthropic:claude-sonnet-4-6",
        image=_SANDBOX_IMAGE,
        id_source=lambda: "sub-1",
    )
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("container", lambda: sandbox)
    vp.register("real", lambda: domain.verifier)
    cp = ControlPlane(
        store,
        policy=OrchestrationPolicy(stop_on_accept=True),
        sandbox_providers=sp,
        verifier_providers=vp,
    )
    config = TaskConfig(
        task_id="t1",
        instructions=(
            "Write submission.py to outbox/ that prints 'ok' (run it to check), then "
            "submit it as a Submission with entrypoint submission.py, using dataset id 'ds' as the "
            "parent."
        ),
        domain_instructions="a Submission names an entrypoint script written to outbox/",
        schema=domain.schema,
        gated_types=domain.gated_types,
        retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator,
        sandbox_key="container",
        verifier_key="real",
    )
    asyncio.run(cp.configure(config))

    result = asyncio.run(cp.run_cycle("t1", goal="submit a feature as code, using dataset ds"))

    assert result.commit is not None and result.commit.outcome is CommitOutcome.ACCEPTED
    got = store.get_artifact("sub-1")
    assert got is not None and got.status is ArtifactStatus.ACCEPTED
