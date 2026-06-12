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
from verity.sandbox.errors import SandboxError
from verity.sandbox.model_spec import ModelSpec
from verity.sandbox.providers import DEFAULT_LOCAL_BASE_URL, local_spec, openai_spec
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
        step_budget=30,  # per-cycle model-step budget (5.1)
    )
    back = CycleInput.from_json(ci.to_json())
    assert back == ci
    assert back.deadline_s == 1500.0 and back.tool_names == ("read_pdf",)
    assert back.step_budget == 30


def test_a_missing_docker_binary_degrades_to_a_sandbox_error() -> None:
    # No daemon needed: a non-existent binary makes create_subprocess_exec raise OSError, which the
    # driver types as a recoverable SandboxError instead of a raw OSError aborting the run (5.1).
    driver = DeepAgentsContainerDriver(model="m", docker_bin="verity-no-such-docker-binary-xyz")
    with pytest.raises(SandboxError, match="could not launch the sandbox container"):
        asyncio.run(driver._run_container([driver.docker_bin, "run"], name="t"))


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


def test_docker_command_anthropic_spec_is_unchanged(tmp_path: Path) -> None:
    # A plain provider string (no ModelSpec) must produce today's exact model env and NO --add-host,
    # so Phase 6 leaves the Anthropic path byte-identical (guards the posture test above).
    workspace = DefaultLayout().provision(tmp_path / "ws", {})
    driver = DeepAgentsContainerDriver(model="anthropic:claude-sonnet-4-6", image=_SANDBOX_IMAGE)
    cmd = driver.docker_command(workspace, tmp_path / "in", name="n")
    assert "VERITY_SANDBOX_MODEL=anthropic:claude-sonnet-4-6" in cmd
    assert "--add-host=host.docker.internal:host-gateway" not in cmd
    assert not any(a.startswith("VERITY_SANDBOX_BASE_URL=") for a in cmd)


def test_docker_command_local_spec_adds_host_gateway_and_forwards_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An OpenAI-compatible local endpoint: the container needs --add-host to reach the host server,
    # the base_url rides as an env var, and the spec's own key env is forwarded by name (5.1/6.1).
    monkeypatch.setenv("LOCAL_KEY", "sk-local")
    workspace = DefaultLayout().provision(tmp_path / "ws", {})
    spec = ModelSpec(
        "openai-compatible", "qwen2.5-coder",
        base_url="http://host.docker.internal:8000/v1", api_key_env="LOCAL_KEY",
    )
    driver = DeepAgentsContainerDriver(model="unused", spec=spec, image=_SANDBOX_IMAGE)
    cmd = driver.docker_command(workspace, tmp_path / "in", name="n")
    joined = " ".join(cmd)
    assert "--add-host=host.docker.internal:host-gateway" in cmd
    assert "VERITY_SANDBOX_MODEL=openai-compatible:qwen2.5-coder" in cmd
    assert "VERITY_SANDBOX_BASE_URL=http://host.docker.internal:8000/v1" in cmd
    # the key is forwarded by name only (value stays in the daemon env)
    assert cmd.count("LOCAL_KEY") == 1 and "sk-local" not in joined


def test_docker_command_maps_dmr_gateway_host(tmp_path: Path) -> None:
    # Docker Model Runner's container hostname (model-runner.docker.internal) is also a
    # *.docker.internal host -> the driver maps THAT host, not a hardcoded host.docker.internal.
    workspace = DefaultLayout().provision(tmp_path / "ws", {})
    spec = ModelSpec(
        "openai-compatible", "qwen3", base_url="http://model-runner.docker.internal/engines/v1"
    )
    driver = DeepAgentsContainerDriver(model="unused", spec=spec, image=_SANDBOX_IMAGE)
    cmd = driver.docker_command(workspace, tmp_path / "in", name="n")
    assert "--add-host=model-runner.docker.internal:host-gateway" in cmd
    assert "--add-host=host.docker.internal:host-gateway" not in cmd


def test_docker_command_openai_spec_forwards_key_no_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A cheaper hosted OpenAI model (6.2): native openai:<model> path, OPENAI_API_KEY forwarded by
    # name, public endpoint so NO --add-host and NO base_url env.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    workspace = DefaultLayout().provision(tmp_path / "ws", {})
    driver = DeepAgentsContainerDriver(
        model="unused", spec=openai_spec("gpt-4o-mini"), image=_SANDBOX_IMAGE
    )
    cmd = driver.docker_command(workspace, tmp_path / "in", name="n")
    assert "VERITY_SANDBOX_MODEL=openai:gpt-4o-mini" in cmd
    assert "--add-host=host.docker.internal:host-gateway" not in cmd
    assert not any(a.startswith("VERITY_SANDBOX_BASE_URL=") for a in cmd)
    assert cmd.count("OPENAI_API_KEY") == 1 and "sk-openai" not in " ".join(cmd)


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


def _assert_container_commits_submission(sandbox: object) -> None:
    """Drive one end-to-end submission cycle through ``sandbox`` and assert it commits ACCEPTED.

    Shared by the Anthropic and local-model live smokes — the only thing that differs is the model
    behind the sandbox; the task, verifier, and assertions are identical.
    """
    domain = build_code_domain(FakeCodeRunner(script=lambda _r: RunResult(0, "", "")))
    store = SqliteStore()
    store.propose(
        Artifact("ds", DATASET, {"n": 10}, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation("op-ds", "load", (), "ds", OperationStatus.SUCCESS, "t0"),
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


@pytest.mark.docker
@pytest.mark.live
@pytest.mark.skipif(
    not (docker_available() and _sandbox_image_present() and os.environ.get("ANTHROPIC_API_KEY")),
    reason="needs Docker, the verity-sandbox image, and ANTHROPIC_API_KEY",
)
def test_container_sandbox_commits_a_submission_end_to_end(tmp_path: Path) -> None:
    sandbox = build_container_sandbox(
        schema=build_code_domain(FakeCodeRunner(script=lambda _r: RunResult(0, "", ""))).schema,
        root=tmp_path / "ws",
        model="anthropic:claude-sonnet-4-6",
        image=_SANDBOX_IMAGE,
        id_source=lambda: "sub-1",
    )
    _assert_container_commits_submission(sandbox)


@pytest.mark.docker
@pytest.mark.live
@pytest.mark.skipif(
    not (docker_available() and _sandbox_image_present() and os.environ.get("VERITY_LOCAL_MODEL")),
    reason="needs Docker, the verity-sandbox image, a host-local OpenAI-compatible server, and "
    "VERITY_LOCAL_MODEL set to the served model name (optionally VERITY_LOCAL_MODEL_BASE_URL)",
)
def test_container_sandbox_commits_via_local_model(tmp_path: Path) -> None:
    # The thesis smoke (6.1): the same submission cycle, driven by a host-local OpenAI-compatible
    # model reached over host.docker.internal. Opt-in via env (any vLLM/Ollama/llama.cpp server).
    base_url = os.environ.get("VERITY_LOCAL_MODEL_BASE_URL", DEFAULT_LOCAL_BASE_URL)
    sandbox = build_container_sandbox(
        schema=build_code_domain(FakeCodeRunner(script=lambda _r: RunResult(0, "", ""))).schema,
        root=tmp_path / "ws",
        model="unused",
        model_spec=local_spec(os.environ["VERITY_LOCAL_MODEL"], base_url=base_url),
        image=_SANDBOX_IMAGE,
        id_source=lambda: "sub-1",
    )
    _assert_container_commits_submission(sandbox)


@pytest.mark.docker
@pytest.mark.live
@pytest.mark.skipif(
    not (docker_available() and _sandbox_image_present() and os.environ.get("OPENAI_API_KEY")),
    reason="needs Docker, the verity-sandbox image, and OPENAI_API_KEY",
)
def test_container_sandbox_commits_via_openai(tmp_path: Path) -> None:
    # Deferred hosted smoke (6.2): the same submission cycle on a cheaper hosted OpenAI model. Ready
    # to run the moment a key exists; auto-skips today (no live OpenAI call in the Phase 6 work).
    model = os.environ.get("VERITY_OPENAI_MODEL", "gpt-4o-mini")
    sandbox = build_container_sandbox(
        schema=build_code_domain(FakeCodeRunner(script=lambda _r: RunResult(0, "", ""))).schema,
        root=tmp_path / "ws",
        model="unused",
        model_spec=openai_spec(model),
        image=_SANDBOX_IMAGE,
        id_source=lambda: "sub-1",
    )
    _assert_container_commits_submission(sandbox)
