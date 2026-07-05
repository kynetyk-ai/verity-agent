"""Live model-breadth smokes on the PRODUCTION sandbox driver (Phase 6 / ROADMAP 6.1–6.2).

One end-to-end submission cycle in a real isolated worker — the same task, verifier, and
assertions across three model targets (native Anthropic; a host-local OpenAI-compatible server; a
hosted OpenAI model): the model seam is the only variable. Ported from the retired
legacy-container-driver test module (#129) onto `BackendSandboxDriver` + `DockerBackend`, the
substrate every production task type runs on. Auto-skipped without Docker, the image, or a key.
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
from verity.control_plane.registries import DefaultRetrievalPolicy
from verity.control_plane.store import SqliteStore
from verity.domains.code import DATASET, build_code_domain, declared_objects
from verity.provisioning import DockerBackend
from verity.sandbox import AgentSandbox
from verity.sandbox.backend_driver import BackendSandboxDriver
from verity.sandbox.model_spec import ModelSpec
from verity.sandbox.providers import DEFAULT_LOCAL_BASE_URL, local_spec, openai_spec
from verity.sandbox.registration import build_sandbox
from verity.verifier import FakeCodeRunner, RunResult, docker_available

_SANDBOX_IMAGE = "verity-sandbox:latest"


def _sandbox_image_present() -> bool:
    try:
        done = subprocess.run(
            ["docker", "image", "inspect", _SANDBOX_IMAGE],
            capture_output=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _build_worker_sandbox(
    root: Path, *, model: str = "unused", spec: ModelSpec | None = None
) -> AgentSandbox:
    """The production posture: `AgentSandbox` over `BackendSandboxDriver` on the Docker backend."""
    schema = build_code_domain(FakeCodeRunner(script=lambda _r: RunResult(0, "", ""))).schema
    return build_sandbox(
        schema=schema,
        root=root,
        driver=BackendSandboxDriver(
            backend=DockerBackend(), model=model, spec=spec, image=_SANDBOX_IMAGE,
            config="live-smoke", timeout_s=600.0,
        ),
        proposer_identity=f"deepagents-worker:{spec.provider_string() if spec else model}",
        id_source=lambda: "sub-1",
    )


def _assert_worker_commits_submission(sandbox: AgentSandbox) -> None:
    """Drive one end-to-end submission cycle through ``sandbox`` and assert it commits ACCEPTED.

    Shared by all three live smokes — the only thing that differs is the model behind the sandbox;
    the task, verifier, and assertions are identical.
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
        object_namer=declared_objects,
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
def test_worker_sandbox_commits_a_submission_end_to_end(tmp_path: Path) -> None:
    sandbox = _build_worker_sandbox(tmp_path / "ws", model="anthropic:claude-sonnet-4-6")
    _assert_worker_commits_submission(sandbox)


@pytest.mark.docker
@pytest.mark.live
@pytest.mark.skipif(
    not (docker_available() and _sandbox_image_present() and os.environ.get("VERITY_LOCAL_MODEL")),
    reason="needs Docker, the verity-sandbox image, a host-local OpenAI-compatible server, and "
    "VERITY_LOCAL_MODEL set to the served model name (optionally VERITY_LOCAL_MODEL_BASE_URL)",
)
def test_worker_sandbox_commits_via_local_model(tmp_path: Path) -> None:
    # The thesis smoke (6.1): the same submission cycle, driven by a host-local OpenAI-compatible
    # model reached over host.docker.internal. Opt-in via env (any vLLM/Ollama/llama.cpp server).
    base_url = os.environ.get("VERITY_LOCAL_MODEL_BASE_URL", DEFAULT_LOCAL_BASE_URL)
    sandbox = _build_worker_sandbox(
        tmp_path / "ws", spec=local_spec(os.environ["VERITY_LOCAL_MODEL"], base_url=base_url)
    )
    _assert_worker_commits_submission(sandbox)


@pytest.mark.docker
@pytest.mark.live
@pytest.mark.skipif(
    not (docker_available() and _sandbox_image_present() and os.environ.get("OPENAI_API_KEY")),
    reason="needs Docker, the verity-sandbox image, and OPENAI_API_KEY",
)
def test_worker_sandbox_commits_via_openai(tmp_path: Path) -> None:
    # Hosted smoke (6.2): the same submission cycle on a cheaper hosted OpenAI model. Ready to run
    # the moment a key exists; auto-skips otherwise.
    model = os.environ.get("VERITY_OPENAI_MODEL", "gpt-4o-mini")
    sandbox = _build_worker_sandbox(tmp_path / "ws", spec=openai_spec(model))
    _assert_worker_commits_submission(sandbox)
