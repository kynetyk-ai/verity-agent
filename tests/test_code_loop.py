"""End-to-end: real code artifacts through read → propose → gate → commit (Phase 2.4).

Drives the real :class:`ControlPlane` with the stub agent emitting **genuine** Python submissions
and the real :class:`SdkVerifier` running the code domain's gates. The offline tests earn one
verdict for real (the cheap gate runs ``ast.parse``) and prove the wiring with the fake runner; the
``@pytest.mark.docker`` tests earn both verdicts by **actually executing** the code in a container.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tools.harness.sample_code import CLEAN, RAISES, SYNTAX_ERROR
from tools.harness.stub_agent import StubAgent

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    Operation,
    OperationStatus,
    ProposalEnvelope,
    ProviderRegistry,
    SandboxPort,
    VerifierPort,
)
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.commit import CommitOutcome
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import DefaultRetrievalPolicy
from verity.control_plane.store import SqliteStore
from verity.domains.code import DATASET, ENTRYPOINT, SUBMISSION, build_code_domain, declared_objects
from verity.verifier import (
    CodeRunner,
    ContainerCodeRunner,
    FakeCodeRunner,
    RunResult,
    SdkVerifier,
    docker_available,
)


def docker_test(fn):
    """Mark ``docker`` and skip when no Docker daemon is reachable (so CI needs no Docker)."""
    fn = pytest.mark.skipif(not docker_available(), reason="no Docker daemon reachable")(fn)
    return pytest.mark.docker(fn)


def _submission(sub_id: str, code: bytes) -> ProposalEnvelope:
    artifact = Artifact(
        sub_id, SUBMISSION, {"entrypoint": ENTRYPOINT}, ArtifactStatus.PROPOSED, "agent", ""
    )
    op = Operation(f"op-{sub_id}", "submit", ("ds",), sub_id, OperationStatus.SUCCESS, "")
    return ProposalEnvelope(artifact=artifact, operation=op, objects={ENTRYPOINT: code})


def _build(
    tmp_path: Path, *, runner: CodeRunner, steps: list[ProposalEnvelope]
) -> tuple[ControlPlane, StubAgent, SdkVerifier, SqliteStore]:
    domain = build_code_domain(runner)
    store = SqliteStore()
    store.propose(
        Artifact("ds", DATASET, {"n": 10}, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation("op-ds", "load", (), "ds", OperationStatus.SUCCESS, "t0"),
    )
    agent = StubAgent(root=tmp_path / "ws", steps=steps)
    verifier = domain.verifier  # the opaque verifier package (built over the injected runner)
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("stub", lambda: agent)
    vp.register("stub", lambda: verifier)
    cp = ControlPlane(
        store,
        policy=OrchestrationPolicy(stop_on_accept=True),
        sandbox_providers=sp,
        verifier_providers=vp,
    )
    config = TaskConfig(
        task_id="t1",
        instructions="submit a feature as code",
        domain_instructions="a Submission names an entrypoint script written to outbox/",
        schema=domain.schema,
        gated_types=domain.gated_types,
        retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator,
        object_namer=declared_objects,
        sandbox_key="stub",
        verifier_key="stub",
    )
    asyncio.run(cp.configure(config))
    return cp, agent, verifier, store


# ----------------------------------------------------------------- offline (real parse + wiring)


def test_clean_submission_parses_and_commits_with_fake_runner(tmp_path: Path) -> None:
    runner = FakeCodeRunner(script=lambda _r: RunResult(0, "", ""))  # stands in for execution
    cp, _agent, verifier, store = _build(tmp_path, runner=runner, steps=[_submission("s1", CLEAN)])
    result = asyncio.run(cp.run_cycle("t1", goal="submit a feature"))

    assert result.commit is not None and result.commit.outcome is CommitOutcome.ACCEPTED
    assert store.get_artifact("s1") is not None
    # the genuine code object travelled the loop: the verifier got it, and ran the runner on it

    assert any(ENTRYPOINT in req.objects for req in verifier.requests)
    assert runner.calls and runner.calls[0].code == CLEAN


def test_syntax_error_yields_a_refine_before_any_run(tmp_path: Path) -> None:
    runner = FakeCodeRunner(script=lambda _r: RunResult(0, "", ""))  # would accept if ever reached
    cp, _agent, _verifier, store = _build(
        tmp_path, runner=runner, steps=[_submission("s1", SYNTAX_ERROR)]
    )
    result = asyncio.run(cp.run_cycle("t1", goal="submit a feature"))

    # the cheap parse gate (real ast.parse) localizes the syntax error into a fixable defect →
    # refine, not a flat reject; the hard execution gate never runs
    assert result.commit is not None and result.commit.outcome is CommitOutcome.REVISED
    assert result.commit.defects  # the syntax error, named for the agent to fix
    assert runner.calls == []


# ----------------------------------------------------------------- real execution (Docker)


@docker_test
def test_clean_submission_runs_in_a_container_and_commits(tmp_path: Path) -> None:
    cp, _agent, _verifier, store = _build(
        tmp_path, runner=ContainerCodeRunner(), steps=[_submission("s1", CLEAN)]
    )
    result = asyncio.run(cp.run_cycle("t1", goal="submit a feature"))
    # parsed AND executed cleanly in a container — an earned acceptance, end to end
    assert result.commit is not None and result.commit.outcome is CommitOutcome.ACCEPTED
    got = store.get_artifact("s1")
    assert got is not None and got.status is ArtifactStatus.ACCEPTED


@docker_test
def test_raising_submission_is_rejected_by_real_execution(tmp_path: Path) -> None:
    cp, _agent, _verifier, store = _build(
        tmp_path, runner=ContainerCodeRunner(), steps=[_submission("s1", RAISES)]
    )
    result = asyncio.run(cp.run_cycle("t1", goal="submit a feature"))
    # parses fine, but raises when executed — the execution gate earns the rejection
    assert result.commit is not None and result.commit.outcome is CommitOutcome.REJECTED
    got = store.get_artifact("s1")
    assert got is not None and got.status is ArtifactStatus.REJECTED
