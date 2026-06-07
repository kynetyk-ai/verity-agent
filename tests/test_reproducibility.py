"""The Phase 2 exit bar: the real verifier renders **reproducible** verdicts (spec §5.8) — 2.5.

Determinism is the property that makes a verdict auditable: the same proposal, slice, and objects
must yield the same verdict every time, on any instance of the verifier. These tests pin it for the
gate-primitive verdicts and for the code domain end to end (with the deterministic fake runner).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tools.harness.sample_code import CLEAN
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
    VerifierRequest,
)
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import DefaultRetrievalPolicy
from verity.control_plane.store import SqliteStore
from verity.domains.code import DATASET, ENTRYPOINT, SUBMISSION, build_code_domain
from verity.domains.fake import NOTE, build_fake_verifier
from verity.verifier import FakeCodeRunner, RunResult, SdkVerifier


def _note(note_id: str, **markers) -> Artifact:
    return Artifact(note_id, NOTE, {"text": "x", **markers}, ArtifactStatus.PROPOSED, "agent", "t1")


def _request(proposal: Artifact, gate: str) -> VerifierRequest:
    return VerifierRequest(proposal=proposal, gate=gate)


# ----------------------------------------------------------------- verdict-level reproducibility


def test_same_request_yields_an_identical_verdict_across_instances() -> None:
    # two independently constructed verifiers must rule identically on identical input (§5.8)
    request = _request(_note("n1", defect="tone"), "well-formed")
    v1 = asyncio.run(build_fake_verifier().dispatch(request))
    v2 = asyncio.run(build_fake_verifier().dispatch(request))
    assert v1 == v2  # GateVerdict is a frozen value — equality is structural
    assert v1 is not None and v1.defects == ("tone",)


def test_repeated_dispatch_is_stable() -> None:
    verifier = build_fake_verifier()
    request = _request(_note("n1", keep=False), "worth-keeping")
    verdicts = [asyncio.run(verifier.dispatch(request)) for _ in range(5)]
    assert len({(v.kind, v.rationale) for v in verdicts if v is not None}) == 1


# ----------------------------------------------------------------- end-to-end reproducibility


def _run_code_submission(tmp_path: Path, code: bytes) -> str:
    """Run one Submission of ``code`` through a fresh stack with a deterministic runner."""
    domain = build_code_domain(FakeCodeRunner(script=lambda _r: RunResult(0, "", "")))
    store = SqliteStore()
    store.propose(
        Artifact("ds", DATASET, {"n": 1}, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation("op-ds", "load", (), "ds", OperationStatus.SUCCESS, "t0"),
    )
    submission = ProposalEnvelope(
        artifact=Artifact("s1", SUBMISSION, {"entrypoint": ENTRYPOINT}, ArtifactStatus.PROPOSED,
                          "agent", ""),
        operation=Operation("op-s1", "submit", ("ds",), "s1", OperationStatus.SUCCESS, ""),
        objects={ENTRYPOINT: code},
    )
    agent = StubAgent(root=tmp_path, steps=[submission])
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("stub", lambda: agent)
    vp.register("stub", lambda: SdkVerifier(plugins=domain.plugins))
    cp = ControlPlane(
        store, policy=OrchestrationPolicy(stop_on_accept=True), sandbox_providers=sp,
        verifier_providers=vp,
    )
    config = TaskConfig(
        task_id="t1", instructions="submit", domain_instructions="entrypoint script",
        schema=domain.schema, gates=domain.gates, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator, sandbox_key="stub", verifier_key="stub",
    )
    asyncio.run(cp.configure(config))
    result = asyncio.run(cp.run_cycle("t1", goal="submit a feature"))
    assert result.commit is not None
    return result.commit.outcome.value


def test_code_loop_outcome_is_reproducible(tmp_path: Path) -> None:
    # the same submission through two independent stacks reaches the same committed outcome
    first = _run_code_submission(tmp_path / "a", CLEAN)
    second = _run_code_submission(tmp_path / "b", CLEAN)
    assert first == second == "accepted"
