"""Control-plane §13 criteria, demonstrated against the **real** verifier (ROADMAP Phase 1/2.5).

Drives the real :class:`ControlPlane` and the deterministic fake domain on the **real**
:class:`~verity.verifier.SdkVerifier` (Phase 2.5 retired the stub verifier from the happy path; it
survives only as a port-protocol double, asserted below). The stub agent/workspace stands in for the
sandbox. Each control-plane-level acceptance criterion is a runnable check.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tools.harness.stub_agent import StubAgent
from tools.harness.stub_verifier import StubVerifier

from verity.contracts import (
    ProposalEnvelope,
    ProviderRegistry,
    SandboxPort,
    VerifierPort,
)
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.commit import CommitOutcome, NoImplicitAccept
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import DefaultRetrievalPolicy
from verity.control_plane.store import (
    Artifact,
    ArtifactStatus,
    Operation,
    OperationStatus,
    SqliteStore,
    VerdictKind,
)
from verity.domains.fake import NOTE, ORPHAN, SOURCE, build_fake_domain, build_fake_verifier
from verity.verifier import SdkVerifier

# --------------------------------------------------------------------------- helpers


def _config(domain) -> TaskConfig:
    return TaskConfig(
        task_id="t1",
        instructions="produce good notes",
        domain_instructions="a Note has a text field",
        schema=domain.schema,
        gates=domain.gates,
        retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator,
        sandbox_key="stub",
        verifier_key="stub",
    )


def _env(
    artifact_id: str,
    *,
    artifact_type: str = NOTE,
    payload: dict | None = None,
    metadata: str = "",
    objects: dict[str, bytes] | None = None,
    parents: tuple[str, ...] = ("src",),
    op_name: str = "author",
) -> ProposalEnvelope:
    if payload is None:
        payload = {"text": "hi"} if artifact_type == NOTE else {"x": 1}
    artifact = Artifact(artifact_id, artifact_type, payload, ArtifactStatus.PROPOSED, "agent", "")
    op = Operation(f"op-{artifact_id}", op_name, parents, artifact_id, OperationStatus.SUCCESS, "")
    return ProposalEnvelope(
        artifact=artifact, operation=op, metadata=metadata, objects=objects or {}
    )


def _build(
    tmp_path: Path,
    *,
    steps: list[ProposalEnvelope],
    policy: OrchestrationPolicy | None = None,
) -> tuple[ControlPlane, StubAgent, SdkVerifier, SqliteStore]:
    store = SqliteStore()
    store.propose(
        Artifact("src", SOURCE, {"raw": 1}, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation("op-src", "load", (), "src", OperationStatus.SUCCESS, "t0"),
    )
    agent = StubAgent(root=tmp_path / "ws", steps=steps)
    verifier = build_fake_verifier()  # the REAL verifier, not the stub (§2.5)
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("stub", lambda: agent)
    vp.register("stub", lambda: verifier)
    cp = ControlPlane(
        store,
        policy=policy or OrchestrationPolicy(stop_on_accept=True),
        sandbox_providers=sp,
        verifier_providers=vp,
    )
    asyncio.run(cp.configure(_config(build_fake_domain())))
    return cp, agent, verifier, store


# --------------------------------------------------------------------------- the doubles


def test_stubs_satisfy_their_port_protocols(tmp_path: Path) -> None:
    agent = StubAgent(root=tmp_path / "ws", steps=[])
    assert isinstance(agent, SandboxPort)
    assert isinstance(StubVerifier(), VerifierPort)


# --------------------------------------------------------------------------- §13 subset


def test_propose_gate_commit(tmp_path: Path) -> None:
    cp, agent, _verifier, store = _build(tmp_path, steps=[_env("n1")])
    results = asyncio.run(cp.run("t1", goal="make a note"))
    assert results[-1].commit is not None
    assert results[-1].commit.outcome is CommitOutcome.ACCEPTED
    got = store.get_artifact("n1")
    assert got is not None and got.status is ArtifactStatus.ACCEPTED


def test_shape_error_is_correctable_and_unrecorded(tmp_path: Path) -> None:
    # first step is malformed (no 'text'); second is the correction
    steps = [_env("n1", payload={"oops": 1}), _env("n1", payload={"text": "fixed"})]
    cp, agent, verifier, store = _build(tmp_path, steps=steps)
    results = asyncio.run(cp.run("t1", goal="make a note"))

    assert results[0].entered_protocol is False
    assert results[0].shape_error is not None
    # nothing recorded for the malformed attempt: no harvest, and crucially no trial-count entry —
    # a shape-error never becomes a rejection (§7.0). (No `proposed` row either: intake returns
    # before proposing.)
    assert results[0].harvested == {}
    assert store.rejected_log() == []
    assert "shape-error" in agent.served[1].feedback  # the agent received the correction
    # the corrected resubmission then proceeds normally
    assert results[-1].commit is not None and results[-1].commit.outcome is CommitOutcome.ACCEPTED


def test_refine_yields_tracked_revision_with_intact_lineage(tmp_path: Path) -> None:
    # the 'defect' marker drives the real well-formed gate to refine n1; n2 is its clean revision
    steps = [
        _env("n1", payload={"text": "hi", "defect": "tone"}),
        _env("n2", parents=("n1",), op_name="revises"),
    ]
    cp, agent, _verifier, store = _build(tmp_path, steps=steps)
    asyncio.run(cp.run("t1", goal="make a note"))

    n1 = store.get_artifact("n1")
    n2 = store.get_artifact("n2")
    assert n1 is not None and n1.status is ArtifactStatus.REVISED
    assert n1.revised_by == "n2"  # revised_by set (§7.5b)
    assert n2 is not None and n2.status is ArtifactStatus.ACCEPTED
    assert "refine" in agent.served[1].feedback  # the agent received the defect feedback
    # a revises operation carries the lineage; provenance spans the revision (§13.8)
    assert any(op.op_name == "revises" for op in store.operations_into("n2"))
    prov = cp.provenance("n2")
    assert "n1" in {a.id for a in prov.ancestors}
    # the refine defect was recorded on n1's decision
    assert any(d.defects == ("tone",) for d in store.decisions_for("n1"))


def test_object_harvested_before_teardown_and_round_trips(tmp_path: Path) -> None:
    code = b"def feature(df):\n    return df['a'] * 2\n"
    cp, agent, verifier, store = _build(
        tmp_path, steps=[_env("n1", objects={"submission.py": code})]
    )
    result = asyncio.run(cp.run_cycle("t1", goal="make a note"))

    # harvested, content-addressed, and retrievable by reference (§13.10)
    assert "submission.py" in result.harvested
    ref = result.harvested["submission.py"]
    assert store.get_object(ref.content_hash) == code
    # the verifier was handed the object attachment to (in v1) execute
    assert any("submission.py" in req.objects for req in verifier.requests)
    # teardown discarded the workspace: the outbox is empty after regeneration, yet the object
    # survives in the store — harvest happened before teardown
    assert list(agent.workspace.outbox().iterdir()) == []
    assert agent.regenerations == 1


def test_no_implicit_accept_refusal(tmp_path: Path) -> None:
    cp, _agent, _verifier, store = _build(
        tmp_path, steps=[_env("orph", artifact_type=ORPHAN, payload={"x": 1})]
    )
    with pytest.raises(NoImplicitAccept):
        asyncio.run(cp.submit_proposal("t1", _env("orph", artifact_type=ORPHAN, payload={"x": 1})))
    got = store.get_artifact("orph")
    # the gateless type was refused, never accepted
    assert got is not None and got.status is ArtifactStatus.PROPOSED


def test_privileged_mutator_boundary_holds(tmp_path: Path) -> None:
    _cp, agent, _verifier, _store = _build(tmp_path, steps=[])
    # the sandbox has no write path to the store and no path to the verifier (§3.3, §13.9)
    forbidden = [
        "set_status",
        "record_decision",
        "accept_superseding",
        "link_revision",
        "propose",
        "put_object",
        "dispatch",
    ]
    for name in forbidden:
        assert not hasattr(agent, name)
    assert not hasattr(agent, "store")
    assert not hasattr(agent, "verifier")


def test_bounded_context_through_the_control_plane(tmp_path: Path) -> None:
    cp, _agent, _verifier, store = _build(tmp_path, steps=[])
    counter = 0

    def size_after(n: int) -> int:
        nonlocal counter
        for _ in range(n):
            aid = f"x{counter:06d}"
            stamp = f"t{counter:06d}"
            counter += 1
            store.propose(
                Artifact(aid, "Thing", {"v": 1}, ArtifactStatus.PROPOSED, "loader", stamp),
                Operation(f"op-{aid}", "load", (), aid, OperationStatus.SUCCESS, stamp),
            )
        served = asyncio.run(cp.serve_context("t1", goal="g"))
        return len(served.system_prompt) + len(served.tail)

    # both stores are already past the manifest cap, so this measures growth, not fill-up
    small = size_after(100)
    big = size_after(2000)
    # 20x more artifacts, but the served context does not grow with store size (§13.5)
    assert abs(big - small) < 80


def test_why_do_we_believe_x_is_answerable_from_provenance(tmp_path: Path) -> None:
    cp, _agent, _verifier, _store = _build(tmp_path, steps=[_env("n1", metadata="i reasoned thus")])
    asyncio.run(cp.run("t1", goal="make a note"))

    prov = cp.provenance("n1")
    assert prov.artifact.id == "n1"
    assert "src" in {a.id for a in prov.ancestors}  # lineage back to the root input
    assert any(d.verdict is VerdictKind.ACCEPT for d in prov.decisions)  # the gate decisions
    # the agent's rationale is retained as provenance/context, separate from the gate inputs
    assert cp.rationale_for("n1") == "i reasoned thus"
