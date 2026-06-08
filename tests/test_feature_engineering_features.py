"""Harvested Features, the grounding gate, and refine-on-a-bad-feature (Phase 4.3) — offline.

Proves the §12 harvested-``Feature`` machinery: the harness mints one Feature per declared feature
via a ``harvest`` operation when a submission is accepted; each clears its own cheap **grounding**
gate (genuinely defined by the code) or is refused (§5.7, §13.6); and a described-but-absent feature
drives the submission gate to **refine**, yielding a tracked revision with intact lineage (§13.8).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    Operation,
    OperationStatus,
    ProviderRegistry,
    SandboxPort,
    VerifierPort,
    VerifierRequest,
)
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.commit import CommitOutcome
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import DefaultRetrievalPolicy
from verity.control_plane.store import SqliteStore
from verity.domains.feature_engineering import (
    DATASET_VERSION,
    ENTRYPOINT,
    FEATURE,
    REQUIREMENTS,
    SUBMISSION,
    build_feature_engineering_domain,
    build_feature_engineering_verifier,
)
from verity.sandbox import AgentSandbox, ProposalDescriptor
from verity.sandbox.descriptor import RESERVED_PROPOSAL_NAME
from verity.verifier import FakeCodeRunner, RunResult

RESERVED = {"1": "STAR", "2": "GALAXY", "3": "QSO"}


def _preds_csv(preds: dict[str, str]) -> bytes:
    return ("id,class\n" + "\n".join(f"{k},{v}" for k, v in preds.items()) + "\n").encode()


def _verifier(runner: FakeCodeRunner):
    return build_feature_engineering_verifier(
        runner,
        agent_train_csv=b"id,class\n0,STAR\n",
        reserved_test_csv=b"id\n1\n2\n3\n",
        reserved_labels=RESERVED,
    )


# --------------------------------------------------------------------------- grounding (Feature)


def _feature_request(name: str, code: bytes) -> VerifierRequest:
    proposal = Artifact(
        "feat", FEATURE, {"name": name, "definition": "d", "rationale": "r"},
        ArtifactStatus.PROPOSED, "agent", "t1",
    )
    return VerifierRequest(proposal=proposal, store_slice=(), objects={ENTRYPOINT: code})


def test_grounding_accepts_a_feature_defined_in_the_code() -> None:
    verifier = _verifier(FakeCodeRunner())
    bundle = asyncio.run(verifier.dispatch(_feature_request("u_g", b"df['u_g'] = df.u - df.g")))
    assert bundle.status is ArtifactStatus.ACCEPTED
    assert bundle.decisions[-1].gate == "grounding"


def test_grounding_rejects_a_described_but_absent_feature() -> None:
    verifier = _verifier(FakeCodeRunner())
    bundle = asyncio.run(verifier.dispatch(_feature_request("ghost", b"# nothing here")))
    assert bundle.status is ArtifactStatus.REJECTED  # no implicit accept for Features (§13.6)
    assert "not defined" in bundle.decisions[-1].rationale


# ----------------------------------------------------- features-defined refine (Submission)


def test_submission_refines_when_a_declared_feature_is_absent() -> None:
    runner = FakeCodeRunner(script=lambda _r: RunResult(0, "", "", output=_preds_csv(RESERVED)))
    verifier = _verifier(runner)
    proposal = Artifact(
        "s", SUBMISSION,
        {"entrypoint": ENTRYPOINT, "requirements": REQUIREMENTS,
         "features": [{"name": "ghost", "definition": "d", "rationale": "r"}]},
        ArtifactStatus.PROPOSED, "agent", "t1",
    )
    request = VerifierRequest(
        proposal=proposal, store_slice=(),
        objects={ENTRYPOINT: b"# a clean script that adds no new columns", REQUIREMENTS: b"x"},
    )
    bundle = asyncio.run(verifier.dispatch(request))
    # the code runs clean, but one feature is absent -> a localized refine, not a reject
    assert bundle.status is ArtifactStatus.REVISED
    assert bundle.decisions[-1].defects == ("ghost",)


# --------------------------------------------------------------------- harvest + refine end-to-end


@dataclass
class _Driver:
    """Emits each scripted (op_name, parents, code, feature-names) proposal in turn."""

    steps: list[tuple[str, tuple[str, ...], bytes, list[str]]]
    cursor: int = 0

    async def run(
        self, *, system_prompt: str, user_message: str, operations: object, workspace: object,
    ) -> None:
        op_name, parents, code, names = self.steps[self.cursor]
        self.cursor += 1
        out = workspace.outbox()  # type: ignore[attr-defined]
        (out / ENTRYPOINT).write_bytes(code)
        (out / REQUIREMENTS).write_bytes(b"pandas==2.2.2")
        features = [{"name": n, "definition": "d", "rationale": "r"} for n in names]
        payload = {"entrypoint": ENTRYPOINT, "requirements": REQUIREMENTS, "features": features}
        (out / RESERVED_PROPOSAL_NAME).write_bytes(
            ProposalDescriptor(op_name, parents, payload, metadata="why").to_json()
        )


def _build(
    tmp_path: Path, *, steps: list[tuple[str, tuple[str, ...], bytes, list[str]]]
) -> tuple[ControlPlane, SqliteStore]:
    store = SqliteStore()
    store.propose(
        Artifact("ds", DATASET_VERSION, {"rows": 3}, ArtifactStatus.PROPOSED, "loader", "t0",
                 is_root=True),
        Operation("op-ds", "load", (), "ds", OperationStatus.SUCCESS, "t0"),
    )
    ids = iter(f"sub-{i}" for i in range(1, 99))
    domain = build_feature_engineering_domain()
    runner = FakeCodeRunner(script=lambda _r: RunResult(0, "", "", output=_preds_csv(RESERVED)))
    sandbox = AgentSandbox(
        root=tmp_path / "ws", driver=_Driver(steps),  # type: ignore[arg-type]
        schema=domain.schema, proposer_identity="fe:agent",
        clock=lambda: "t1", id_source=lambda: next(ids),
    )
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("fe", lambda: sandbox)
    vp.register("fe", lambda: _verifier(runner))
    cp = ControlPlane(
        store, policy=OrchestrationPolicy(max_cycles=5),
        sandbox_providers=sp, verifier_providers=vp,
    )
    config = TaskConfig(
        task_id="fe", instructions="design features",
        domain_instructions=domain.domain_instructions, schema=domain.schema,
        gated_types=domain.gated_types, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator, sandbox_key="fe", verifier_key="fe",
        harvester=domain.harvester,
    )
    asyncio.run(cp.configure(config))
    return cp, store


def test_accepted_submission_harvests_grounded_features(tmp_path: Path) -> None:
    code = b"df['u_g'] = df.u - df.g\ndf['redshift2'] = df.redshift ** 2"
    cp, store = _build(
        tmp_path, steps=[("submit", ("ds",), code, ["u_g", "redshift2"])]
    )
    result = asyncio.run(cp.run_cycle("fe", goal="make features"))
    assert result.commit is not None and result.commit.outcome is CommitOutcome.ACCEPTED

    # one Feature per declared feature, minted via a harvest op linking back to the submission
    features = store.query_artifacts(type=FEATURE)
    assert {f.id for f in features} == {"sub-1::feature::0", "sub-1::feature::1"}
    for feature in features:
        assert feature.status is ArtifactStatus.ACCEPTED  # each grounded in the code
        harvest_ops = store.operations_into(feature.id)
        assert any(op.op_name == "harvest" and op.parents == ("sub-1",) for op in harvest_ops)
        # the child carries the parent's code provenance as its sidecar
        assert any(name == ENTRYPOINT for name, _ in feature.objects)
    # §13.3 — the harvested features are reachable in the submission's provenance lineage
    prov = cp.provenance("sub-1::feature::0")
    assert "sub-1" in {a.id for a in prov.ancestors} and "ds" in {a.id for a in prov.ancestors}


def test_refine_on_a_bad_feature_yields_a_tracked_revision(tmp_path: Path) -> None:
    # cycle 1 declares a feature the code never defines -> refine; cycle 2 revises it in.
    cp, store = _build(
        tmp_path,
        steps=[
            ("submit", ("ds",), b"# a clean script that adds no new columns", ["wedge"]),
            ("revises", ("sub-1",), b"df['wedge'] = df.u - 2*df.g + df.r", ["wedge"]),
        ],
    )
    r1 = asyncio.run(cp.run_cycle("fe", goal="cycle 1"))
    r2 = asyncio.run(cp.run_cycle("fe", goal="cycle 2"))

    assert r1.commit is not None and r1.commit.outcome is CommitOutcome.REVISED
    assert r1.commit.defects == ("wedge",)
    assert r2.commit is not None and r2.commit.outcome is CommitOutcome.ACCEPTED

    # §13.8 — tracked revision with intact lineage across the revises operation
    sub1, sub2 = store.get_artifact("sub-1"), store.get_artifact("sub-2")
    assert sub1 is not None and sub1.status is ArtifactStatus.REVISED and sub1.revised_by == "sub-2"
    assert sub2 is not None and sub2.status is ArtifactStatus.ACCEPTED
    assert any(op.op_name == "revises" for op in store.operations_into("sub-2"))
    prov = cp.provenance("sub-2")
    assert "sub-1" in {a.id for a in prov.ancestors}
    # the refine defect was recorded on the revised submission's decision
    assert any(d.defects == ("wedge",) for d in store.decisions_for("sub-1"))
    # the accepted revision harvested its (now-grounded) feature
    assert store.get_artifact("sub-2::feature::0") is not None
