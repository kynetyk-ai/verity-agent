"""Feature-engineering domain — offline wiring, shape, object provisioning, one cycle (Phase 4.1).

No model, no Docker, no LLM. Proves: the §12 schema/gating/shape are wired; the durable-object
provisioning policy selects by status/recency in control-plane-native terms and materializes refs
into a **writable** role that is **re-provisioned every cycle**; and a full ``run_cycle`` commits a
``Submission`` against the control plane, with the prior accepted script provisioned on the next
cycle. The two real ``Submission`` gates land in 4.2; here the verifier is a scripted accept.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from tools.harness.dataset import stratified_split
from tools.harness.stub_verifier import StubVerifier

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    Operation,
    OperationStatus,
    ProviderRegistry,
    SandboxPort,
    ServedContext,
    VerifierPort,
)
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.commit import CommitOutcome, ShapeError
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import (
    DefaultRetrievalPolicy,
    ObjectProvisioningPolicy,
    ObjectProvisionMode,
)
from verity.control_plane.store import SqliteStore
from verity.domains.feature_engineering import (
    DATASET_VERSION,
    ENTRYPOINT,
    FEATURE,
    REQUIREMENTS,
    SUBMISSION,
    build_feature_engineering_domain,
)
from verity.sandbox import AgentSandbox, ProposalDescriptor
from verity.sandbox.descriptor import RESERVED_PROPOSAL_NAME

# --------------------------------------------------------------------------- domain wiring


def test_schema_and_gating_match_section_12() -> None:
    domain = build_feature_engineering_domain()
    types = {t.name: t for t in domain.schema.types()}
    assert types[DATASET_VERSION].is_root is True
    assert types[SUBMISSION].is_root is False and types[FEATURE].is_root is False
    ops = {o.name: o for o in domain.schema.operations()}
    assert ops["submit"].inputs == (DATASET_VERSION,) and ops["submit"].output == SUBMISSION
    assert ops["harvest"].inputs == (SUBMISSION,) and ops["harvest"].output == FEATURE
    assert ops["revises"].inputs == (SUBMISSION,) and ops["revises"].output == SUBMISSION
    # Submission is gated with the incumbents + rejected-log slice; Feature is gated, slice empty.
    assert domain.gated_types.is_gated(SUBMISSION) and domain.gated_types.is_gated(FEATURE)
    assert domain.gated_types.resolve(FEATURE) == frozenset()


def _submission_payload() -> dict[str, object]:
    return {
        "entrypoint": ENTRYPOINT,
        "requirements": REQUIREMENTS,
        "features": [{"name": "u_g", "definition": "u - g", "rationale": "colour index"}],
    }


def _artifact(payload: object, *, type_: str = SUBMISSION) -> Artifact:
    return Artifact("x", type_, payload, ArtifactStatus.PROPOSED, "agent", "t1")


def test_shape_accepts_a_well_formed_submission() -> None:
    domain = build_feature_engineering_domain()
    assert domain.shape_validator(_artifact(_submission_payload())) is None
    # a non-Submission type is not this validator's concern
    assert domain.shape_validator(_artifact({}, type_=DATASET_VERSION)) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("entrypoint"),
        lambda p: p.pop("requirements"),
        lambda p: p.update(features=[]),
        lambda p: p.update(features=[{"name": "x"}]),  # missing definition/rationale
        lambda p: p.update(features=[{"name": "x", "definition": "y", "rationale": ""}]),
    ],
)
def test_shape_rejects_malformed_submissions(mutate: object) -> None:
    domain = build_feature_engineering_domain()
    payload = _submission_payload()
    mutate(payload)  # type: ignore[operator]
    assert isinstance(domain.shape_validator(_artifact(payload)), ShapeError)


def test_more_than_five_features_is_malformed() -> None:
    domain = build_feature_engineering_domain()
    payload = _submission_payload()
    payload["features"] = [
        {"name": f"f{i}", "definition": "d", "rationale": "r"} for i in range(6)
    ]
    assert isinstance(domain.shape_validator(_artifact(payload)), ShapeError)


# --------------------------------------------------------------------------- the split helper

_TINY_CSV = (
    b"id,alpha,class\n"
    b"1,0.1,STAR\n2,0.2,STAR\n3,0.3,STAR\n4,0.4,STAR\n5,0.5,STAR\n"
    b"6,0.6,GALAXY\n7,0.7,GALAXY\n8,0.8,GALAXY\n9,0.9,GALAXY\n10,1.0,GALAXY\n"
)


def test_stratified_split_holds_out_labels_and_drops_target() -> None:
    split = stratified_split(_TINY_CSV, target="class", id_column="id", reserved_fraction=0.6)
    # reserved is larger (per class: 3 of 5 → 6 total), labels held out, agent keeps the rest.
    assert len(split.reserved_labels) == 6
    assert b"class" in split.agent_train_csv
    # the reserved 'test' set drops the target column entirely (the agent's script must predict it)
    assert b"class" not in split.reserved_test_csv.splitlines()[0]
    # stratification: both classes appear in the held-out labels
    assert set(split.reserved_labels.values()) == {"STAR", "GALAXY"}
    # disjoint + total coverage
    assert len(split.reserved_labels) + split.agent_train_csv.count(b"\n") - 1 == 10


def test_stratified_split_is_deterministic() -> None:
    a = stratified_split(_TINY_CSV)
    b = stratified_split(_TINY_CSV)
    assert a.agent_train_csv == b.agent_train_csv and a.reserved_labels == b.reserved_labels


def test_stratified_split_validates_inputs() -> None:
    with pytest.raises(ValueError, match="target column"):
        stratified_split(_TINY_CSV, target="missing")
    with pytest.raises(ValueError, match="reserved_fraction"):
        stratified_split(_TINY_CSV, reserved_fraction=1.5)


# --------------------------------------------------------------------- object-provisioning policy


def _seed_with_object(
    store: SqliteStore, *, art_id: str, status: ArtifactStatus, name: str, data: bytes, ts: str
) -> None:
    ref = store.put_object(data)
    store.propose(
        Artifact(
            art_id, SUBMISSION, {"k": 1}, ArtifactStatus.PROPOSED, "agent", ts,
            objects=((name, ref),),
        ),
        Operation(f"op-{art_id}", "load", (), art_id, OperationStatus.SUCCESS, ts),
    )
    if status is not ArtifactStatus.PROPOSED:
        store.set_status(art_id, status)


def _store_with_three() -> SqliteStore:
    store = SqliteStore()
    _seed_with_object(store, art_id="s1", status=ArtifactStatus.ACCEPTED,
                      name=ENTRYPOINT, data=b"v1", ts="t1")
    _seed_with_object(store, art_id="s2", status=ArtifactStatus.ACCEPTED,
                      name=ENTRYPOINT, data=b"v2", ts="t2")
    _seed_with_object(store, art_id="s3", status=ArtifactStatus.REJECTED,
                      name=ENTRYPOINT, data=b"bad", ts="t3")
    return store


def test_policy_none_provisions_nothing() -> None:
    policy = ObjectProvisioningPolicy(mode=ObjectProvisionMode.NONE)
    assert policy.materialize(_store_with_three()) == {}


def test_policy_last_accepted_takes_the_most_recent_accepted() -> None:
    policy = ObjectProvisioningPolicy(
        mode=ObjectProvisionMode.LAST_ACCEPTED, type_filter=SUBMISSION
    )
    out = policy.materialize(_store_with_three())["scratch"]
    # the rejected s3 is more recent but not accepted; s2 (latest accepted) wins, flat + a manifest.
    assert out["provided/submission.py"] == b"v2"
    assert "provided/INDEX.md" in out and b"s2" in out["provided/INDEX.md"]


def test_policy_all_accepted_ranks_by_recency_with_a_manifest() -> None:
    policy = ObjectProvisioningPolicy(
        mode=ObjectProvisionMode.ALL_ACCEPTED, type_filter=SUBMISSION
    )
    out = policy.materialize(_store_with_three())["scratch"]
    # newest-first ranked subdirs (01 = s2, the most recent accepted), plus the manifest.
    assert out["provided/01-s2/submission.py"] == b"v2"
    assert out["provided/02-s1/submission.py"] == b"v1"
    manifest = out["provided/INDEX.md"].decode()
    assert "| 01 | s2 |" in manifest and "| 02 | s1 |" in manifest  # ordered, meaningful


def test_policy_all_includes_terminal_and_respects_cap() -> None:
    policy = ObjectProvisioningPolicy(
        mode=ObjectProvisionMode.ALL, type_filter=SUBMISSION, max_objects=2
    )
    out = policy.materialize(_store_with_three())["scratch"]
    # the cap bounds object files (2), but the manifest always rides along (metadata, not payload).
    object_files = [k for k in out if not k.endswith("INDEX.md")]
    assert len(object_files) == 2 and "provided/INDEX.md" in out


# ----------------------------------------------------------- sandbox materializes + re-provisions


@dataclass
class _RecordingDriver:
    """A scripted ``SandboxDriver`` that emits a Submission and records what it finds provided."""

    counter: int = 0
    seen_provided: dict[str, bytes] = field(default_factory=dict)

    async def run(
        self, *, system_prompt: str, user_message: str, operations: object,
        workspace: object,
    ) -> None:
        provided = workspace.path_for("scratch") / "provided"  # type: ignore[attr-defined]
        if provided.exists():
            for p in sorted(provided.rglob("*")):
                if p.is_file():
                    self.seen_provided[p.name] = p.read_bytes()
        self.counter += 1
        out = workspace.outbox()  # type: ignore[attr-defined]
        (out / ENTRYPOINT).write_bytes(f"# script v{self.counter}".encode())
        (out / REQUIREMENTS).write_bytes(b"pandas==2.2.2")
        (out / RESERVED_PROPOSAL_NAME).write_bytes(
            ProposalDescriptor("submit", ("ds",), _submission_payload(), metadata="m").to_json()
        )


def test_serve_context_materializes_writable_and_reprovisions(tmp_path: Path) -> None:
    domain = build_feature_engineering_domain()
    sandbox = AgentSandbox(
        root=tmp_path / "ws",
        driver=_RecordingDriver(),  # type: ignore[arg-type]
        schema=domain.schema,
        proposer_identity="fe:test",
    )
    objects = {"scratch": {"provided/prior.py": b"PRIOR"}}

    async def go() -> tuple[bytes, bytes]:
        await sandbox.provision()
        await sandbox.serve_context(ServedContext(system_prompt="S", workspace_objects=objects))
        prior = (tmp_path / "ws" / "scratch" / "provided" / "prior.py").read_bytes()
        # the agent edits the provisioned ref — harmless; the verifier judges the proposal, not it.
        (tmp_path / "ws" / "scratch" / "provided" / "prior.py").write_bytes(b"AGENT EDIT")
        await sandbox.regenerate()
        await sandbox.serve_context(ServedContext(system_prompt="S", workspace_objects=objects))
        refreshed = (tmp_path / "ws" / "scratch" / "provided" / "prior.py").read_bytes()
        return prior, refreshed

    prior, refreshed = asyncio.run(go())
    assert prior == b"PRIOR"
    assert refreshed == b"PRIOR"  # re-provisioned fresh each cycle; the edit did not persist


# ------------------------------------------------------------- full run_cycle through the CP


def _build_cp(
    tmp_path: Path, *, driver: _RecordingDriver, policy: ObjectProvisioningPolicy
) -> tuple[ControlPlane, SqliteStore]:
    store = SqliteStore()
    store.propose(
        Artifact("ds", DATASET_VERSION, {"rows": 10}, ArtifactStatus.PROPOSED, "loader", "t0",
                 is_root=True),
        Operation("op-ds", "load", (), "ds", OperationStatus.SUCCESS, "t0"),
    )
    ids = iter(f"sub-{i}" for i in range(1, 99))
    domain = build_feature_engineering_domain()
    sandbox = AgentSandbox(
        root=tmp_path / "ws",
        driver=driver,  # type: ignore[arg-type]
        schema=domain.schema,
        proposer_identity="fe:test",
        clock=lambda: "t1",
        id_source=lambda: next(ids),
    )
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("fe", lambda: sandbox)
    vp.register("stub", lambda: StubVerifier(identity="fe-verifier"))
    cp = ControlPlane(
        store,
        policy=OrchestrationPolicy(max_cycles=5),
        sandbox_providers=sp,
        verifier_providers=vp,
    )
    config = TaskConfig(
        task_id="fe",
        instructions="design features that improve the model",
        domain_instructions=domain.domain_instructions,
        schema=domain.schema,
        gated_types=domain.gated_types,
        retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator,
        sandbox_key="fe",
        verifier_key="stub",
        object_provisioning=policy,
    )
    asyncio.run(cp.configure(config))
    return cp, store


def test_two_cycles_commit_and_provision_prior_script(tmp_path: Path) -> None:
    driver = _RecordingDriver()
    policy = ObjectProvisioningPolicy(
        mode=ObjectProvisionMode.LAST_ACCEPTED, type_filter=SUBMISSION
    )
    cp, store = _build_cp(tmp_path, driver=driver, policy=policy)

    r1 = asyncio.run(cp.run_cycle("fe", goal="make a submission"))
    assert r1.commit is not None and r1.commit.outcome is CommitOutcome.ACCEPTED
    sub1 = store.get_artifact("sub-1")
    assert sub1 is not None and sub1.type == SUBMISSION and sub1.status is ArtifactStatus.ACCEPTED
    # the harvested script object rode the loop and is recorded as the artifact's sidecar
    assert any(name == ENTRYPOINT for name, _ in sub1.objects)

    # cycle 2: the control plane materializes cycle-1's accepted script under scratch/provided/
    r2 = asyncio.run(cp.run_cycle("fe", goal="improve the submission"))
    assert r2.commit is not None and r2.commit.outcome is CommitOutcome.ACCEPTED
    assert driver.seen_provided.get(ENTRYPOINT) == b"# script v1"
