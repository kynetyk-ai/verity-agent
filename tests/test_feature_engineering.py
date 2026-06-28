"""Feature-engineering domain — offline wiring, shape, object provisioning, one cycle (Phase 4.1).

No model, no Docker, no LLM. Proves: the §12 schema/gating/shape are wired; the durable-object
provisioning policy selects by status/recency in control-plane-native terms and materializes refs
into a **writable** role that is **re-provisioned every cycle**; and a full ``run_cycle`` commits a
``Submission`` against the control plane, with the prior accepted script provisioned on the next
cycle. The two real ``Submission`` gates land in 4.2; here the verifier is a scripted accept.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from tools.harness.dataset import stratified_split
from tools.harness.stub_verifier import StubVerifier

from verity.composition.fe_kaggle import configure_fe_kaggle_task
from verity.contracts import (
    Artifact,
    ArtifactStatus,
    Operation,
    OperationStatus,
    ProviderRegistry,
    SandboxPort,
    ServedContext,
    VerdictKind,
    VerifierPort,
)
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.commit import CommitOutcome, ShapeError
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import (
    _PRESETS,
    DefaultRetrievalPolicy,
    ObjectProvisioningPolicy,
    ObjectProvisionMode,
    ProvisionSelect,
    provisioning_policy,
)
from verity.control_plane.store import Decision, SqliteStore
from verity.domains.feature_engineering import (
    DATASET_VERSION,
    ENTRYPOINT,
    REQUIREMENTS,
    SUBMISSION,
    build_feature_engineering_domain,
    declared_objects,
)
from verity.sandbox import AgentSandbox, ProposalDescriptor
from verity.sandbox.descriptor import RESERVED_PROPOSAL_NAME

# --------------------------------------------------------------------------- domain wiring


def test_schema_and_gating_match_section_12() -> None:
    domain = build_feature_engineering_domain()
    types = {t.name: t for t in domain.schema.types()}
    assert types[DATASET_VERSION].is_root is True
    assert types[SUBMISSION].is_root is False
    # the broadened domain dropped the harvested Feature type (submission is the unit)
    assert FEATURE_REMOVED not in types
    ops = {o.name: o for o in domain.schema.operations()}
    assert ops["submit"].inputs == (DATASET_VERSION,) and ops["submit"].output == SUBMISSION
    assert ops["revises"].inputs == (SUBMISSION,) and ops["revises"].output == SUBMISSION
    assert "harvest" not in ops  # no feature harvest
    # Submission is gated with the incumbents + rejected-log slice; no Feature type is gated.
    assert domain.gated_types.is_gated(SUBMISSION)


FEATURE_REMOVED = "Feature"  # the type the broadened domain no longer registers


def _submission_payload() -> dict[str, object]:
    return {"entrypoint": ENTRYPOINT, "requirements": REQUIREMENTS}


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
    ],
)
def test_shape_rejects_malformed_submissions(mutate: object) -> None:
    domain = build_feature_engineering_domain()
    payload = _submission_payload()
    mutate(payload)  # type: ignore[operator]
    assert isinstance(domain.shape_validator(_artifact(payload)), ShapeError)


def test_declared_objects_names_the_submission_attachments() -> None:
    # the harvest keeps only what a Submission declares: its entrypoint + requirements files
    declared = declared_objects(_artifact(_submission_payload()))
    assert declared == frozenset({ENTRYPOINT, REQUIREMENTS})
    # a non-Submission (or a payload without the keys) declares no objects
    assert declared_objects(_artifact({}, type_=DATASET_VERSION)) == frozenset()


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
    policy = provisioning_policy(ObjectProvisionMode.NONE)
    assert policy.materialize(_store_with_three()) == {}


def test_policy_last_accepted_takes_the_most_recent_accepted() -> None:
    policy = provisioning_policy(
        ObjectProvisionMode.LAST_ACCEPTED, type_filter=SUBMISSION
    )
    out = policy.materialize(_store_with_three())["scratch"]
    # the rejected s3 is more recent but not accepted; s2 (latest accepted) wins, flat + a manifest.
    assert out["provided/submission.py"] == b"v2"
    assert "provided/INDEX.md" in out and b"s2" in out["provided/INDEX.md"]


def test_policy_all_accepted_ranks_by_recency_with_a_manifest() -> None:
    policy = provisioning_policy(
        ObjectProvisionMode.ALL_ACCEPTED, type_filter=SUBMISSION
    )
    out = policy.materialize(_store_with_three())["scratch"]
    # newest-first ranked subdirs (01 = s2, the most recent accepted), plus the manifest.
    assert out["provided/01-s2/submission.py"] == b"v2"
    assert out["provided/02-s1/submission.py"] == b"v1"
    manifest = out["provided/INDEX.md"].decode()
    assert "| 01 | s2 |" in manifest and "| 02 | s1 |" in manifest  # ordered, meaningful


def test_policy_all_includes_terminal_and_respects_cap() -> None:
    policy = provisioning_policy(
        ObjectProvisionMode.ALL, type_filter=SUBMISSION, max_objects=2
    )
    out = policy.materialize(_store_with_three())["scratch"]
    # the cap bounds object files (2), but the manifest always rides along (metadata, not payload).
    object_files = [k for k in out if not k.endswith("INDEX.md")]
    assert len(object_files) == 2 and "provided/INDEX.md" in out


def test_policy_last_revised_or_accepted_picks_the_in_flight_revised() -> None:
    # The agent's most-recent submission is `revised` (its refine target). Provision THAT so the
    # next cycle edits it, not just the older accepted incumbent or nothing at all (#51).
    store = SqliteStore()
    _seed_with_object(store, art_id="s1", status=ArtifactStatus.ACCEPTED,
                      name=ENTRYPOINT, data=b"accepted", ts="t1")
    _seed_with_object(store, art_id="s2", status=ArtifactStatus.REVISED,
                      name=ENTRYPOINT, data=b"refining", ts="t2")
    policy = provisioning_policy(
        ObjectProvisionMode.LAST_REVISED_OR_ACCEPTED, type_filter=SUBMISSION
    )
    out = policy.materialize(store)["scratch"]
    assert out["provided/submission.py"] == b"refining"  # the in-flight revised, not the accepted
    assert b"s2" in out["provided/INDEX.md"]


def test_policy_last_revised_or_accepted_falls_back_to_accepted() -> None:
    # With no in-flight revised it behaves like LAST_ACCEPTED, ignoring the more-recent rejected.
    policy = provisioning_policy(
        ObjectProvisionMode.LAST_REVISED_OR_ACCEPTED, type_filter=SUBMISSION
    )
    out = policy.materialize(_store_with_three())["scratch"]
    assert out["provided/submission.py"] == b"v2"  # s2 (latest accepted); rejected s3 ignored


def _record_score(store: SqliteStore, art_id: str, score: float) -> None:
    """Record a gate decision carrying ``score`` — what ``_artifact_score`` (and BEST) reads."""
    store.record_decision(
        Decision(
            artifact_id=art_id, gate="selection", verdict=VerdictKind.ACCEPT,
            rationale="seeded", score=score, created_at="t",
        )
    )


def test_policy_best_picks_higher_score_not_newest() -> None:
    # The #95 fix: among accepted/revised, BEST hands the agent its highest-SCORING prior, even when
    # an older artifact outscores a newer one — so the agent never iterates away from its peak.
    store = SqliteStore()
    _seed_with_object(store, art_id="s1", status=ArtifactStatus.ACCEPTED,
                      name=ENTRYPOINT, data=b"older-better", ts="t1")
    _seed_with_object(store, art_id="s2", status=ArtifactStatus.ACCEPTED,
                      name=ENTRYPOINT, data=b"newer-worse", ts="t2")
    _record_score(store, "s1", 0.90)
    _record_score(store, "s2", 0.50)
    policy = provisioning_policy(
        ObjectProvisionMode.BEST_REVISED_OR_ACCEPTED, type_filter=SUBMISSION
    )
    out = policy.materialize(store)["scratch"]
    assert out["provided/submission.py"] == b"older-better"  # score beats recency
    assert b"s1" in out["provided/INDEX.md"]


def test_policy_best_falls_back_to_recency_when_no_scores() -> None:
    # No decisions recorded → no scores → BEST degrades to LAST (newest wins, a defined fallback).
    store = SqliteStore()
    _seed_with_object(store, art_id="s1", status=ArtifactStatus.ACCEPTED,
                      name=ENTRYPOINT, data=b"old", ts="t1")
    _seed_with_object(store, art_id="s2", status=ArtifactStatus.ACCEPTED,
                      name=ENTRYPOINT, data=b"new", ts="t2")
    policy = provisioning_policy(
        ObjectProvisionMode.BEST_REVISED_OR_ACCEPTED, type_filter=SUBMISSION
    )
    out = policy.materialize(store)["scratch"]
    assert out["provided/submission.py"] == b"new"


def test_policy_best_accepted_ignores_revised() -> None:
    # The status-set filter precedes the score sort: BEST_ACCEPTED never picks a higher-scoring
    # revised over a lower-scoring accepted.
    store = SqliteStore()
    _seed_with_object(store, art_id="s1", status=ArtifactStatus.ACCEPTED,
                      name=ENTRYPOINT, data=b"accepted", ts="t1")
    _seed_with_object(store, art_id="s2", status=ArtifactStatus.REVISED,
                      name=ENTRYPOINT, data=b"revised", ts="t2")
    _record_score(store, "s1", 0.80)
    _record_score(store, "s2", 0.90)
    policy = provisioning_policy(
        ObjectProvisionMode.BEST_ACCEPTED, type_filter=SUBMISSION
    )
    out = policy.materialize(store)["scratch"]
    assert out["provided/submission.py"] == b"accepted"  # 0.80 accepted beats 0.90 revised


def test_policy_all_revised_or_accepted_ranks_accepted_and_revised() -> None:
    # Both non-rejected statuses, recency-ranked under subdirs; the rejected artifact is excluded.
    store = SqliteStore()
    _seed_with_object(store, art_id="s1", status=ArtifactStatus.ACCEPTED,
                      name=ENTRYPOINT, data=b"acc", ts="t1")
    _seed_with_object(store, art_id="s2", status=ArtifactStatus.REVISED,
                      name=ENTRYPOINT, data=b"rev", ts="t2")
    _seed_with_object(store, art_id="s3", status=ArtifactStatus.REJECTED,
                      name=ENTRYPOINT, data=b"bad", ts="t3")
    policy = provisioning_policy(
        ObjectProvisionMode.ALL_REVISED_OR_ACCEPTED, type_filter=SUBMISSION
    )
    out = policy.materialize(store)["scratch"]
    assert out["provided/01-s2/submission.py"] == b"rev"  # newest non-rejected first
    assert out["provided/02-s1/submission.py"] == b"acc"
    object_files = [k for k in out if not k.endswith("INDEX.md")]
    assert len(object_files) == 2  # rejected s3 excluded
    assert b"s3" not in out["provided/INDEX.md"]


def test_policy_all_accepted_single_hit_is_flat() -> None:
    # Count-based naming: a single selected artifact flattens (no clash possible), even in a
    # multi-select mode that would otherwise namespace under subdirs.
    store = SqliteStore()
    _seed_with_object(store, art_id="s1", status=ArtifactStatus.ACCEPTED,
                      name=ENTRYPOINT, data=b"only", ts="t1")
    policy = provisioning_policy(
        ObjectProvisionMode.ALL_ACCEPTED, type_filter=SUBMISSION
    )
    out = policy.materialize(store)["scratch"]
    assert out["provided/submission.py"] == b"only"
    assert not any(k.startswith("provided/01-") for k in out)  # not namespaced


def test_provision_name_clash_namespaced() -> None:
    # Two artifacts declaring the SAME object name must not clobber: count>1 ⇒ each under its own
    # <NN>-<id>/ subdir, both bytes survive, both listed in the index.
    store = _store_with_three()  # s1, s2 both declare ENTRYPOINT; s3 rejected
    policy = provisioning_policy(
        ObjectProvisionMode.ALL_ACCEPTED, type_filter=SUBMISSION
    )
    out = policy.materialize(store)["scratch"]
    assert out["provided/01-s2/submission.py"] == b"v2"
    assert out["provided/02-s1/submission.py"] == b"v1"
    object_files = [k for k in out if not k.endswith("INDEX.md")]
    assert len(object_files) == 2  # no overwrite despite identical object name
    idx = out["provided/INDEX.md"]
    assert b"s1" in idx and b"s2" in idx


def test_manifest_drops_instruction() -> None:
    # INDEX.md is a dumb metadata list: no hardwired agent instruction, just a title + table.
    out = provisioning_policy(
        ObjectProvisionMode.ALL_ACCEPTED, type_filter=SUBMISSION
    ).materialize(_store_with_three())["scratch"]
    idx = out["provided/INDEX.md"]
    assert b"Build on" not in idx
    assert b"newest first" not in idx
    assert b"starting from scratch" not in idx
    assert idx.startswith(b"# Provisioned durable objects")
    assert b"| rank | artifact | type | status | created_at | score | files |" in idx


def test_every_named_mode_has_a_preset() -> None:
    # The named modes are sugar; each must resolve to axes (none can silently mean "nothing").
    assert {m.value for m in ObjectProvisionMode} <= set(_PRESETS)


def test_provisioning_policy_resolves_presets_and_explicit_axes() -> None:
    # A preset name (str or enum), case-insensitive, resolves to the right axes.
    p = provisioning_policy("all_accepted", type_filter=SUBMISSION)
    assert p.statuses == frozenset({ArtifactStatus.ACCEPTED}) and p.select is ProvisionSelect.ALL
    assert provisioning_policy("BEST_ACCEPTED").select is ProvisionSelect.BEST
    assert provisioning_policy(ObjectProvisionMode.NONE).statuses == frozenset()  # nothing
    assert provisioning_policy("not-a-mode").statuses == frozenset()  # unknown → none + warn
    # The decoupling: the explicit form reaches superseded artifacts, no named mode required.
    lineage = provisioning_policy({"statuses": ["accepted", "superseded"], "select": "all"})
    assert lineage.statuses == frozenset({ArtifactStatus.ACCEPTED, ArtifactStatus.SUPERSEDED})
    assert provisioning_policy({"statuses": "any", "select": "last"}).statuses is None  # any


def test_fe_kaggle_default_provisioning_is_best() -> None:
    # The chosen default (#95): fe-kaggle builds on the best prior unless a knob overrides it.
    param = inspect.signature(configure_fe_kaggle_task).parameters["provisioning_spec"]
    assert param.default is ObjectProvisionMode.BEST_REVISED_OR_ACCEPTED


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
        object_namer=declared_objects,
        sandbox_key="fe",
        verifier_key="stub",
        object_provisioning=policy,
    )
    asyncio.run(cp.configure(config))
    return cp, store


def test_two_cycles_commit_and_provision_prior_script(tmp_path: Path) -> None:
    driver = _RecordingDriver()
    policy = provisioning_policy(
        ObjectProvisionMode.LAST_ACCEPTED, type_filter=SUBMISSION
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
