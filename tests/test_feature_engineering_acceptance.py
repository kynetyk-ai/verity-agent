"""The §13 acceptance criteria, on the feature-engineering domain (ROADMAP Phase 4.4 → MVP).

Each criterion is a runnable check driving the **real** control plane, the real ``SdkVerifier`` (the
two ``Submission`` gates), and the real ``AgentSandbox`` with a scripted driver over a deterministic
``FakeCodeRunner`` — so the whole read → propose → gate → commit loop runs offline. Genuine
in-container execution and a real multi-round model run are the ``@pytest.mark.docker`` /
``@pytest.mark.live`` demonstrations (``test_feature_engineering_live``).

The domain was broadened beyond feature engineering (submission is the unit; reject-only), so the
two §13 criteria that exercised the harvested-``Feature`` machinery and the ``refine`` path
(§13.8) are no longer demonstrated here — see ``verity.domains.feature_engineering`` for the
divergence note.
"""

from __future__ import annotations

import asyncio
import csv
import io
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    Operation,
    OperationStatus,
    ProposalEnvelope,
    ProviderRegistry,
    SandboxPort,
    VerdictKind,
    VerifierPort,
)
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.commit import CommitOutcome, NoImplicitAccept
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import (
    DefaultRetrievalPolicy,
    ObjectProvisionMode,
    provisioning_policy,
)
from verity.control_plane.store import SqliteStore
from verity.domains.feature_engineering import (
    DATASET_VERSION,
    ENTRYPOINT,
    REQUIREMENTS,
    SUBMISSION,
    build_feature_engineering_domain,
    build_feature_engineering_verifier,
    declared_objects,
)
from verity.sandbox import AgentSandbox, ProposalDescriptor
from verity.sandbox.descriptor import RESERVED_PROPOSAL_NAME
from verity.verifier import (
    ContainerCodeRunner,
    FakeCodeRunner,
    RunResult,
    docker_available,
)

RESERVED = {"1": "STAR", "2": "STAR", "3": "GALAXY", "4": "GALAXY", "5": "QSO", "6": "QSO"}


def _preds_csv(preds: dict[str, str]) -> bytes:
    return ("id,class\n" + "\n".join(f"{k},{v}" for k, v in preds.items()) + "\n").encode()


def _entrypoint(marker: bytes, names: list[str]) -> bytes:
    return marker + b"\n# defines: " + " ".join(names).encode()


def _marker_runner(by_marker: dict[bytes, dict[str, str]]) -> FakeCodeRunner:
    """Predictions depend on which marker the submitted code contains (the gate scores them)."""

    def script(req: RunResult) -> RunResult:  # type: ignore[valid-type]
        for marker, preds in by_marker.items():
            if marker in req.code:  # type: ignore[attr-defined]
                return RunResult(0, "", "", output=_preds_csv(preds))
        return RunResult(0, "", "", output=None)

    return FakeCodeRunner(script=script)  # type: ignore[arg-type]


# A submission: marker -> predictions; features named so the static features-defined gate passes.
_GOOD = (b"good", dict(RESERVED))  # balanced accuracy 1.0
_OK = (b"ok", {**RESERVED, "5": "STAR", "6": "STAR"})  # 0.667
_LEAKY = (b"leaky", dict.fromkeys(RESERVED, "STAR"))  # great on the agent's eyes, 0.333 on reserved


@dataclass
class _Driver:
    """Emits each scripted ``(op_name, parents, marker, feature-names)`` proposal in turn."""

    steps: list[tuple[str, tuple[str, ...], bytes, list[str]]]
    cursor: int = 0

    async def run(
        self, *, system_prompt: str, user_message: str, operations: object, workspace: object,
    ) -> None:
        op_name, parents, marker, names = self.steps[self.cursor]
        self.cursor += 1
        out = workspace.outbox()  # type: ignore[attr-defined]
        (out / ENTRYPOINT).write_bytes(_entrypoint(marker, names))
        (out / REQUIREMENTS).write_bytes(b"pandas==2.2.2")
        payload = {"entrypoint": ENTRYPOINT, "requirements": REQUIREMENTS}
        (out / RESERVED_PROPOSAL_NAME).write_bytes(
            ProposalDescriptor(op_name, parents, payload, metadata="i reasoned thus").to_json()
        )


def _build(
    tmp_path: Path,
    *,
    steps: list[tuple[str, tuple[str, ...], bytes, list[str]]],
    runner: FakeCodeRunner,
    policy: OrchestrationPolicy | None = None,
) -> tuple[ControlPlane, SqliteStore, AgentSandbox]:
    store = SqliteStore()
    store.propose(
        Artifact("ds", DATASET_VERSION, {"rows": 6}, ArtifactStatus.PROPOSED, "loader", "t0",
                 is_root=True),
        Operation("op-ds", "load", (), "ds", OperationStatus.SUCCESS, "t0"),
    )
    ids = iter(f"sub-{i}" for i in range(1, 999))
    domain = build_feature_engineering_domain()
    sandbox = AgentSandbox(
        root=tmp_path / "ws", driver=_Driver(steps),  # type: ignore[arg-type]
        schema=domain.schema, proposer_identity="fe:agent",
        clock=lambda: "t1", id_source=lambda: next(ids),
    )
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("fe", lambda: sandbox)
    vp.register("fe", lambda: build_feature_engineering_verifier(
        runner, agent_train_csv=b"id,class\n0,STAR\n",
        reserved_test_csv=b"id\n1\n2\n3\n4\n5\n6\n", reserved_labels=RESERVED,
    ))
    cp = ControlPlane(
        store, policy=policy or OrchestrationPolicy(max_cycles=10),
        sandbox_providers=sp, verifier_providers=vp,
    )
    config = TaskConfig(
        task_id="fe", instructions="design features that improve the model",
        domain_instructions=domain.domain_instructions, schema=domain.schema,
        gated_types=domain.gated_types, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator, object_namer=declared_objects,
        sandbox_key="fe", verifier_key="fe",
        object_provisioning=provisioning_policy(
            ObjectProvisionMode.LAST_ACCEPTED, type_filter=SUBMISSION
        ),
    )
    asyncio.run(cp.configure(config))
    return cp, store, sandbox


def _submission_envelope(
    sub_id: str, *, marker: bytes, names: list[str], parents: tuple[str, ...] = ("ds",),
    op_name: str = "submit",
) -> ProposalEnvelope:
    artifact = Artifact(
        sub_id, SUBMISSION,
        {"entrypoint": ENTRYPOINT, "requirements": REQUIREMENTS},
        ArtifactStatus.PROPOSED, "fe:agent", "t1",
    )
    op = Operation(f"op-{sub_id}", op_name, parents, sub_id, OperationStatus.SUCCESS, "t1")
    return ProposalEnvelope(
        artifact=artifact, operation=op, metadata="why",
        objects={ENTRYPOINT: _entrypoint(marker, names), REQUIREMENTS: b"pandas==2.2.2"},
    )


# =========================================================================== the twelve criteria


def test_01_append_only_history(tmp_path: Path) -> None:
    cp, store, _ = _build(
        tmp_path, runner=_marker_runner(dict([_OK, _GOOD])),
        steps=[("submit", ("ds",), b"ok", ["feat0"]), ("submit", ("ds",), b"good", ["feat0"])],
    )
    asyncio.run(cp.run_cycle("fe", goal="c1"))
    before = store.get_artifact("sub-1")
    assert before is not None and before.status is ArtifactStatus.ACCEPTED
    asyncio.run(cp.run_cycle("fe", goal="c2"))  # sub-2 supersedes sub-1
    after = store.get_artifact("sub-1")
    # the incumbent's payload was never overwritten; the change is a new artifact + supersession
    assert after is not None and after.payload == before.payload
    assert after.status is ArtifactStatus.SUPERSEDED and after.superseded_by == "sub-2"


def test_02_submission_rejected_and_retained(tmp_path: Path) -> None:
    # an incumbent at 1.0, then a leaky challenger that cannot beat it -> rejected and kept
    cp, store, _ = _build(
        tmp_path, runner=_marker_runner(dict([_GOOD, _LEAKY])),
        steps=[("submit", ("ds",), b"good", ["feat0"]), ("submit", ("ds",), b"leaky", ["feat0"])],
    )
    asyncio.run(cp.run_cycle("fe", goal="c1"))
    asyncio.run(cp.run_cycle("fe", goal="c2"))
    rejected = store.rejected_log(type=SUBMISSION)
    assert [a.id for a in rejected] == ["sub-2"]  # retained and queryable
    assert any(d.verdict is VerdictKind.REJECT for d in store.decisions_for("sub-2"))


def test_03_superseded_with_intact_lineage(tmp_path: Path) -> None:
    cp, store, _ = _build(
        tmp_path, runner=_marker_runner(dict([_OK, _GOOD])),
        steps=[("submit", ("ds",), b"ok", ["feat0"]), ("submit", ("ds",), b"good", ["feat0"])],
    )
    asyncio.run(cp.run_cycle("fe", goal="c1"))
    asyncio.run(cp.run_cycle("fe", goal="c2"))
    sub1 = store.get_artifact("sub-1")
    assert sub1 is not None and sub1.status is ArtifactStatus.SUPERSEDED
    assert sub1.superseded_by == "sub-2"
    sub2 = store.get_artifact("sub-2")
    assert sub2 is not None and sub2.status is ArtifactStatus.ACCEPTED
    # lineage intact: the accepted submission still traces to the dataset root (§4.2, §13.3)
    prov = cp.provenance("sub-2")
    assert "ds" in {a.id for a in prov.ancestors}


def test_04_why_accepted_is_answerable_from_provenance(tmp_path: Path) -> None:
    cp, _store, _ = _build(
        tmp_path, runner=_marker_runner(dict([_GOOD])),
        steps=[("submit", ("ds",), b"good", ["feat0"])],
    )
    asyncio.run(cp.run_cycle("fe", goal="c1"))
    prov = cp.provenance("sub-1")
    assert "ds" in {a.id for a in prov.ancestors}
    accept = [
        d for d in prov.decisions if d.verdict is VerdictKind.ACCEPT and d.gate == "selection"
    ]
    assert accept and accept[0].score is not None  # the scored accept decision
    # the submit op that produced it is recorded out of the dataset root
    assert any(op.op_name == "submit" for op in _store_ops_out(cp, "ds"))
    # the agent's rationale is retained off the gate channel
    assert cp.rationale_for("sub-1") == "i reasoned thus"


def _store_ops_out(cp: ControlPlane, artifact_id: str) -> list[Operation]:
    return cp._store.operations_out_of(artifact_id)  # noqa: SLF001 — a test reaching into the store


def test_05_bounded_context_across_a_long_run(tmp_path: Path) -> None:
    cp, store, _ = _build(tmp_path, runner=_marker_runner({}), steps=[])
    counter = 0

    def size_after(n: int) -> int:
        nonlocal counter
        for _ in range(n):
            aid = f"x{counter:06d}"
            counter += 1
            store.propose(
                Artifact(aid, SUBMISSION, {"v": 1}, ArtifactStatus.PROPOSED, "loader",
                         f"t{counter:06d}"),
                Operation(f"op-{aid}", "submit", ("ds",), aid, OperationStatus.SUCCESS, "t0"),
            )
        served = asyncio.run(cp.serve_context("fe", goal="g"))
        return len(served.system_prompt) + len(served.tail)

    small = size_after(100)
    big = size_after(2000)
    assert abs(big - small) < 80  # 20x the artifacts, but the served context does not grow (§13.5)


def test_06_refusal_to_commit_a_gateless_type(tmp_path: Path) -> None:
    cp, store, _ = _build(tmp_path, runner=_marker_runner({}), steps=[])
    # DatasetVersion is a registered type with no gate binding -> committing one is an error
    envelope = ProposalEnvelope(
        artifact=Artifact(
            "ds2", DATASET_VERSION, {"x": 1}, ArtifactStatus.PROPOSED, "fe:agent", "t1"
        ),
        operation=Operation("op-ds2", "load", ("ds",), "ds2", OperationStatus.SUCCESS, "t1"),
    )
    with pytest.raises(NoImplicitAccept):
        asyncio.run(cp.submit_proposal("fe", envelope))


def test_07_malformed_proposal_is_correctable_not_recorded(tmp_path: Path) -> None:
    cp, store, _ = _build(tmp_path, runner=_marker_runner(dict([_GOOD])), steps=[])
    # a Submission missing its required 'entrypoint' key is malformed (a shape error, not a reject)
    bad = ProposalEnvelope(
        artifact=Artifact(
            "sub-1", SUBMISSION, {"requirements": REQUIREMENTS},
            ArtifactStatus.PROPOSED, "fe:agent", "t1",
        ),
        operation=Operation("op-sub-1", "submit", ("ds",), "sub-1", OperationStatus.SUCCESS, "t1"),
        objects={ENTRYPOINT: _entrypoint(b"good", ["feat0"]), REQUIREMENTS: b"pandas==2.2.2"},
    )
    result = asyncio.run(cp.submit_proposal("fe", bad))
    assert result.entered_protocol is False and result.shape_error is not None
    assert result.harvested == {} and store.rejected_log(type=SUBMISSION) == []
    assert store.decisions_for("sub-1") == []  # nothing recorded, no trial-count increment
    # a corrected resubmission proceeds normally
    good = _submission_envelope("sub-1", marker=b"good", names=["feat0"])
    ok = asyncio.run(cp.submit_proposal("fe", good))
    assert ok.commit is not None and ok.commit.outcome is CommitOutcome.ACCEPTED


def test_09_privileged_mutator_boundary_holds(tmp_path: Path) -> None:
    _cp, _store, sandbox = _build(tmp_path, runner=_marker_runner({}), steps=[])
    for name in ("set_status", "record_decision", "propose", "put_object", "dispatch"):
        assert not hasattr(sandbox, name)
    assert not hasattr(sandbox, "store") and not hasattr(sandbox, "verifier")


def test_10_object_round_trips_and_is_executed_not_trusted(tmp_path: Path) -> None:
    runner = _marker_runner(dict([_GOOD]))
    cp, store, sandbox = _build(
        tmp_path, runner=runner, steps=[("submit", ("ds",), b"good", ["feat0"])]
    )
    asyncio.run(cp.run_cycle("fe", goal="c1"))
    sub1 = store.get_artifact("sub-1")
    assert sub1 is not None
    # content-addressed and retrievable by reference (§4.1, §13.10)
    objects = dict(sub1.objects)
    assert ENTRYPOINT in objects
    assert store.get_object(objects[ENTRYPOINT].content_hash) == _entrypoint(b"good", ["feat0"])
    # the gate executed the object (the runner was handed the code), not trusted a claim
    assert runner.calls and runner.calls[0].entrypoint == ENTRYPOINT
    # harvest happened before teardown: the outbox is empty after the cycle's regeneration
    assert list((tmp_path / "ws" / "outbox").iterdir()) == []


def test_11_leakage_is_caught_on_the_reserved_set(tmp_path: Path) -> None:
    # An honest submission sets a high incumbent; the leaky one would look perfect to the agent, but
    # the judged score is the verifier's on the reserved labels (low), so it does not carry.
    cp, store, _ = _build(
        tmp_path, runner=_marker_runner(dict([_GOOD, _LEAKY])),
        steps=[
            ("submit", ("ds",), b"good", ["feat0"]),
            ("submit", ("ds",), b"leaky", ["leak_feat"]),
        ],
    )
    asyncio.run(cp.run_cycle("fe", goal="c1"))  # honest -> reserved 1.0, accepted
    result = asyncio.run(cp.run_cycle("fe", goal="c2"))  # leaky -> reserved 0.333, cannot carry
    assert result.commit is not None and result.commit.outcome is CommitOutcome.REJECTED
    selection = [d for d in store.decisions_for("sub-2") if d.gate == "selection"]
    assert selection and selection[0].score is not None and selection[0].score < 0.5


def test_12_sandbox_is_ephemeral_between_cycles(tmp_path: Path) -> None:
    gold = b"id,class\n0,STAR\n1,GALAXY\n"
    sandbox = AgentSandbox(
        root=tmp_path / "ws", driver=_Driver([]),  # type: ignore[arg-type]
        schema=build_feature_engineering_domain().schema, proposer_identity="fe:agent",
        static_contents={"data": {"train.csv": gold}},
    )

    async def go() -> bytes:
        await sandbox.provision()
        # the agent mutates the gold data source inside its workspace
        (tmp_path / "ws" / "data" / "train.csv").write_bytes(b"CORRUPTED")
        await sandbox.regenerate()  # one ephemerality event between cycles
        return (tmp_path / "ws" / "data" / "train.csv").read_bytes()

    assert asyncio.run(go()) == gold  # the next cycle shows the unmutated gold copy (§13.12)


# ------------------------------------------------- the live, multi-round demonstration (manual)


def _subsample(data: bytes, *, per_class: int, target: str = "class") -> bytes:
    """Keep up to ``per_class`` rows per target class — a fast, stratified slice for a live run."""
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    header = list(reader.fieldnames or [])
    seen: dict[str, int] = {}
    kept: list[dict[str, str]] = []
    for row in reader:
        label = row[target]
        if seen.get(label, 0) < per_class:
            seen[label] = seen.get(label, 0) + 1
            kept.append(row)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header)
    writer.writeheader()
    writer.writerows(kept)
    return buf.getvalue().encode("utf-8")


@pytest.mark.live
def test_feature_engineering_live(tmp_path: Path) -> None:
    """A real multi-round run: Claude proposes scripts, the verifier trains + scores them in a
    container, and the agent improves on the provisioned incumbent. Manual / billed — auto-skips
    without a key, Docker, the dataset, or the ``sandbox`` extra. The whole system, in play."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("no ANTHROPIC_API_KEY")
    if not docker_available():
        pytest.skip("no Docker for the verifier's code runner")
    dataset = Path("results/prototyping_datasci_test/train.csv")
    if not dataset.exists():
        pytest.skip("the stellar dataset is not present")
    from tools.harness.dataset import stratified_split

    from verity.provisioning import DockerBackend
    from verity.sandbox.backend_driver import BackendSandboxDriver
    from verity.sandbox.registration import build_sandbox

    raw = _subsample(dataset.read_bytes(), per_class=400)  # small slice → fast per-cycle training
    split = stratified_split(raw, target="class", id_column="id", reserved_fraction=0.5)

    domain = build_feature_engineering_domain()
    store = SqliteStore()
    store.propose(
        Artifact("ds", DATASET_VERSION, {"source": "stellar"}, ArtifactStatus.PROPOSED,
                 "loader", "t0", is_root=True),
        Operation("op-ds", "load", (), "ds", OperationStatus.SUCCESS, "t0"),
    )
    # The CONTAINER sandbox: the agent gets a real shell + network, so it can install libraries and
    # *run and test* its script before proposing (the in-process backend cannot execute code, so a
    # code-writing agent loops to the recursion cap). The data rides the workspace mount via
    # static_contents (the gold-data role is re-mounted read-only on top — physical isolation).
    driver = BackendSandboxDriver(
        backend=DockerBackend(), model="anthropic:claude-sonnet-4-6",
        image="verity-sandbox:latest", recursion_limit=200, timeout_s=1500.0, memory="4g",
    )
    sandbox = build_sandbox(
        schema=domain.schema, root=tmp_path / "ws", driver=driver,
        proposer_identity="deepagents-worker:claude",
        static_contents={"data": {"train.csv": split.agent_train_csv,
                                  "test.csv": split.reserved_test_csv}},
    )
    runner = ContainerCodeRunner(memory="2g", tmpfs_size="1g")
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("fe", lambda: sandbox)
    vp.register("fe", lambda: build_feature_engineering_verifier(
        runner, agent_train_csv=split.agent_train_csv, reserved_test_csv=split.reserved_test_csv,
        reserved_labels=split.reserved_labels, timeout_s=600.0,
    ))
    cp = ControlPlane(
        store, policy=OrchestrationPolicy(max_cycles=4),
        sandbox_providers=sp, verifier_providers=vp,
    )
    config = TaskConfig(
        task_id="fe",
        instructions=(
            "Improve balanced accuracy by any means that fits in one script — engineered features, "
            "model choice, a small ensemble, calibration, imbalance handling. If there's no "
            "incumbent yet, explore the data with code first; if there is one (under "
            "scratch/provided/), read it and target its weakness. Keep the pipeline fast enough to "
            "finish the gate's time budget; run the script once to confirm it works, then submit."
        ),
        domain_instructions=domain.domain_instructions, schema=domain.schema,
        gated_types=domain.gated_types, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator, object_namer=declared_objects,
        sandbox_key="fe", verifier_key="fe",
        object_provisioning=provisioning_policy(
            ObjectProvisionMode.LAST_ACCEPTED, type_filter=SUBMISSION
        ),
    )
    asyncio.run(cp.configure(config))

    # A slow agent that exceeds the sandbox timeout is now a skipped cycle, not a crashed run (#21),
    # so run() returns normally; assert on what committed across the cycles.
    asyncio.run(cp.run(
        "fe",
        goal="Improve balanced accuracy on a held-out set; keep the pipeline fast and simple.",
    ))
    accepted = store.query_artifacts(type=SUBMISSION, status=ArtifactStatus.ACCEPTED)
    assert accepted, "no accepted submission across the live run"
