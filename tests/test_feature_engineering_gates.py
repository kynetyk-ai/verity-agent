"""The two §12 Submission gates + the deps-aware runner (Phase 4.2) — offline, no Docker, no model.

Drives the real ``SdkVerifier`` built by ``build_feature_engineering_verifier`` over a
``FakeCodeRunner`` whose output we script, so accept / reject are *earned* by balanced accuracy,
deflated by the trial count, against the incumbent — without training a real model. (The broadened
domain is reject-only: no refine, no per-feature penalty.) Plus the end-to-end §13.2 (rejected +
retained), §13.3 (superseded with lineage), and §13.4 (why-accepted from provenance) through the
control plane, and the runner command construction for the live deps/network path.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    Operation,
    OperationStatus,
    ProviderRegistry,
    SandboxPort,
    VerdictKind,
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
    REQUIREMENTS,
    SUBMISSION,
    balanced_accuracy,
    build_feature_engineering_domain,
    build_feature_engineering_verifier,
)
from verity.sandbox import AgentSandbox, ProposalDescriptor
from verity.sandbox.descriptor import RESERVED_PROPOSAL_NAME
from verity.verifier import (
    ContainerCodeRunner,
    FakeCodeRunner,
    RunRequest,
    RunResult,
    docker_available,
)


def docker_test(fn):
    """Mark ``docker`` and skip when no Docker daemon is reachable (so CI needs no Docker)."""
    fn = pytest.mark.skipif(not docker_available(), reason="no Docker daemon reachable")(fn)
    return pytest.mark.docker(fn)

# Reserved hold-out the gate holds (never exposed): three classes of two rows each.
RESERVED = {"1": "STAR", "2": "STAR", "3": "GALAXY", "4": "GALAXY", "5": "QSO", "6": "QSO"}
PERFECT = dict(RESERVED)  # balanced accuracy 1.0
TWO_THIRDS = {**RESERVED, "5": "STAR", "6": "STAR"}  # QSO recall 0 -> 0.667
ONE_THIRD = dict.fromkeys(RESERVED, "STAR")  # only STAR right -> 0.333


def _preds_csv(preds: dict[str, str]) -> bytes:
    rows = "\n".join(f"{rid},{label}" for rid, label in preds.items())
    return f"id,class\n{rows}\n".encode()


def _feature_names(n: int) -> list[str]:
    return [f"feat{i}" for i in range(n)]


def _entrypoint(marker: bytes, names: list[str]) -> bytes:
    """Script bytes: a marker the fake runner keys predictions on; ``names`` ride along as a comment
    for readability only (the broadened domain no longer gates on declared features)."""
    return marker + b"\n# defines: " + " ".join(names).encode()


def _runner(by_marker: dict[bytes, dict[str, str]]) -> FakeCodeRunner:
    """A fake runner whose predictions depend on which marker the submitted code contains."""

    def script(req: RunRequest) -> RunResult:
        for marker, preds in by_marker.items():
            if marker in req.code:
                return RunResult(exit_code=0, stdout="", stderr="", output=_preds_csv(preds))
        return RunResult(exit_code=0, stdout="", stderr="", output=None)

    return FakeCodeRunner(script=script)


def _request(
    code: bytes,
    *,
    n_features: int = 1,
    proposal_id: str = "p1",
    store_slice: tuple[Artifact, ...] = (),
    with_code: bool = True,
) -> VerifierRequest:
    names = _feature_names(n_features)
    proposal = Artifact(
        proposal_id, SUBMISSION,
        {"entrypoint": ENTRYPOINT, "requirements": REQUIREMENTS},
        ArtifactStatus.PROPOSED, "agent", "t1",
    )
    objects = (
        {ENTRYPOINT: _entrypoint(code, names), REQUIREMENTS: b"pandas==2.2.2"} if with_code else {}
    )
    return VerifierRequest(proposal=proposal, store_slice=store_slice, objects=objects)


def _verifier(runner: FakeCodeRunner, **kwargs: float):
    return build_feature_engineering_verifier(
        runner,
        agent_train_csv=b"id,class\n0,STAR\n",
        reserved_test_csv=b"id\n1\n2\n3\n4\n5\n6\n",
        reserved_labels=RESERVED,
        **kwargs,  # type: ignore[arg-type]
    )


def _accepted(art_id: str) -> Artifact:
    return Artifact(art_id, SUBMISSION, {}, ArtifactStatus.ACCEPTED, "agent", "t1")


def _rejected(art_id: str) -> Artifact:
    return Artifact(art_id, SUBMISSION, {}, ArtifactStatus.REJECTED, "agent", "t1")


# --------------------------------------------------------------------------- metric + parsing


def test_balanced_accuracy_weights_classes_equally() -> None:
    assert balanced_accuracy(RESERVED, PERFECT) == pytest.approx(1.0)
    assert balanced_accuracy(RESERVED, TWO_THIRDS) == pytest.approx(2 / 3)
    assert balanced_accuracy(RESERVED, ONE_THIRD) == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- runnable (cheap) gate


def _dispatch(verifier, request: VerifierRequest):
    return asyncio.run(verifier.dispatch(request))


def test_runnable_rejects_a_submission_with_no_code() -> None:
    verifier = _verifier(_runner({}))
    bundle = _dispatch(verifier, _request(b"x", with_code=False))
    assert bundle.status is ArtifactStatus.REJECTED
    assert bundle.decisions[-1].gate == "runs-clean"


def test_runnable_rejects_a_crash() -> None:
    runner = FakeCodeRunner(script=lambda _r: RunResult(1, "", "boom\nTraceback line"))
    bundle = _dispatch(_verifier(runner), _request(b"crashy"))
    assert bundle.status is ArtifactStatus.REJECTED
    assert "exited 1" in bundle.decisions[-1].rationale


def test_runnable_rejects_incomplete_predictions() -> None:
    runner = _runner({b"partial": {"1": "STAR", "2": "STAR"}})  # only 2 of 6 reserved rows
    bundle = _dispatch(_verifier(runner), _request(b"partial"))
    assert bundle.status is ArtifactStatus.REJECTED
    assert "missing" in bundle.decisions[-1].rationale


def test_both_gates_share_a_single_run() -> None:
    runner = _runner({b"good": PERFECT})
    verifier = _verifier(runner)
    _dispatch(verifier, _request(b"good"))
    # runnable + selection both evaluate the submission, but the script trains exactly once.
    assert len(runner.calls) == 1


# --------------------------------------------------------------------------- selection (hard) gate


def test_selection_accepts_an_improving_submission() -> None:
    verifier = _verifier(_runner({b"good": PERFECT}))
    bundle = _dispatch(verifier, _request(b"good"))
    assert bundle.status is ArtifactStatus.ACCEPTED
    selection = bundle.decisions[-1]
    assert selection.gate == "selection" and selection.kind is VerdictKind.ACCEPT
    assert selection.score == pytest.approx(1.0)  # raw balanced accuracy; no complexity penalty


def test_selection_rejects_when_it_cannot_beat_the_incumbent() -> None:
    # one verifier instance scores the incumbent (perfect) then judges a weaker challenger.
    verifier = _verifier(_runner({b"incumbent": PERFECT, b"weak": ONE_THIRD}))
    assert _dispatch(verifier, _request(b"incumbent", proposal_id="inc")).status is (
        ArtifactStatus.ACCEPTED
    )
    bundle = _dispatch(
        verifier, _request(b"weak", proposal_id="chal", store_slice=(_accepted("inc"),))
    )
    assert bundle.status is ArtifactStatus.REJECTED
    assert "does not improve" in bundle.decisions[-1].rationale


def test_trial_count_deflation_raises_the_bar() -> None:
    request = lambda slice_: _request(b"ok", store_slice=slice_)  # noqa: E731
    # TWO_THIRDS scores 0.667; with no prior trials and no incumbent it clears the 0 bar.
    assert _dispatch(_verifier(_runner({b"ok": TWO_THIRDS})), request(())).status is (
        ArtifactStatus.ACCEPTED
    )
    # two prior rejected submissions with deflation 0.4 raise the bar to 0.8 > 0.667 -> rejected.
    deflated = _verifier(_runner({b"ok": TWO_THIRDS}), trial_deflation=0.4)
    bundle = _dispatch(deflated, request((_rejected("r1"), _rejected("r2"))))
    assert bundle.status is ArtifactStatus.REJECTED  # bar 0.8 > score 0.667


# --------------------------------------------------------------------------- runner deps/network


def test_runner_command_installs_deps_over_network_when_requested(tmp_path: Path) -> None:
    runner = ContainerCodeRunner()
    req = RunRequest(
        code=b"print(1)", requirements=b"pandas==2.2.2", network=True,
        env={"VERITY_DATA": "/data", "VERITY_OUT": "/out"},
    )
    cmd = runner._docker_cmd("n", tmp_path / "c", tmp_path / "d", tmp_path / "o", req)
    joined = " ".join(cmd)
    assert "--network=none" not in cmd  # network enabled for the pip install
    assert "-e" in cmd and "VERITY_DATA=/data" in cmd and "VERITY_OUT=/out" in cmd
    assert any(a.startswith("--tmpfs=/tmp:") and "exec" in a for a in cmd)  # native wheels mmap .so
    assert cmd[-3:-1] == ["sh", "-c"]
    assert "pip install" in cmd[-1] and "/work/requirements.txt" in cmd[-1]
    assert "python /work/submission.py" in cmd[-1] and "PYTHONPATH=/tmp/site" in joined


def test_runner_command_keeps_the_no_network_default() -> None:
    runner = ContainerCodeRunner()
    cmd = runner._docker_cmd("n", Path("c"), Path("d"), Path("o"), RunRequest(code=b"x"))
    assert "--network=none" in cmd  # the default hostile-input posture is unchanged
    assert any(a.startswith("--tmpfs=/tmp:") and "exec" not in a for a in cmd)  # hardened tmpfs
    assert cmd[-2:] == ["python", "/work/submission.py"]


# --------------------------------------------------------------------- end-to-end through the CP


@dataclass
class _FeDriver:
    """Scripted ``SandboxDriver``: emits the n-th (code, feature-count) submission per cycle."""

    scripts: list[tuple[bytes, int]]
    cursor: int = 0

    async def run(
        self, *, system_prompt: str, user_message: str, operations: object, workspace: object,
    ) -> None:
        code, n_features = self.scripts[self.cursor]
        self.cursor += 1
        names = _feature_names(n_features)
        out = workspace.outbox()  # type: ignore[attr-defined]
        (out / ENTRYPOINT).write_bytes(_entrypoint(code, names))
        (out / REQUIREMENTS).write_bytes(b"pandas==2.2.2")
        payload = {"entrypoint": ENTRYPOINT, "requirements": REQUIREMENTS}
        (out / RESERVED_PROPOSAL_NAME).write_bytes(
            ProposalDescriptor("submit", ("ds",), payload, metadata="reasoned").to_json()
        )


def _e2e(tmp_path: Path) -> tuple[ControlPlane, SqliteStore]:
    store = SqliteStore()
    store.propose(
        Artifact("ds", DATASET_VERSION, {"rows": 6}, ArtifactStatus.PROPOSED, "loader", "t0",
                 is_root=True),
        Operation("op-ds", "load", (), "ds", OperationStatus.SUCCESS, "t0"),
    )
    ids = iter(f"sub-{i}" for i in range(1, 99))
    domain = build_feature_engineering_domain()
    verifier = _verifier(_runner({b"A": TWO_THIRDS, b"B": PERFECT, b"C": ONE_THIRD}))
    driver = _FeDriver(scripts=[(b"A", 1), (b"B", 1), (b"C", 1)])
    sandbox = AgentSandbox(
        root=tmp_path / "ws", driver=driver, schema=domain.schema,  # type: ignore[arg-type]
        proposer_identity="fe:test", clock=lambda: "t1", id_source=lambda: next(ids),
    )
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("fe", lambda: sandbox)
    vp.register("fe", lambda: verifier)
    cp = ControlPlane(
        store, policy=OrchestrationPolicy(max_cycles=5),
        sandbox_providers=sp, verifier_providers=vp,
    )
    config = TaskConfig(
        task_id="fe", instructions="improve the model",
        domain_instructions=domain.domain_instructions,
        schema=domain.schema, gated_types=domain.gated_types, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator, sandbox_key="fe", verifier_key="fe",
    )
    asyncio.run(cp.configure(config))
    return cp, store


def test_supersession_rejection_and_provenance_end_to_end(tmp_path: Path) -> None:
    cp, store = _e2e(tmp_path)

    r1 = asyncio.run(cp.run_cycle("fe", goal="cycle 1"))  # 0.667 -> accepted (no incumbent)
    r2 = asyncio.run(cp.run_cycle("fe", goal="cycle 2"))  # 1.000 -> accepted, supersedes sub-1
    r3 = asyncio.run(cp.run_cycle("fe", goal="cycle 3"))  # 0.333 -> rejected (cannot beat sub-2)

    assert r1.commit is not None and r1.commit.outcome is CommitOutcome.ACCEPTED
    assert r2.commit is not None and r2.commit.outcome is CommitOutcome.ACCEPTED
    assert r3.commit is not None and r3.commit.outcome is CommitOutcome.REJECTED

    # §13.3 — superseded with intact lineage
    sub1, sub2 = store.get_artifact("sub-1"), store.get_artifact("sub-2")
    assert sub1 is not None and sub1.status is ArtifactStatus.SUPERSEDED
    assert sub1.superseded_by == "sub-2"
    assert sub2 is not None and sub2.status is ArtifactStatus.ACCEPTED

    # §13.2 — the rejected submission is retained and queryable with a rationale
    rejected = store.rejected_log(type=SUBMISSION)
    assert [a.id for a in rejected] == ["sub-3"]
    assert any(d.verdict is VerdictKind.REJECT for d in store.decisions_for("sub-3"))

    # §13.4 — "why accepted" answerable from provenance: the submit op + a scored accept decision
    prov = cp.provenance("sub-2")
    assert "ds" in {a.id for a in prov.ancestors}
    accept = [
        d for d in prov.decisions if d.verdict is VerdictKind.ACCEPT and d.gate == "selection"
    ]
    assert accept and accept[0].score == pytest.approx(1.0)

    # §13.1 — append-only: the superseded incumbent's payload was never overwritten
    assert sub1.payload["entrypoint"] == ENTRYPOINT


# ----------------------------------------------------------- real container run (deps + network)

# A genuine submission: installs and uses pandas, reads the gate's CSVs from $VERITY_DATA, predicts
# the class by a redshift rule, and writes $VERITY_OUT/predictions.csv (the script contract).
_REAL_SCRIPT = b"""
import os
import pandas as pd

data = os.environ.get("VERITY_DATA", "data")
out = os.environ.get("VERITY_OUT", "out")
os.makedirs(out, exist_ok=True)

pd.read_csv(os.path.join(data, "train.csv"))  # the gate trains on the agent's data
test = pd.read_csv(os.path.join(data, "test.csv"))


def predict(z: float) -> str:
    if z < 0.1:
        return "STAR"
    if z < 1.0:
        return "GALAXY"
    return "QSO"


test["redshift_rule"] = test["redshift"]  # the declared engineered feature
test["class"] = test["redshift_rule"].apply(predict)
test[["id", "class"]].to_csv(os.path.join(out, "predictions.csv"), index=False)
"""

_REAL_CSV = b"".join(
    [b"id,redshift,class\n"]
    + [f"{i},{z},{c}\n".encode() for i, (z, c) in enumerate(
        [(0.01, "STAR"), (0.02, "STAR"), (0.03, "STAR"), (0.04, "STAR"),
         (0.3, "GALAXY"), (0.4, "GALAXY"), (0.5, "GALAXY"), (0.6, "GALAXY"),
         (1.5, "QSO"), (1.8, "QSO"), (2.2, "QSO"), (2.6, "QSO")], start=1)]
)


@docker_test
def test_real_script_installs_pandas_runs_and_is_accepted() -> None:
    from tools.harness.dataset import stratified_split

    split = stratified_split(_REAL_CSV, target="class", id_column="id", reserved_fraction=0.5)
    verifier = build_feature_engineering_verifier(
        # A larger tmpfs (RAM-backed, so a matching memory limit) holds the pip-installed deps.
        ContainerCodeRunner(memory="2g", tmpfs_size="1g"),
        agent_train_csv=split.agent_train_csv,
        reserved_test_csv=split.reserved_test_csv,
        reserved_labels=split.reserved_labels,
        timeout_s=600.0,
    )
    proposal = Artifact(
        "real", SUBMISSION,
        {"entrypoint": ENTRYPOINT, "requirements": REQUIREMENTS},
        ArtifactStatus.PROPOSED, "agent", "t1",
    )
    request = VerifierRequest(
        proposal=proposal, store_slice=(),
        objects={ENTRYPOINT: _REAL_SCRIPT, REQUIREMENTS: b"pandas==2.2.2"},
    )
    bundle = asyncio.run(verifier.dispatch(request))
    # the rule separates the classes perfectly -> balanced accuracy 1.0 -> accepted
    assert bundle.status is ArtifactStatus.ACCEPTED
    assert bundle.decisions[-1].score == pytest.approx(1.0)
