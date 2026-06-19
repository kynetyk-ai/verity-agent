"""Run identity stamped onto worker labels + a `RunRecord` at run end (ROADMAP 7.4.h / ADR 0003).

Drives the config-driven FE-Kaggle loop over `FakeBackend` (reusing the shared offline worker
doubles + a `FakeKaggleScorer`) and asserts the placed seams: the control plane binds a
`RunContext`, which reaches both the sandbox driver and the verifier's code-runner, so **every**
worker carries ``{harness,tenant,run,cycle,role,config}``; the ``run`` id is one stable value across
the run while ``cycle`` advances per sandbox cycle; and the run emits a `RunRecord` keyed by
``(tenant_id, run_id)`` that points back at the accepted artifacts.
"""

from __future__ import annotations

import asyncio

from tests._fe_offline import FeWorkers
from verity.composition.dataset import stratified_split, subsample
from verity.composition.fe_kaggle import FE_KAGGLE_TASK_ID, configure_fe_kaggle_task
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.store import SqliteStore
from verity.provisioning import FakeBackend
from verity.verifier.kaggle import FakeKaggleScorer

_RAW = b"id,a,class\n0,0,X\n1,2,X\n2,4,X\n3,6,Y\n4,8,Y\n5,10,Y\n"
_REAL_TEST = b"id\n100\n101\n102\n103\n"


def _run_kaggle_loop(*, max_cycles: int) -> tuple[FakeBackend, ControlPlane, str]:
    """Config-driven FE-Kaggle cycles over a FakeBackend; returns the backend, cp, and run id."""
    sub = subsample(_RAW, per_class=100, target="class")
    split = stratified_split(sub, target="class", id_column="id", reserved_fraction=0.5)
    reserved = dict(split.reserved_labels)
    steps = [("submit", ("ds",), marker, ["f0"]) for marker in (b"a", b"b")][:max_cycles]
    by_marker = {marker: reserved for (_, _, marker, _) in steps}  # perfect preds -> accepted
    backend = FakeBackend(script=FeWorkers(steps=steps, by_marker=by_marker))
    store = SqliteStore()
    cp = ControlPlane(store, policy=OrchestrationPolicy(max_cycles=max_cycles))
    asyncio.run(
        configure_fe_kaggle_task(
            cp, backend=backend, scorer=FakeKaggleScorer(scores=[0.80, 0.85]),
            split=split, full_train_csv=sub, real_test_csv=_REAL_TEST,
            poll_interval_s=0.0, wait_deadline_s=5.0,
        )
    )
    asyncio.run(cp.run(FE_KAGGLE_TASK_ID, goal="climb the leaderboard"))
    run_ids = {s.labels.run for s in backend.launched}
    assert len(run_ids) == 1  # one run id across every worker the run launched
    return backend, cp, run_ids.pop()


def test_every_worker_is_labelled_with_run_identity() -> None:
    backend, _cp, run_id = _run_kaggle_loop(max_cycles=2)
    sandboxes = [s for s in backend.launched if s.labels.role == "sandbox"]
    code_runners = [s for s in backend.launched if s.labels.role == "code-runner"]
    assert len(sandboxes) == 2  # a fresh sandbox worker per cycle
    assert code_runners  # the verifier's gates ran the submission in code-runner worker(s)

    # the run id is minted (non-empty) and tenant + harness + config are stamped on every worker
    assert run_id != ""
    for spec in backend.launched:
        assert spec.labels.tenant == "default"
        assert spec.labels.harness == "verity"
        assert spec.labels.config == "fe-kaggle"
        assert spec.labels.run == run_id

    # the cycle stamp advances per cycle for the sandbox (one sandbox launched per cycle, in order)
    assert [s.labels.cycle for s in sandboxes] == ["1", "2"]


def test_run_emits_a_run_record_pointing_into_the_store() -> None:
    _backend, cp, run_id = _run_kaggle_loop(max_cycles=2)
    record = cp.run_records().get("default", run_id)
    assert record is not None
    assert record.task_id == "fe-kaggle"  # which task (configuration) this run executed
    assert record.run_id == run_id  # the run's own identity, distinct from the task
    assert record.status == "complete"
    assert record.accepted_artifact_ids  # a pointer into the provenance store, not a copy


def test_a_second_run_of_the_same_task_gets_a_fresh_run_id() -> None:
    sub = subsample(_RAW, per_class=100, target="class")
    split = stratified_split(sub, target="class", id_column="id", reserved_fraction=0.5)
    reserved = dict(split.reserved_labels)
    store = SqliteStore()
    backend = FakeBackend(
        script=FeWorkers(steps=[("submit", ("ds",), b"a", ["f0"])], by_marker={b"a": reserved})
    )
    cp = ControlPlane(store, policy=OrchestrationPolicy(max_cycles=1))
    asyncio.run(
        configure_fe_kaggle_task(
            cp, backend=backend, scorer=FakeKaggleScorer(scores=[0.80, 0.85]),
            split=split, full_train_csv=sub, real_test_csv=_REAL_TEST,
            poll_interval_s=0.0, wait_deadline_s=5.0,
        )
    )

    asyncio.run(cp.run(FE_KAGGLE_TASK_ID, goal="g"))
    first = {s.labels.run for s in backend.launched}.pop()
    asyncio.run(cp.run(FE_KAGGLE_TASK_ID, goal="g"))
    run_ids = {s.labels.run for s in backend.launched}
    # task != run: the second run minted its own id, so the run records do not collide
    assert len(run_ids) == 2 and first in run_ids
    assert cp.run_records().get("default", first) is not None
