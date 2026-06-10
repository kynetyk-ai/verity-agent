"""Run identity stamped onto worker labels + a `RunRecord` at run end (ROADMAP 7.4.h / ADR 0003).

Drives the config-driven FE loop over `FakeBackend` (reusing the 7.4.g worker doubles) and asserts
the placed seams: the control plane binds a `RunContext`, which reaches both the sandbox driver and
the verifier's code-runner, so **every** worker carries ``{harness,tenant,run,cycle,role,config}``;
the ``run`` id is one stable value across the run while ``cycle`` advances per cycle; and the run
emits a `RunRecord` keyed by ``(tenant_id, run_id)`` that points back at the accepted artifacts.
"""

from __future__ import annotations

import asyncio

from tests.test_fe_via_config_acceptance import _GOOD, _OK, _FeWorkers, _Split
from verity.composition import build_fe_control_plane
from verity.control_plane.api import ControlPlane
from verity.control_plane.store import SqliteStore
from verity.provisioning import FakeBackend


def _run_fe_loop() -> tuple[FakeBackend, ControlPlane, str]:
    """Two config-driven FE cycles over a FakeBackend; returns the backend, the cp, and the run."""
    split = _Split()
    workers = _FeWorkers(
        steps=[
            ("submit", ("ds",), _OK[0], ["feat0"]),
            ("submit", ("ds",), _GOOD[0], ["feat1"]),
        ],
        by_marker=dict([_OK, _GOOD]),
    )
    backend = FakeBackend(script=workers)
    store = SqliteStore()
    cp, config = build_fe_control_plane(backend=backend, split=split, max_cycles=2, store=store)
    asyncio.run(cp.configure(config))
    asyncio.run(cp.run("fe", goal="improve balanced accuracy"))
    run_ids = {s.labels.run for s in backend.launched}
    assert len(run_ids) == 1  # one run id across every worker the run launched
    return backend, cp, run_ids.pop()


def test_every_worker_is_labelled_with_run_identity() -> None:
    backend, _cp, run_id = _run_fe_loop()
    sandboxes = [s for s in backend.launched if s.labels.role == "sandbox"]
    code_runners = [s for s in backend.launched if s.labels.role == "code-runner"]
    assert len(sandboxes) == 2 and len(code_runners) == 2  # a fresh worker of each role per cycle

    # the run id is minted (non-empty) and the tenant + harness + config are stamped on every worker
    assert run_id != ""
    for spec in backend.launched:
        assert spec.labels.tenant == "default"
        assert spec.labels.harness == "verity"
        assert spec.labels.config == "fe"
        assert spec.labels.run == run_id

    # the cycle stamp advances per cycle, for both roles (workers are launched in cycle order)
    assert [s.labels.cycle for s in sandboxes] == ["1", "2"]
    assert [s.labels.cycle for s in code_runners] == ["1", "2"]


def test_run_emits_a_run_record_pointing_into_the_store() -> None:
    _backend, cp, run_id = _run_fe_loop()
    record = cp.run_records().get("default", run_id)
    assert record is not None
    assert record.task_id == "fe"  # which task (configuration) this run executed
    assert record.run_id == run_id  # the run's own identity, distinct from the task
    assert record.status == "complete"
    assert record.accepted_artifact_ids  # a pointer into the provenance store, not a copy


def test_a_second_run_of_the_same_task_gets_a_fresh_run_id() -> None:
    split = _Split()
    store = SqliteStore()
    backend = FakeBackend(
        script=_FeWorkers(steps=[("submit", ("ds",), _GOOD[0], ["f"])], by_marker=dict([_GOOD]))
    )
    cp, config = build_fe_control_plane(backend=backend, split=split, max_cycles=1, store=store)
    asyncio.run(cp.configure(config))

    asyncio.run(cp.run("fe", goal="g"))
    first = {s.labels.run for s in backend.launched}.pop()
    asyncio.run(cp.run("fe", goal="g"))
    run_ids = {s.labels.run for s in backend.launched}
    # task != run: the second run minted its own id, so the run records do not collide
    assert len(run_ids) == 2 and first in run_ids
    assert cp.run_records().get("default", first) is not None
