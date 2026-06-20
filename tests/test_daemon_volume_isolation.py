"""Daemon-layer volume isolation (ROADMAP 8.3, ADR 0004 §f) — neither host volume reaches a worker.

8.3 mounts two host directories into the control-plane container: the client↔CP **exchange**
(``ingest`` in / ``export`` out) and the **persistent store** volume. The ADR's non-negotiable rule
is that *neither ever reaches a worker* — workers get bytes only through the CP provisioning path
(``static_contents`` → per-worker ephemeral staging). This is largely structural: a `WorkerSpec` has
**no host-bind-mount field at all**, only ``readonly_inputs`` (container-path → bytes), so the
exchange/store dirs are not even expressible as a worker mount. What a *builder bug* could still do
is copy a daemon path string (so a worker could find the volume) or the wrong bytes into a worker.

This drives the full daemon flow — exchange-style ingest → create → run — over a `ControlService`
with a **real on-disk store root** and a **distinct exchange dir**, then asserts over every launched
worker that (a) neither host dir's path leaks (into env, env-passthrough, command, image, or input
keys) and (b) the verifier's answer key never appears — the 8.1 byte-provenance invariant, restated
at the service layer on a persistent store. Offline on `FakeBackend`, so the lean CI gate covers it.
"""

from __future__ import annotations

import asyncio

from tools.harness.dataset import stratified_split, subsample

from tests._fe_offline import FeWorkers, offline_catalog, preds_csv, role_files_from_raw
from verity.composition.task_request import DataRequest, PolicyRequest, TaskRequest
from verity.contracts.jobqueue import JobStatus
from verity.domains.feature_engineering import ENTRYPOINT, SUBMISSION
from verity.provisioning import FakeBackend
from verity.service import ControlService

_RAW = b"id,a,class\n0,0,X\n1,2,X\n2,4,X\n3,6,Y\n4,8,Y\n5,10,Y\n"
_REAL_TEST = b"id\n100\n101\n102\n103\n"
_PER_CLASS = 100
_RESERVED_FRACTION = 0.5


def _reserved() -> dict[str, str]:
    return stratified_split(
        subsample(_RAW, per_class=_PER_CLASS, target="class"),
        target="class", id_column="id", reserved_fraction=_RESERVED_FRACTION,
    ).reserved_labels


def _fe_request() -> TaskRequest:
    return TaskRequest(
        type_name="fe-kaggle", goal="improve balanced accuracy",
        policy=PolicyRequest(max_cycles=1, stop_on_accept=True),
        data=DataRequest(),
    )


def _fe_files() -> dict[str, dict[str, bytes]]:
    return role_files_from_raw(
        _RAW, _REAL_TEST, per_class=_PER_CLASS, reserved_fraction=_RESERVED_FRACTION
    )


def test_neither_exchange_nor_store_reaches_a_worker(tmp_path) -> None:
    # The two daemon host volumes, as `verity serve` wires them: a persistent store root and an
    # exchange dir, kept distinct so a leak of either is attributable.
    store_root = tmp_path / "store"
    exchange_in = tmp_path / "exchange" / "in"
    exchange_in.mkdir(parents=True)
    dataset = exchange_in / "train.csv"
    dataset.write_bytes(_RAW)

    answer_key = preds_csv(_reserved())  # the verifier's secret: id -> true-label pairs

    workers = FeWorkers(
        steps=[("submit", ("ds",), b"good", ["feat0"])],
        by_marker={b"good": _reserved()},  # predicts truth -> 1.0 -> accepted
    )
    backend = FakeBackend(script=workers)
    service = ControlService(backend=backend, root=store_root, catalog=offline_catalog())

    # The exchange flow: read the ingested file's bytes (as the daemon's ingest route does), then
    # create the task from a declarative request with the user's pre-prepared, role-keyed files —
    # they land in the task's own on-disk store.
    assert dataset.read_bytes() == _RAW
    task_id = service.create_task(_fe_request(), files=_fe_files())
    run_id = asyncio.run(service.run(task_id))

    # The loop genuinely ran on the persistent store: an accepted Submission landed.
    assert asyncio.run(service.status(run_id)) is JobStatus.DONE
    assert asyncio.run(service.accepted_artifacts(task_id, type=SUBMISSION))

    launched = backend.launched
    sandboxes = [s for s in launched if s.labels.role == "sandbox"]
    code_runners = [s for s in launched if s.labels.role == "code-runner"]
    assert sandboxes and code_runners, "both worker roles should have run"

    # The host-side daemon paths a worker must never learn of — neither as a mount (inexpressible)
    # nor as a path string it could otherwise reach.
    leak_paths = [str(store_root), str(exchange_in), str(exchange_in.parent)]
    for spec in launched:
        haystack = [
            *spec.env.values(),
            *spec.env_passthrough,
            *spec.command,
            spec.image,
            *spec.readonly_inputs.keys(),  # container paths must not echo a host daemon dir
        ]
        for leak in leak_paths:
            for value in haystack:
                assert leak not in value, f"daemon path {leak!r} leaked into a {spec.labels.role}"
        # The daemon never forwards its volume env vars to a worker.
        assert "VERITY_STORE_ROOT" not in spec.env_passthrough
        assert "VERITY_EXCHANGE" not in spec.env_passthrough
        # The answer key never appears in any worker input or env (byte-provenance, real store).
        for blob in spec.readonly_inputs.values():
            assert answer_key not in blob, f"answer key leaked into a {spec.labels.role} input"
        for value in spec.env.values():
            assert answer_key.decode() not in value

    # The untrusted submission still ran in a code-runner-labelled worker (not in-process).
    assert all(f"/work/{ENTRYPOINT}" in s.readonly_inputs for s in code_runners)
