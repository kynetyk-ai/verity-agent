"""The `ControlService` core lifecycle (ROADMAP 8.1, ADR 0004): create → run → results, durable.

These pin the service facade the daemon (8.2) and HTTP API (8.4) will adapt: a task is created from
a declarative `TaskRequest` (its data ingested into its own store), run once over a `WorkerBackend`,
and its results read back as a daemon-level `RunRecord` keyed by ``(tenant_id, run_id)``. The
durability property — a created task survives a restart because its definition persists — is proven
by rebuilding a fresh service over the same on-disk root. The catalog dispatches >1 task type.
"""

from __future__ import annotations

import asyncio
import json

from tests._fe_offline import FeWorkers
from verity.composition.dataset import stratified_split, subsample
from verity.composition.task_request import (
    DataRequest,
    PolicyRequest,
    SandboxRequest,
    TaskRequest,
    VerifierRequest,
)
from verity.contracts.jobqueue import JobStatus
from verity.domains.feature_engineering import SUBMISSION
from verity.provisioning import FakeBackend
from verity.service import ControlService

_RAW = b"id,a,class\n0,0,X\n1,2,X\n2,4,X\n3,6,Y\n4,8,Y\n5,10,Y\n"


def _reserved() -> dict[str, str]:
    return stratified_split(
        subsample(_RAW, per_class=100, target="class"),
        target="class", id_column="id", reserved_fraction=0.5,
    ).reserved_labels


def _fe_request() -> TaskRequest:
    return TaskRequest(type_name="fe", goal="improve balanced accuracy",
                       policy=PolicyRequest(stop_on_accept=True),
                       data=DataRequest(per_class=100, reserved_fraction=0.5))


def _fe_workers() -> FeWorkers:
    return FeWorkers(steps=[("submit", ("ds",), b"good", ["feat0"])],
                     by_marker={b"good": _reserved()})


def test_create_run_results() -> None:
    service = ControlService(backend=FakeBackend(script=_fe_workers()), root=None)
    task_id = service.create_task(_fe_request(), data=_RAW)
    run_id = asyncio.run(service.run(task_id))

    assert asyncio.run(service.status(run_id)) is JobStatus.DONE
    record = service.results(run_id)
    assert record is not None
    assert record.status == "complete"
    assert record.run_id == run_id and record.tenant_id == "default"
    # the accepted Submission is reachable both via the record's pointers and the live store.
    assert record.accepted_artifact_ids
    assert asyncio.run(service.accepted_artifacts(task_id, type=SUBMISSION))


def test_task_definition_is_durable_across_restart(tmp_path) -> None:
    # Service 1 only *creates* the task (persisting its TaskRequest + ingesting data to disk).
    creator = ControlService(backend=FakeBackend(script=FeWorkers(steps=[], by_marker={})),
                             root=tmp_path)
    task_id = creator.create_task(_fe_request(), data=_RAW)
    assert task_id in creator.list_tasks()

    # Service 2 is a *fresh* process view over the same root: no resident tasks, no shared memory.
    restarted = ControlService(backend=FakeBackend(script=_fe_workers()), root=tmp_path)
    assert task_id in restarted.list_tasks(), "the task definition did not survive the restart"
    run_id = asyncio.run(restarted.run(task_id))  # rebuilds the CP from the persisted request

    assert asyncio.run(restarted.status(run_id)) is JobStatus.DONE
    assert asyncio.run(restarted.accepted_artifacts(task_id, type=SUBMISSION))


def test_task_request_json_round_trip() -> None:
    request = TaskRequest(
        type_name="fe", goal="g",
        sandbox=SandboxRequest(model="anthropic:claude-sonnet-4-6", sandbox_memory="2g"),
        verifier=VerifierRequest(approach="balanced_accuracy", knobs={"margin": 0.01}),
        policy=PolicyRequest(max_cycles=7, stop_on_accept=True),
        data=DataRequest(data_ref="abc123", per_class=50),
        tenant_id="default",
    )
    assert TaskRequest.from_dict(json.loads(json.dumps(request.to_dict()))) == request


def test_local_model_name_with_colons_is_not_split() -> None:
    # A base_url means an OpenAI-compatible endpoint, where the model is a literal server-side name
    # that can contain colons (Ollama's "qwen3.6:27b-coding-mxfp8"). It must not be split on ':'.
    spec = SandboxRequest(
        model="qwen3.6:27b-coding-mxfp8", base_url="http://host.docker.internal:11434/v1",
    ).to_model_spec()
    assert spec is not None
    assert spec.model == "qwen3.6:27b-coding-mxfp8"  # full name preserved
    # The native-provider path (no base_url) still splits provider:model.
    native = SandboxRequest(model="anthropic:claude-sonnet-4-6").to_model_spec()
    assert native is not None
    assert (native.provider, native.model) == ("anthropic", "claude-sonnet-4-6")


def test_catalog_dispatches_a_second_task_type() -> None:
    # Genericity on the service path: the same multiplexer instantiates an unrelated task type.
    service = ControlService(backend=FakeBackend(script=FeWorkers(steps=[], by_marker={})),
                             root=None)
    assert set(service.catalog_types()) == {"fe", "code"}
    code_task = service.create_task(TaskRequest(type_name="code"))
    resident = asyncio.run(service._resident_cp(code_task))
    assert resident.in_cp_task_id == "code"

    # It is a distinct control plane + store from an FE instance on the same service.
    fe_task = service.create_task(_fe_request(), data=_RAW)
    fe_resident = asyncio.run(service._resident_cp(fe_task))
    assert resident.cp is not fe_resident.cp
    assert resident.cp.store is not fe_resident.cp.store


def test_unknown_task_and_type_are_typed_errors() -> None:
    from verity.service import UnknownTask

    service = ControlService(backend=FakeBackend(script=FeWorkers(steps=[], by_marker={})),
                             root=None)
    try:
        service.create_task(TaskRequest(type_name="nope"))
        raise AssertionError("expected UnknownTask for an unknown task type")
    except UnknownTask:
        pass
    try:
        asyncio.run(service.run("no-such-task"))
        raise AssertionError("expected UnknownTask for an unknown task id")
    except UnknownTask:
        pass
