"""The positive byte-provenance guarantee for the FE carve-out (ROADMAP 8.1, ADR 0004 (f)).

The riskiest part of 8.1 is that the split + answer-key derivation now happens *inside the builder*
against client-supplied data. The real isolation guarantee is not "the exchange dir is unmounted"
(an argv check) but a **byte-provenance** assertion: a worker's read-only inputs contain *only* the
intended role's bytes, and the verifier's ``reserved_labels`` (the answer key) never appear in *any*
worker input. This test builds the FE task from a `TaskRequest` over a `FakeBackend`, runs a cycle,
and asserts exactly that over every captured `WorkerSpec` — proving the answer key cannot reach a
worker through the new client-data path.
"""

from __future__ import annotations

import asyncio

from tests._fe_offline import FeWorkers, preds_csv
from verity.composition.dataset import stratified_split, subsample
from verity.composition.fe import build_fe_task
from verity.composition.task_request import DataRequest, TaskRequest
from verity.contracts import ArtifactStatus
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.store import SqliteStore
from verity.domains.feature_engineering import ENTRYPOINT, SUBMISSION
from verity.provisioning import FakeBackend

# 2 classes × 3 rows; with reserved_fraction 0.5 the even interleave reserves the middle row of each
# class -> reserved ids {"1","4"} with labels {X, Y}. Recomputed in-test, never hard-coded.
_RAW = b"id,a,class\n0,0,X\n1,2,X\n2,4,X\n3,6,Y\n4,8,Y\n5,10,Y\n"
_PER_CLASS = 100
_RESERVED_FRACTION = 0.5


def test_answer_key_never_reaches_a_worker() -> None:
    expected = stratified_split(
        subsample(_RAW, per_class=_PER_CLASS, target="class"),
        target="class", id_column="id", reserved_fraction=_RESERVED_FRACTION,
    )
    answer_key = preds_csv(expected.reserved_labels)  # id->true-label pairs: the verifier's secret

    workers = FeWorkers(
        steps=[("submit", ("ds",), b"good", ["feat0"])],
        by_marker={b"good": dict(expected.reserved_labels)},  # predicts truth -> 1.0 -> accepted
    )
    backend = FakeBackend(script=workers)
    store = SqliteStore()
    ref = store.put_object(_RAW)  # client data ingested into the task's own store (immutable)
    cp = ControlPlane(store, policy=OrchestrationPolicy(max_cycles=1, stop_on_accept=True))
    asyncio.run(build_fe_task(
        cp, backend=backend,
        request=TaskRequest(
            type_name="fe",
            data=DataRequest(data_ref=ref.content_hash, per_class=_PER_CLASS,
                             reserved_fraction=_RESERVED_FRACTION),
        ),
    ))
    asyncio.run(cp.run("fe", goal="improve balanced accuracy"))

    # The submission was accepted on a real (scripted) evaluation -> the loop genuinely ran.
    assert store.query_artifacts(type=SUBMISSION, status=ArtifactStatus.ACCEPTED)

    launched = backend.launched
    sandboxes = [s for s in launched if s.labels.role == "sandbox"]
    code_runners = [s for s in launched if s.labels.role == "code-runner"]
    assert sandboxes and code_runners, "both worker roles should have run"

    # (1) the answer key never appears in any worker input or env, by construction.
    for spec in launched:
        for blob in spec.readonly_inputs.values():
            assert answer_key not in blob, f"answer key leaked into a {spec.labels.role} input"
        for value in spec.env.values():
            assert answer_key.decode() not in value

    # (2) the sandbox worker receives only the agent's (labelled) training rows — the intended role.
    assert all(
        s.readonly_inputs.get("/work/data/train.csv") == expected.agent_train_csv
        for s in sandboxes
    )

    # (3) the code-runner receives only the unlabelled reserved rows (no 'class' column).
    assert all(
        s.readonly_inputs.get("/data/test.csv") == expected.reserved_test_csv for s in code_runners
    )
    assert b"class" not in expected.reserved_test_csv

    # (4) the untrusted submission ran in a code-runner-labelled worker (not in-process).
    assert all(f"/work/{ENTRYPOINT}" in s.readonly_inputs for s in code_runners)
