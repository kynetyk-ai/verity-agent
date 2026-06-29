"""The `fe-holdout` ablation task type — offline (no Docker, model, or network).

Drives the real read→propose→gate→commit loop over a `FakeBackend` (the shared `FeWorkers`
substrate) with a loopback holdout verifier built from the per-task `VerifierSetup`, exactly as the
verifier image's builder would over the wire. Pins the load-bearing claim of the ablation: the
verdict policy is the *only* difference across conditions — the same worse-than-prior submission is
**accepted** under ``always`` (Exp 1/2) and **rejected** under ``improve_over_best_prior`` (Exp
3/4/4b) — plus the catalog builder's role-file validation.
"""

from __future__ import annotations

import asyncio

import pytest

from tests._fe_offline import FeWorkers, role_files_from_raw
from verity.composition.fe_holdout import (
    FE_HOLDOUT_TASK_ID,
    build_fe_holdout_task,
    configure_fe_holdout_task,
)
from verity.composition.task_request import (
    DataRequest,
    PolicyRequest,
    SandboxRequest,
    TaskRequest,
    VerifierRequest,
)
from verity.contracts import ArtifactStatus
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.registries import ObjectProvisionMode
from verity.control_plane.store import SqliteStore
from verity.domains.feature_engineering import SUBMISSION
from verity.provisioning import FakeBackend

# Three classes of two rows each so balanced accuracy is meaningful; per_class/reserved_fraction
# below keep all of it for a deterministic split.
_RAW = b"id,a,class\n0,0,X\n1,2,X\n2,4,Y\n3,6,Y\n4,8,Z\n5,10,Z\n"
_TEST = b"id\n100\n101\n"


def _loopback_holdout_factory(backend):  # type: ignore[no-untyped-def]
    """An offline ``make_verifier`` for `configure_fe_holdout_task` — the §12 scorer over a loopback
    `RemoteVerifier`, built from the shipped `VerifierSetup` (the same seam the HTTP path uses)."""
    from verity.contracts.ports import VerifierSetup
    from verity.domains.feature_engineering import (
        build_feature_engineering_verifier,
        parse_label_csv,
    )
    from verity.transport import LoopbackTransport, RemoteVerifier, VerifierServer
    from verity.verifier import BackendCodeRunner

    def _builder(setup: VerifierSetup):  # type: ignore[no-untyped-def]
        runner = BackendCodeRunner(backend=backend, config="holdout-experiment")
        return build_feature_engineering_verifier(
            runner,
            agent_train_csv=setup.objects["train.csv"],
            reserved_test_csv=setup.objects["holdout.csv"],
            reserved_labels=parse_label_csv(setup.objects["holdout_labels.csv"]),
            margin=float(setup.params.get("margin", 0.0)),
            timeout_s=float(setup.params.get("timeout_s", 900.0)),
            accept_policy=str(setup.params.get("accept_policy", "improve_over_best_prior")),
        )

    async def make(setup: VerifierSetup):  # type: ignore[no-untyped-def]
        return RemoteVerifier(
            LoopbackTransport(VerifierServer(builder=_builder).handle), setup_payload=setup
        )

    return make


def _run(accept_policy: str) -> SqliteStore:
    """Two cycles: a perfect submission then a strictly-worse one, under ``accept_policy``."""
    from tools.harness.dataset import stratified_split, subsample

    rf = role_files_from_raw(_RAW, _TEST, per_class=2, reserved_fraction=0.5)
    sub = subsample(_RAW, per_class=2, target="class")
    split = stratified_split(sub, target="class", id_column="id", reserved_fraction=0.5)
    reserved = dict(split.reserved_labels)
    worse = dict.fromkeys(reserved, next(iter(reserved.values())))  # all one class → low score

    steps = [("submit", ("ds",), b"a", ["f0"]), ("submit", ("ds",), b"b", ["f0"])]
    by_marker = {b"a": reserved, b"b": worse}
    backend = FakeBackend(script=FeWorkers(steps=steps, by_marker=by_marker))
    store = SqliteStore()
    cp = ControlPlane(store, policy=OrchestrationPolicy(max_cycles=2))
    asyncio.run(
        configure_fe_holdout_task(
            cp, backend=backend, make_verifier=_loopback_holdout_factory(backend),
            agent_files=rf["agent"], verifier_files=rf["verifier"],
            accept_policy=accept_policy, provisioning_spec=ObjectProvisionMode.ALL,
        )
    )
    asyncio.run(cp.run(FE_HOLDOUT_TASK_ID, goal="optimize balanced accuracy"))
    return store


def test_always_accepts_the_worse_second_submission() -> None:
    store = _run("always")
    accepted = store.query_artifacts(type=SUBMISSION, status=ArtifactStatus.ACCEPTED)
    assert len(accepted) == 2  # both submissions accepted; nothing rejected, nothing superseded
    assert not store.rejected_log(type=SUBMISSION)


def test_improve_over_best_prior_rejects_the_worse_second_submission() -> None:
    store = _run("improve_over_best_prior")
    rejected = store.rejected_log(type=SUBMISSION)
    assert len(rejected) == 1  # the worse second submission is rejected
    accepted = store.query_artifacts(type=SUBMISSION, status=ArtifactStatus.ACCEPTED)
    assert len(accepted) == 1  # only the first (perfect) submission stands


def test_build_validates_verifier_role_files() -> None:
    store = SqliteStore()
    rf = role_files_from_raw(_RAW, _TEST, per_class=2, reserved_fraction=0.5)
    data = DataRequest()
    for name, blob in rf["agent"].items():
        data = data.with_role_file("agent", name, store.put_object(blob).content_hash)
    for name, blob in rf["verifier"].items():
        if name == "holdout_labels.csv":
            continue  # the answer key is missing → the builder must fail fast (misuse)
        data = data.with_role_file("verifier", name, store.put_object(blob).content_hash)
    request = TaskRequest(
        type_name=FE_HOLDOUT_TASK_ID, sandbox=SandboxRequest(),
        verifier=VerifierRequest(knobs={"accept_policy": "always"}),
        policy=PolicyRequest(), data=data,
    )
    cp = ControlPlane(store)
    with pytest.raises(ValueError, match="holdout_labels.csv"):
        asyncio.run(build_fe_holdout_task(cp, backend=FakeBackend(), request=request))


def test_fe_tasks_default_to_the_ml_equipped_sandbox_image() -> None:
    # FE tasks must run on the ML-equipped sandbox so the agent can self-test its script (else it
    # submits blind and the gate eats runnable bugs). The generic base default is swapped for the FE
    # image; an explicit operator override is respected.
    from verity.composition.fe import (
        BASE_SANDBOX_IMAGE,
        FE_SANDBOX_IMAGE,
        ProvisioningConfig,
        with_fe_sandbox_image,
    )

    assert ProvisioningConfig().sandbox_image == BASE_SANDBOX_IMAGE  # the generic default
    assert with_fe_sandbox_image(ProvisioningConfig()).sandbox_image == FE_SANDBOX_IMAGE
    custom = ProvisioningConfig(sandbox_image="my-custom-sandbox:tag")
    assert with_fe_sandbox_image(custom).sandbox_image == "my-custom-sandbox:tag"
