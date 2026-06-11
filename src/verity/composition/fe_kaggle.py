"""Apply the FE-Kaggle task to a generic control plane (the `fe-kaggle` task type).

Reuses the §12 feature-engineering domain (schema / shape / instructions / harvester) and the
worker-provisioning wiring of :mod:`verity.composition.fe`; the only swap is the **verifier**, whose
authoritative gate is the real Kaggle leaderboard (`build_feature_engineering_kaggle_verifier`).

Data the gate holds (trusted, in-process): the labeled hold-out split of ``train.csv`` (for the
cheap proxy) plus the full labeled subsample and the **real, unlabeled ``test.csv``** (for the
Kaggle run). The agent's sandbox sees only ``agent_train`` (labeled, hold-out removed) + the real
``test.csv`` (unlabeled) — never the reserved labels nor the real test's labels (Kaggle's). The
Kaggle creds live on the verifier side (the control plane), never in a worker.

``configure_fe_kaggle_task`` takes a ready-made `KaggleScorer` (a `FakeKaggleScorer` offline; the
real client in the catalog builder, Sprint 2). ``describe_fe_kaggle_task`` publishes the contract
for ``verity catalog``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from verity.composition.dataset import stratified_split, subsample
from verity.composition.description import OperationDescription, TaskTypeDescription
from verity.composition.fe import ProvisioningConfig, provisioning_config_from
from verity.composition.task_request import TaskRequest
from verity.contracts import Artifact, ArtifactStatus, Operation, OperationStatus
from verity.control_plane.api import ControlPlane
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import (
    DefaultRetrievalPolicy,
    ObjectProvisioningPolicy,
    ObjectProvisionMode,
)
from verity.domains.feature_engineering import (
    DATASET_VERSION,
    SUBMISSION,
    build_feature_engineering_domain,
)
from verity.domains.feature_engineering_kaggle import build_feature_engineering_kaggle_verifier
from verity.provisioning.backend import WorkerBackend
from verity.sandbox.backend_driver import BackendSandboxDriver
from verity.sandbox.registration import build_sandbox
from verity.verifier import BackendCodeRunner
from verity.verifier.kaggle import KaggleScorer

__all__ = [
    "FE_KAGGLE_TASK_ID",
    "FE_KAGGLE_GOAL",
    "configure_fe_kaggle_task",
    "build_fe_kaggle_task",
    "describe_fe_kaggle_task",
]

FE_KAGGLE_TASK_ID = "fe-kaggle"
FE_KAGGLE_GOAL = (
    "Climb the real Kaggle leaderboard for this competition with a fast, self-contained pipeline."
)

_DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"

_FE_KAGGLE_INSTRUCTIONS = (
    "Improve balanced accuracy by any means that fits in one script — features, model choice, a "
    "small ensemble, calibration, imbalance handling — then submit. If there's no incumbent yet, "
    "explore the data with code first; if there is one (under scratch/provided/), read it and "
    "target its weakness. Predict EVERY row of test.csv: your predictions are scored first on a "
    "hidden hold-out, then submitted to the REAL Kaggle leaderboard, which decides acceptance. "
    "Anything that leaks the target inflates your own number but fails on the held-out and real "
    "data. Keep it fast enough to finish the time budget."
)

_FE_KAGGLE_VERIFIER_APPROACH = (
    "Two-tier ladder over the regenerated script (the gate re-runs the script on gold data; the "
    "agent's own CSV is never trusted). CHEAP, per-cycle: the script runs on a labeled hold-out "
    "carved from train.csv and is scored on balanced accuracy; a submission that does not beat the "
    "best ACCEPTED proxy is rejected before any Kaggle call. HARD, rate-limited (~5/day): the "
    "survivor is regenerated on the full train + real test set and submitted to the live Kaggle "
    "competition; it is ACCEPTED only if the public score beats our best prior Kaggle-confirmed "
    "score (the real leaderboard is the incumbent ladder). When the daily cap is spent the gate "
    "blocks until it frees; a Kaggle API failure degrades the cycle (recorded), not crash the run."
)
_FE_KAGGLE_SANDBOX_NOTES = (
    "A general-purpose coding agent that writes and runs Python in an isolated worker; needs the "
    "container sandbox image. The submission is regenerated and submitted to Kaggle on the trusted "
    "verifier side — credentials never reach the agent's worker."
)


async def configure_fe_kaggle_task(
    cp: ControlPlane,
    *,
    backend: WorkerBackend,
    scorer: KaggleScorer,
    split: Any,
    full_train_csv: bytes,
    real_test_csv: bytes,
    provisioning: ProvisioningConfig | None = None,
    wait_deadline_s: float = 86_400.0,
    poll_interval_s: float = 60.0,
    submit_message: str = "verity",
) -> str:
    """Configure the `fe-kaggle` task onto a generic ``cp`` via its API. Returns the task id.

    ``split`` is the labeled hold-out split of the subsample (``agent_train_csv`` /
    ``reserved_test_csv`` / ``reserved_labels``, the cheap proxy); ``full_train_csv`` is the full
    labeled subsample and ``real_test_csv`` the competition's real (unlabeled) test set (for the
    Kaggle run). ``scorer`` is the `KaggleScorer` the hard gate submits through.
    """
    provisioning = provisioning or ProvisioningConfig()
    spec = provisioning.model_spec
    model = spec.provider_string() if spec is not None else _DEFAULT_MODEL
    domain = build_feature_engineering_domain()

    cp.store.propose(
        Artifact("ds", DATASET_VERSION, {"source": "kaggle"}, ArtifactStatus.PROPOSED,
                 "loader", "t0", is_root=True),
        Operation("op-ds", "load", (), "ds", OperationStatus.SUCCESS, "t0"),
    )

    root = Path(tempfile.mkdtemp(prefix="verity-fe-kaggle-"))
    sandbox = build_sandbox(
        schema=domain.schema, root=root,
        driver=BackendSandboxDriver(
            backend=backend, model=model, spec=spec, image=provisioning.sandbox_image,
            config=FE_KAGGLE_TASK_ID, memory=provisioning.sandbox_memory,
            recursion_limit=provisioning.recursion_limit, timeout_s=provisioning.sandbox_timeout_s,
            runtime=provisioning.runtime,
        ),
        proposer_identity=f"deepagents-worker:{model}",
        # The agent develops against the REAL (unlabeled) test set; agent_train has the reserved
        # rows removed, so it can never see the hold-out labels (nor the real test's labels).
        static_contents={"data": {"train.csv": split.agent_train_csv, "test.csv": real_test_csv}},
    )
    runner = BackendCodeRunner(
        backend=backend, image=provisioning.code_image, memory=provisioning.code_memory,
        tmpfs_size=provisioning.code_tmpfs_size, runtime=provisioning.runtime,
        config=FE_KAGGLE_TASK_ID,
    )
    verifier = build_feature_engineering_kaggle_verifier(
        runner,
        scorer=scorer,
        agent_train_csv=split.agent_train_csv,
        reserved_test_csv=split.reserved_test_csv,
        reserved_labels=split.reserved_labels,
        full_train_csv=full_train_csv,
        real_test_csv=real_test_csv,
        timeout_s=provisioning.code_timeout_s,
        wait_deadline_s=wait_deadline_s,
        poll_interval_s=poll_interval_s,
        submit_message=submit_message,
    )

    cp.register_sandbox(FE_KAGGLE_TASK_ID, lambda: sandbox)
    cp.register_verifier(FE_KAGGLE_TASK_ID, lambda: verifier)
    config = TaskConfig(
        task_id=FE_KAGGLE_TASK_ID, instructions=_FE_KAGGLE_INSTRUCTIONS,
        domain_instructions=domain.domain_instructions, schema=domain.schema,
        gated_types=domain.gated_types, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator,
        sandbox_key=FE_KAGGLE_TASK_ID, verifier_key=FE_KAGGLE_TASK_ID,
        harvester=domain.harvester,
        object_provisioning=ObjectProvisioningPolicy(
            mode=ObjectProvisionMode.LAST_REVISED_OR_ACCEPTED, type_filter=SUBMISSION
        ),
    )
    await cp.configure(config)
    return FE_KAGGLE_TASK_ID


async def build_fe_kaggle_task(
    cp: ControlPlane, *, backend: WorkerBackend, request: TaskRequest
) -> str:
    """The `fe-kaggle` catalog builder: configure the task from a declarative `TaskRequest`.

    Needs two ingested inputs (the carve-out is the same as `build_fe_task`, plus the real test):
    ``data.data_ref`` (the labeled ``train.csv``, split here for the cheap proxy) and
    ``data.test_ref`` (the real, unlabeled ``test.csv`` for Kaggle). ``verifier.knobs.competition``
    is the competition slug (required); optional knobs tune polling/wait. Constructs the live
    `RealKaggleScorer` (lazy `kaggle` import) and hands it to `configure_fe_kaggle_task`.
    """
    if request.data.data_ref is None:
        raise ValueError("the fe-kaggle task requires training data (create the task with `data=`)")
    if request.data.test_ref is None:
        raise ValueError("the fe-kaggle task requires the real test set (`data.test_ref`)")
    knobs = request.verifier.knobs
    competition = knobs.get("competition")
    if not competition:
        raise ValueError("fe-kaggle requires verifier.knobs['competition'] (the competition slug)")

    raw = cp.store.get_object(request.data.data_ref)
    real_test = cp.store.get_object(request.data.test_ref)
    sub = subsample(raw, per_class=request.data.per_class, target=request.data.target)
    split = stratified_split(
        sub, target=request.data.target, id_column=request.data.id_column,
        reserved_fraction=request.data.reserved_fraction,
    )

    from verity.verifier import (
        RealKaggleScorer,  # lazy: only the live path needs the `kaggle` extra
    )

    scorer = RealKaggleScorer(
        competition=str(competition),
        poll_interval_s=float(knobs.get("score_poll_interval_s", 20.0)),
    )
    return await configure_fe_kaggle_task(
        cp, backend=backend, scorer=scorer, split=split,
        full_train_csv=sub, real_test_csv=real_test,
        provisioning=provisioning_config_from(request.sandbox),
        wait_deadline_s=float(knobs.get("wait_deadline_s", 86_400.0)),
        poll_interval_s=float(knobs.get("budget_poll_interval_s", 60.0)),
        submit_message=str(knobs.get("submit_message", "verity fe-kaggle")),
    )


def describe_fe_kaggle_task() -> TaskTypeDescription:
    """The `fe-kaggle` task type's published contract (ADR 0004 (c)) for ``verity catalog``."""
    domain = build_feature_engineering_domain()
    schema = domain.schema
    return TaskTypeDescription(
        type_name=FE_KAGGLE_TASK_ID,
        artifact_types=tuple(t.name for t in schema.types()),
        gated_types=tuple(t.name for t in schema.types() if domain.gated_types.is_gated(t.name)),
        operations=tuple(
            OperationDescription(o.name, o.inputs, o.output, o.required_payload_keys)
            for o in schema.operations()
        ),
        domain_instructions=domain.domain_instructions,
        verifier_approach=_FE_KAGGLE_VERIFIER_APPROACH,
        sandbox_notes=_FE_KAGGLE_SANDBOX_NOTES,
    )
