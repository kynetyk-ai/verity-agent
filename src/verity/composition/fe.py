"""Apply the feature-engineering task to a generic control plane (ROADMAP 7.4.f, decoupled).

``configure_fe_task`` takes an already-constructed, **task-agnostic** `ControlPlane` and configures
the §12 FE task onto it *through the CP's own API* — registering the FE sandbox + verifier providers
(`cp.register_sandbox`/`register_verifier`), seeds the dataset root into the CP's store, and calls
`cp.configure`. The control plane is never built around FE; FE is one task among many it could run
(see `configure_code_task` for the same CP running a different task).

The sandbox runs as a worker (`BackendSandboxDriver`); the FE verifier's gate logic stays **trusted
and in-process** and executes the untrusted submission as a worker (`BackendCodeRunner`) — topology
(b), so a per-cycle-disposed verifier never flushes its in-memory incumbent ledger, and the reserved
labels (the answer key) never enter any worker. This `composition` layer is the one place allowed to
import across services; the substrate shape (`ProvisioningConfig`) stays off `TaskConfig`.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.composition.dataset import stratified_split, subsample
from verity.composition.description import OperationDescription, TaskTypeDescription
from verity.composition.task_request import SandboxRequest, TaskRequest
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
    build_feature_engineering_verifier,
    declared_objects,
)
from verity.provisioning.backend import WorkerBackend
from verity.sandbox.backend_driver import BackendSandboxDriver
from verity.sandbox.model_spec import ModelSpec
from verity.sandbox.registration import build_sandbox
from verity.verifier import BackendCodeRunner

__all__ = [
    "ProvisioningConfig",
    "configure_fe_task",
    "build_fe_task",
    "describe_fe_task",
    "provisioning_config_from",
    "FE_GOAL",
    "FE_TASK_ID",
]

_FE_VERIFIER_APPROACH = (
    "Two gates over one cached run of the submitted script (train on the agent's data, predict the "
    "reserved hold-out the agent never sees). The cheap 'runs-clean' gate (rung 2, free) earns "
    "'tentative' on a clean exit with a well-formed prediction for every reserved row. The hard "
    "'selection' gate (rung 1) scores balanced accuracy, deflated by the rejected-log trial "
    "count, and accepts only when it beats the incumbent — so acceptance means a measured "
    "improvement on held-out data, not just that the code ran. The submission is the unit; the "
    "gate is reject-only (no refine) and does not care how the gain was achieved (features, model, "
    "ensemble, ...)."
)
_FE_SANDBOX_NOTES = (
    "A general-purpose coding agent that writes and runs Python in an isolated worker; needs the "
    "container sandbox image (the in-process sandbox cannot execute code)."
)

_DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"

FE_TASK_ID = "fe"
FE_GOAL = "Improve balanced accuracy on a held-out set; keep the pipeline fast and self-contained."
_FE_INSTRUCTIONS = (
    "Improve balanced accuracy by any means that fits in one script — engineered features, model "
    "choice, a small ensemble, calibration, imbalance handling. If there's no incumbent yet, "
    "explore the data with code first; if there is one (under scratch/provided/), read it and "
    "target its weakness. Keep the pipeline fast enough to finish the gate's time budget; run the "
    "script once to confirm it works, then submit."
)


@dataclass(frozen=True, slots=True)
class ProvisioningConfig:
    """The substrate-and-worker shape, kept OFF `TaskConfig` (the control plane stays agnostic).

    ``model_spec`` is the sandbox's model target (default frontier Anthropic). ``runtime`` selects
    the OCI runtime per worker (e.g. ``"runsc"`` for gVisor; ``None`` = the daemon default).
    """

    sandbox_image: str = "verity-sandbox:latest"
    code_image: str = "verity-code-runner:latest"
    model_spec: ModelSpec | None = None
    runtime: str | None = None
    sandbox_memory: str = "4g"
    code_memory: str = "2g"
    code_tmpfs_size: str = "1g"
    recursion_limit: int = 200
    sandbox_timeout_s: float = 1500.0
    code_timeout_s: float = 600.0


async def configure_fe_task(
    cp: ControlPlane,
    *,
    backend: WorkerBackend,
    split: Any,
    provisioning: ProvisioningConfig | None = None,
) -> str:
    """Configure the §12 FE task onto a generic ``cp`` via its API. Returns the task id.

    ``split`` is a stratified split (``agent_train_csv`` / ``reserved_test_csv`` /
    ``reserved_labels``). After this returns, ``await cp.run(FE_TASK_ID, goal=FE_GOAL)`` runs it.
    """
    provisioning = provisioning or ProvisioningConfig()
    spec = provisioning.model_spec
    model = spec.provider_string() if spec is not None else _DEFAULT_MODEL
    domain = build_feature_engineering_domain()

    # Seed the dataset root into the control plane's store (task input data, set up once).
    cp.store.propose(
        Artifact("ds", DATASET_VERSION, {"source": "stellar"}, ArtifactStatus.PROPOSED,
                 "loader", "t0", is_root=True),
        Operation("op-ds", "load", (), "ds", OperationStatus.SUCCESS, "t0"),
    )

    root = Path(tempfile.mkdtemp(prefix="verity-fe-"))
    sandbox = build_sandbox(
        schema=domain.schema, root=root,
        driver=BackendSandboxDriver(
            backend=backend, model=model, spec=spec, image=provisioning.sandbox_image,
            config=FE_TASK_ID, memory=provisioning.sandbox_memory,
            recursion_limit=provisioning.recursion_limit, timeout_s=provisioning.sandbox_timeout_s,
            runtime=provisioning.runtime,
        ),
        proposer_identity=f"deepagents-worker:{model}",
        static_contents={"data": {"train.csv": split.agent_train_csv,
                                  "test.csv": split.reserved_test_csv}},
    )
    runner = BackendCodeRunner(
        backend=backend, image=provisioning.code_image, memory=provisioning.code_memory,
        tmpfs_size=provisioning.code_tmpfs_size, runtime=provisioning.runtime, config=FE_TASK_ID,
    )
    verifier = build_feature_engineering_verifier(
        runner,
        agent_train_csv=split.agent_train_csv,
        reserved_test_csv=split.reserved_test_csv,
        reserved_labels=split.reserved_labels,
        timeout_s=provisioning.code_timeout_s,
    )

    # Apply the task through the CP's API — the CP never knew what FE is.
    cp.register_sandbox(FE_TASK_ID, lambda: sandbox)
    cp.register_verifier(FE_TASK_ID, lambda: verifier)
    config = TaskConfig(
        task_id=FE_TASK_ID, instructions=_FE_INSTRUCTIONS,
        domain_instructions=domain.domain_instructions, schema=domain.schema,
        gated_types=domain.gated_types, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator, object_namer=declared_objects,
        sandbox_key=FE_TASK_ID, verifier_key=FE_TASK_ID,
        object_provisioning=ObjectProvisioningPolicy(
            mode=ObjectProvisionMode.LAST_REVISED_OR_ACCEPTED, type_filter=SUBMISSION
        ),
    )
    await cp.configure(config)
    return FE_TASK_ID


def provisioning_config_from(sandbox: SandboxRequest) -> ProvisioningConfig:
    """Reconstruct the (off-`TaskConfig`) `ProvisioningConfig` from a declarative `SandboxRequest`.

    Shared by the FE and `code` catalog builders so the request -> substrate mapping lives in one
    place. The control plane never sees this; it stays on the provisioning side (ADR 0003).
    """
    return ProvisioningConfig(
        sandbox_image=sandbox.sandbox_image,
        code_image=sandbox.code_image,
        model_spec=sandbox.to_model_spec(),
        runtime=sandbox.runtime,
        sandbox_memory=sandbox.sandbox_memory,
        code_memory=sandbox.code_memory,
        code_tmpfs_size=sandbox.code_tmpfs_size,
        recursion_limit=sandbox.recursion_limit,
        sandbox_timeout_s=sandbox.sandbox_timeout_s,
        code_timeout_s=sandbox.code_timeout_s,
    )


async def build_fe_task(cp: ControlPlane, *, backend: WorkerBackend, request: TaskRequest) -> str:
    """The FE catalog builder: configure the §12 task from a declarative `TaskRequest`.

    **The carve-out (ADR 0004 (b), the riskiest new logic):** the dataset split + the answer-key
    (``reserved_labels``) derivation, which today happen in the *caller* (`fe_run`), move *here* —
    against client-supplied data read from the task's own store by its immutable ``data_ref``. The
    split is then wired exactly as `configure_fe_task` does: ``agent_train_csv`` /
    ``reserved_test_csv`` into the sandbox worker's read-only role, ``reserved_labels`` into the
    in-process verifier **only** — so the answer key is absent from every worker (proven by a
    byte-provenance test). Returns the task id.
    """
    if request.data.data_ref is None:
        raise ValueError("the FE task requires input data; create the task with `data=<csv bytes>`")
    raw = cp.store.get_object(request.data.data_ref)
    sub = subsample(raw, per_class=request.data.per_class, target=request.data.target)
    split = stratified_split(
        sub,
        target=request.data.target,
        id_column=request.data.id_column,
        reserved_fraction=request.data.reserved_fraction,
    )
    return await configure_fe_task(
        cp, backend=backend, split=split, provisioning=provisioning_config_from(request.sandbox)
    )


def describe_fe_task() -> TaskTypeDescription:
    """The FE task type's published contract (ADR 0004 (c)) — built from the same domain schema that
    renders the agent's system prompt, so the published and graded shapes are one source."""
    domain = build_feature_engineering_domain()
    schema = domain.schema
    return TaskTypeDescription(
        type_name=FE_TASK_ID,
        artifact_types=tuple(t.name for t in schema.types()),
        gated_types=tuple(t.name for t in schema.types() if domain.gated_types.is_gated(t.name)),
        operations=tuple(
            OperationDescription(o.name, o.inputs, o.output, o.required_payload_keys)
            for o in schema.operations()
        ),
        domain_instructions=domain.domain_instructions,
        verifier_approach=_FE_VERIFIER_APPROACH,
        sandbox_notes=_FE_SANDBOX_NOTES,
    )
