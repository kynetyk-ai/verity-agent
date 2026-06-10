"""Declarative wiring for the feature-engineering task (ROADMAP 7.4.f).

``build_fe_control_plane`` assembles a runnable `ControlPlane` for the §12 FE domain from a backend,
a dataset split, and a `ProvisioningConfig` — no hand-built drivers, code-runners, or registries.
The sandbox runs as a worker (`BackendSandboxDriver`); the FE verifier's gate logic stays **trusted
and in-process** and executes the untrusted submission as a worker (`BackendCodeRunner`) — topology
(b), so a per-cycle-disposed verifier never flushes its in-memory incumbent ledger, and the reserved
labels (the answer key) never enter any worker.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    Operation,
    OperationStatus,
    ProviderRegistry,
    SandboxPort,
    VerifierPort,
)
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import (
    DefaultRetrievalPolicy,
    ObjectProvisioningPolicy,
    ObjectProvisionMode,
)
from verity.control_plane.store import SqliteStore
from verity.domains.feature_engineering import (
    DATASET_VERSION,
    SUBMISSION,
    build_feature_engineering_domain,
    build_feature_engineering_verifier,
)
from verity.provisioning.backend import WorkerBackend
from verity.sandbox.backend_driver import BackendSandboxDriver
from verity.sandbox.model_spec import ModelSpec
from verity.sandbox.registration import build_sandbox
from verity.verifier import BackendCodeRunner

__all__ = ["ProvisioningConfig", "build_fe_control_plane", "FE_GOAL"]

_DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"

FE_GOAL = "Improve balanced accuracy via feature engineering; keep training fast and simple."
_FE_INSTRUCTIONS = (
    "Engineer 1-3 features that improve balanced accuracy. Be FAST: train one small, fixed model "
    "(no hyperparameter search, no cross-validation, no big ensembles); your edge is the features, "
    "not the model. Run the script once to confirm it works, then submit — do not keep retraining."
)


@dataclass(frozen=True, slots=True)
class ProvisioningConfig:
    """The substrate-and-worker shape, kept OFF `TaskConfig` (the control plane stays agnostic).

    ``model_spec`` is the sandbox's model target (default frontier Anthropic). ``runtime`` selects
    the OCI runtime per worker (e.g. ``"runsc"`` for gVisor; ``None`` = the daemon default).
    """

    sandbox_image: str = "verity-sandbox:latest"
    code_image: str = "python:3.12-slim"
    model_spec: ModelSpec | None = None
    runtime: str | None = None
    sandbox_memory: str = "4g"
    code_memory: str = "2g"
    code_tmpfs_size: str = "1g"
    recursion_limit: int = 200
    sandbox_timeout_s: float = 1500.0
    code_timeout_s: float = 600.0


def build_fe_control_plane(
    *,
    backend: WorkerBackend,
    split: Any,
    provisioning: ProvisioningConfig | None = None,
    max_cycles: int = 4,
    store: SqliteStore | None = None,
) -> tuple[ControlPlane, TaskConfig]:
    """Wire the FE task. ``split`` is a stratified split (``agent_train_csv`` /
    ``reserved_test_csv`` / ``reserved_labels``). Returns ``(cp, config)``; the caller then
    does ``await cp.configure(config)``."""
    provisioning = provisioning or ProvisioningConfig()
    spec = provisioning.model_spec
    model = spec.provider_string() if spec is not None else _DEFAULT_MODEL
    domain = build_feature_engineering_domain()

    store = store or SqliteStore()
    store.propose(
        Artifact("ds", DATASET_VERSION, {"source": "stellar"}, ArtifactStatus.PROPOSED,
                 "loader", "t0", is_root=True),
        Operation("op-ds", "load", (), "ds", OperationStatus.SUCCESS, "t0"),
    )

    root = Path(tempfile.mkdtemp(prefix="verity-fe-"))
    sandbox = build_sandbox(
        schema=domain.schema, root=root,
        driver=BackendSandboxDriver(
            backend=backend, model=model, spec=spec, image=provisioning.sandbox_image,
            config="fe", memory=provisioning.sandbox_memory,
            recursion_limit=provisioning.recursion_limit, timeout_s=provisioning.sandbox_timeout_s,
            runtime=provisioning.runtime,
        ),
        proposer_identity=f"deepagents-worker:{model}",
        static_contents={"data": {"train.csv": split.agent_train_csv,
                                  "test.csv": split.reserved_test_csv}},
    )
    runner = BackendCodeRunner(
        backend=backend, image=provisioning.code_image, memory=provisioning.code_memory,
        tmpfs_size=provisioning.code_tmpfs_size, runtime=provisioning.runtime, config="fe",
    )
    verifier = build_feature_engineering_verifier(
        runner,
        agent_train_csv=split.agent_train_csv,
        reserved_test_csv=split.reserved_test_csv,
        reserved_labels=split.reserved_labels,
        timeout_s=provisioning.code_timeout_s,
    )

    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("fe", lambda: sandbox)
    vp.register("fe", lambda: verifier)
    cp = ControlPlane(
        store, policy=OrchestrationPolicy(max_cycles=max_cycles),
        sandbox_providers=sp, verifier_providers=vp,
    )
    config = TaskConfig(
        task_id="fe", instructions=_FE_INSTRUCTIONS,
        domain_instructions=domain.domain_instructions, schema=domain.schema,
        gated_types=domain.gated_types, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator, sandbox_key="fe", verifier_key="fe",
        harvester=domain.harvester,
        object_provisioning=ObjectProvisioningPolicy(
            mode=ObjectProvisionMode.LAST_REVISED_OR_ACCEPTED, type_filter=SUBMISSION
        ),
    )
    return cp, config
