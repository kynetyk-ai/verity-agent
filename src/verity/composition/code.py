"""Apply the trivial `code` task to a generic control plane — the genericity demonstration.

This exists to prove the control plane is task-agnostic: the *same* `ControlPlane` class that runs
FE (`configure_fe_task`) runs a completely different task here, applied the same way — through the
CP's API (`register_sandbox`/`register_verifier` + `configure`). The control plane has no knowledge
of either task; both are configured onto it. The `code` task is the trivial "write submission.py
that runs" task with a boolean runs-clean gate.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from verity.composition.fe import ProvisioningConfig
from verity.contracts import Artifact, ArtifactStatus, Operation, OperationStatus
from verity.control_plane.api import ControlPlane
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import DefaultRetrievalPolicy
from verity.domains.code import DATASET, build_code_domain
from verity.provisioning.backend import WorkerBackend
from verity.sandbox.backend_driver import BackendSandboxDriver
from verity.sandbox.registration import build_sandbox
from verity.verifier import BackendCodeRunner, CodeRunner

__all__ = ["configure_code_task", "CODE_TASK_ID", "CODE_GOAL"]

_DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"

CODE_TASK_ID = "code"
CODE_GOAL = "Write a submission script that runs cleanly."
_CODE_INSTRUCTIONS = (
    "Write submission.py to outbox/ that prints 'ok' (run it to check), then submit it as a "
    "Submission with entrypoint submission.py, using dataset id 'ds' as the parent."
)


async def configure_code_task(
    cp: ControlPlane,
    *,
    backend: WorkerBackend,
    split: Any = None,  # unused; kept symmetric with configure_fe_task
    provisioning: ProvisioningConfig | None = None,
    runner: CodeRunner | None = None,
) -> str:
    """Configure the trivial ``code`` task onto a generic ``cp`` via its API; returns its id."""
    provisioning = provisioning or ProvisioningConfig()
    spec = provisioning.model_spec
    model = spec.provider_string() if spec is not None else _DEFAULT_MODEL
    runner = runner or BackendCodeRunner(
        backend=backend, image=provisioning.code_image, config=CODE_TASK_ID
    )
    domain = build_code_domain(runner)

    cp.store.propose(
        Artifact("ds", DATASET, {"n": 1}, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation("op-ds", "load", (), "ds", OperationStatus.SUCCESS, "t0"),
    )

    root = Path(tempfile.mkdtemp(prefix="verity-code-"))
    sandbox = build_sandbox(
        schema=domain.schema, root=root,
        driver=BackendSandboxDriver(
            backend=backend, model=model, spec=spec, image=provisioning.sandbox_image,
            config=CODE_TASK_ID, memory=provisioning.sandbox_memory,
            recursion_limit=provisioning.recursion_limit, timeout_s=provisioning.sandbox_timeout_s,
            runtime=provisioning.runtime,
        ),
        proposer_identity=f"deepagents-worker:{model}",
    )
    cp.register_sandbox(CODE_TASK_ID, lambda: sandbox)
    cp.register_verifier(CODE_TASK_ID, lambda: domain.verifier)
    config = TaskConfig(
        task_id=CODE_TASK_ID, instructions=_CODE_INSTRUCTIONS,
        domain_instructions="a Submission names an entrypoint script written to outbox/",
        schema=domain.schema, gated_types=domain.gated_types, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator,
        sandbox_key=CODE_TASK_ID, verifier_key=CODE_TASK_ID,
    )
    await cp.configure(config)
    return CODE_TASK_ID
