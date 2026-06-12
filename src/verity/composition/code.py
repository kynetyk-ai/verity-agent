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

from verity.composition.description import OperationDescription, TaskTypeDescription
from verity.composition.fe import ProvisioningConfig, provisioning_config_from
from verity.composition.task_request import TaskRequest
from verity.contracts import Artifact, ArtifactStatus, Operation, OperationStatus
from verity.control_plane.api import ControlPlane
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import DefaultRetrievalPolicy
from verity.domains.code import DATASET, build_code_domain
from verity.provisioning.backend import WorkerBackend
from verity.sandbox.backend_driver import BackendSandboxDriver
from verity.sandbox.registration import build_sandbox
from verity.verifier import BackendCodeRunner, CodeRunner, FakeCodeRunner

__all__ = [
    "configure_code_task",
    "build_code_task",
    "describe_code_task",
    "CODE_TASK_ID",
    "CODE_GOAL",
]

_DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"

CODE_TASK_ID = "code"
CODE_GOAL = "Write a submission script that runs cleanly."
_CODE_INSTRUCTIONS = (
    "Write submission.py to outbox/ that prints 'ok' (run it to check), then submit it as a "
    "Submission with entrypoint submission.py, using dataset id 'ds' as the parent."
)
_CODE_DOMAIN_INSTRUCTIONS = "a Submission names an entrypoint script written to outbox/"
_CODE_VERIFIER_APPROACH = (
    "Two gates on the submitted script: a cheap 'parses' check (rung 2, real ast.parse) earns "
    "'tentative', then a hard 'runs-clean' gate executes the entrypoint in an isolated worker and "
    "accepts on a clean exit. A boolean validity check — no scoring or incumbent comparison."
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
        domain_instructions=_CODE_DOMAIN_INSTRUCTIONS,
        schema=domain.schema, gated_types=domain.gated_types, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator,
        sandbox_key=CODE_TASK_ID, verifier_key=CODE_TASK_ID,
    )
    await cp.configure(config)
    return CODE_TASK_ID


async def build_code_task(cp: ControlPlane, *, backend: WorkerBackend, request: TaskRequest) -> str:
    """The ``code`` catalog builder (ROADMAP 8.1): configure the trivial task from a `TaskRequest`.

    The genericity demonstration on the catalog path — the same multiplexer that runs FE runs this.
    The ``code`` task needs no client data, so ``request.data`` is ignored; only the sandbox
    provisioning selection is read. Returns the task id.
    """
    return await configure_code_task(
        cp, backend=backend, provisioning=provisioning_config_from(request.sandbox)
    )


def describe_code_task() -> TaskTypeDescription:
    """The ``code`` task type's published contract (ADR 0004 (c)). Built from the domain schema; the
    `FakeCodeRunner` is a throwaway just to construct the domain (the gate logic is not invoked)."""
    domain = build_code_domain(FakeCodeRunner())
    schema = domain.schema
    return TaskTypeDescription(
        type_name=CODE_TASK_ID,
        artifact_types=tuple(t.name for t in schema.types()),
        gated_types=tuple(t.name for t in schema.types() if domain.gated_types.is_gated(t.name)),
        operations=tuple(
            OperationDescription(o.name, o.inputs, o.output, o.required_payload_keys)
            for o in schema.operations()
        ),
        domain_instructions=_CODE_DOMAIN_INSTRUCTIONS,
        verifier_approach=_CODE_VERIFIER_APPROACH,
        sandbox_notes=(
            "A general-purpose coding agent that writes and runs Python inside an isolated worker; "
            "needs the container sandbox image."
        ),
    )
