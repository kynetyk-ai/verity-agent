"""Apply the FE-holdout task to a generic control plane (the `fe-holdout` task type).

The ablation-ladder task (Exp 1–4b, ``docs/experimental-design.md`` §1/§6): the same §12
feature-engineering domain (schema / shape / instructions / object-naming) and worker provisioning
as `fe-kaggle`, but scored **locally on a reserved hold-out** — no Kaggle, no competitive bar, no
calibration. The verifier runs as the same kind of **sibling service** (§9.1), here selected by
``VERITY_VERIFIER=holdout-experiment`` and shipped a 3-file verifier role + one experimental knob.

The two experimental axes that define the ladder are **pure configuration** over this one task type:

* ``verifier.knobs.accept_policy`` — ``improve_over_best_prior`` (Exp 3/4/4b) or ``always`` (Exp
  1/2); the only verdict difference across conditions (the new gate logic lives in the §12 scorer).
* ``policy.provisioning`` — the object-provisioning preset (``none`` / ``all`` /
  ``all_revised_or_accepted`` / ``best_revised_or_accepted``), the context-hygiene axis (#95).

Loop-vs-one-shot (``policy.max_cycles`` / ``stop_on_accept``), the model, and budgets are likewise
config. So the whole ladder is this task type configured five ways — no per-condition code.

The data arrives **pre-prepared and role-keyed** (ADR 0005 / #74): the agent role is ``train.csv`` +
``test.csv`` (hold-out removed from train; identical to fe-kaggle's agent role), and the verifier
role is ``train.csv`` + ``holdout.csv`` + ``holdout_labels.csv`` — a **subset** of fe-kaggle's
verifier role, so ``tools/prepare_fe_data.py`` produces it unchanged. The answer key
(``holdout_labels.csv``) is a verifier-role file and structurally cannot reach the agent.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from verity.composition.description import OperationDescription, TaskTypeDescription
from verity.composition.fe import (
    ProvisioningConfig,
    assert_answer_key_isolated,
    provisioning_config_from,
    with_fe_sandbox_image,
)
from verity.composition.task_request import TaskRequest
from verity.composition.verifier_launch import (
    RUNNER_ENV,
    LazyLaunchVerifier,
    VerifierFactory,
    verifier_factory_from_env,
)
from verity.contracts import Artifact, ArtifactStatus, Operation, OperationStatus
from verity.contracts.ports import VerifierSetup
from verity.control_plane.api import ControlPlane
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import (
    DefaultRetrievalPolicy,
    ObjectProvisionMode,
    provisioning_policy,
)
from verity.domains.feature_engineering import (
    ACCEPT_IMPROVE_OVER_BEST_PRIOR,
    DATASET_VERSION,
    SUBMISSION,
    build_feature_engineering_domain,
    declared_objects,
)
from verity.logging import get_logger
from verity.provisioning.backend import WorkerBackend
from verity.sandbox.backend_driver import BackendSandboxDriver
from verity.sandbox.registration import build_sandbox

if TYPE_CHECKING:
    from verity.composition.catalog import TaskCatalog

log = get_logger("verity.composition.fe_holdout")

__all__ = [
    "FE_HOLDOUT_TASK_ID",
    "FE_HOLDOUT_GOAL",
    "configure_fe_holdout_task",
    "build_fe_holdout_task",
    "describe_fe_holdout_task",
    "register",
]

FE_HOLDOUT_TASK_ID = "fe-holdout"
# The verifier-type name this task selects (``VERITY_VERIFIER``); mirrors the verifier-side
# ``holdout_service.HOLDOUT_CONFIG`` without the control plane importing the verifier package.
_HOLDOUT_VERIFIER = "holdout-experiment"
FE_HOLDOUT_GOAL = (
    "Optimize stellar-class prediction for balanced accuracy on a reserved hold-out, improving "
    "across cycles."
)

# The canonical role-keyed input files (ADR 0005; see ``tools/prepare_fe_data.py``). The agent role
# matches fe-kaggle's; the verifier role is a subset (no full_train.csv / real test.csv — there is
# no Kaggle submission). The control plane validates only presence, never contents.
AGENT_INPUT_FILES = ("train.csv", "test.csv")
VERIFIER_INPUT_FILES = ("train.csv", "holdout.csv", "holdout_labels.csv")

_DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"
# No Kaggle creds for this verifier; only the shared code-runner sizing env is forwarded by name.
_FE_HOLDOUT_ENV_PASSTHROUGH = RUNNER_ENV

_FE_HOLDOUT_INSTRUCTIONS = (
    "Optimize stellar-class prediction for balanced accuracy. Each cycle, improve upon any prior "
    "submission. The mechanics — the script contract and the held-out check — are in the domain "
    "instructions above; execute them well rather than restating them."
)

_FE_HOLDOUT_VERIFIER_APPROACH = (
    "A local hold-out scorer over the regenerated script (the gate re-runs the script on gold "
    "data; the agent's own CSV is never trusted): the script trains on the agent's train.csv and "
    "predicts a reserved hold-out carved from it, scored on BALANCED ACCURACY against held labels "
    "the agent never sees. The verdict policy is the one experimental knob (accept_policy): "
    "'improve_over_best_prior' ACCEPTS iff the score beats the best prior else REJECTS (binary, "
    "never refine); 'always' records the score and ACCEPTS unconditionally. No Kaggle, no "
    "competitive bar, no calibration — a controlled, deterministic, local signal for the ablation."
)
_FE_HOLDOUT_SANDBOX_NOTES = (
    "A general-purpose coding agent that writes and runs Python in an isolated worker; needs the "
    "container sandbox image. The script is regenerated and scored on the trusted verifier side — "
    "the agent never sees the hold-out labels."
)


async def configure_fe_holdout_task(
    cp: ControlPlane,
    *,
    backend: WorkerBackend,
    agent_files: Mapping[str, bytes],
    verifier_files: Mapping[str, bytes],
    make_verifier: VerifierFactory | None = None,
    accept_policy: str = ACCEPT_IMPROVE_OVER_BEST_PRIOR,
    margin: float = 0.0,
    provisioning_spec: object = ObjectProvisionMode.ALL,
    provisioning: ProvisioningConfig | None = None,
) -> str:
    """Configure the `fe-holdout` task onto a generic ``cp`` via its API. Returns the task id.

    Reuses the §12 domain and the fe-kaggle worker/verifier wiring; the verifier is the
    ``holdout-experiment`` sibling (§9.1), shipped the 3-file verifier role + ``{accept_policy,
    margin, timeout_s}`` as its :class:`VerifierSetup`. No gate code or Kaggle client is imported.
    """
    # Defense-in-depth: refuse to wire the task if the answer key leaked into the agent role (I1).
    assert_answer_key_isolated(agent_files, verifier_files)
    provisioning = provisioning or ProvisioningConfig()
    spec = provisioning.model_spec
    model = spec.provider_string() if spec is not None else _DEFAULT_MODEL
    domain = build_feature_engineering_domain()

    cp.store.propose(
        Artifact("ds", DATASET_VERSION, {"source": "holdout"}, ArtifactStatus.PROPOSED,
                 "loader", "t0", is_root=True),
        Operation("op-ds", "load", (), "ds", OperationStatus.SUCCESS, "t0"),
    )

    root = Path(tempfile.mkdtemp(prefix="verity-fe-holdout-"))
    sandbox = build_sandbox(
        schema=domain.schema, root=root,
        driver=BackendSandboxDriver(
            backend=backend, model=model, spec=spec, image=provisioning.sandbox_image,
            config=FE_HOLDOUT_TASK_ID, memory=provisioning.sandbox_memory,
            recursion_limit=provisioning.recursion_limit, timeout_s=provisioning.sandbox_timeout_s,
            step_budget=provisioning.step_budget, runtime=provisioning.runtime,
        ),
        proposer_identity=f"deepagents-worker:{model}",
        static_contents={"data": dict(agent_files)},
    )

    # The opaque per-task inputs the (remote) verifier builds its scorer from: the verifier role's
    # files routed by filename + the experimental knobs. The answer key is one of these files.
    setup = VerifierSetup(
        objects=dict(verifier_files),
        params={
            "accept_policy": accept_policy,
            "margin": margin,
            "timeout_s": provisioning.code_timeout_s,
        },
    )
    factory = make_verifier or verifier_factory_from_env(
        backend, verifier_name=_HOLDOUT_VERIFIER, env_passthrough=_FE_HOLDOUT_ENV_PASSTHROUGH
    )
    verifier = LazyLaunchVerifier(factory, setup)

    cp.register_sandbox(FE_HOLDOUT_TASK_ID, lambda: sandbox)
    cp.register_verifier(FE_HOLDOUT_TASK_ID, lambda: verifier)
    config = TaskConfig(
        task_id=FE_HOLDOUT_TASK_ID, instructions=_FE_HOLDOUT_INSTRUCTIONS,
        domain_instructions=domain.domain_instructions, schema=domain.schema,
        gated_types=domain.gated_types, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator, object_namer=declared_objects,
        sandbox_key=FE_HOLDOUT_TASK_ID, verifier_key=FE_HOLDOUT_TASK_ID,
        object_provisioning=provisioning_policy(provisioning_spec, type_filter=SUBMISSION),
    )
    await cp.configure(config)
    return FE_HOLDOUT_TASK_ID


def _require_role_files(role: str, refs: Mapping[str, str], expected: tuple[str, ...]) -> None:
    """Fail fast (misuse) if the user's prep did not supply ``role``'s expected files."""
    missing = [name for name in expected if name not in refs]
    if missing:
        raise ValueError(
            f"the fe-holdout {role} role is missing prepared file(s) {missing}; "
            f"run tools/prepare_fe_data.py and ingest the {role}/ bundle "
            f"(create the task with files={{'{role}': {{...}}}})"
        )


async def build_fe_holdout_task(
    cp: ControlPlane, *, backend: WorkerBackend, request: TaskRequest
) -> str:
    """The `fe-holdout` catalog builder: configure the task from a declarative `TaskRequest`.

    Consumes **pre-prepared, role-keyed** inputs (ADR 0005 / #74): ``data.roles['agent']`` and
    ``data.roles['verifier']``. The verdict policy and provisioning preset — the two ablation axes —
    come from ``verifier.knobs.accept_policy`` and ``policy.provisioning``.
    """
    agent_refs = request.data.files_for("agent")
    verifier_refs = request.data.files_for("verifier")
    _require_role_files("agent", agent_refs, AGENT_INPUT_FILES)
    _require_role_files("verifier", verifier_refs, VERIFIER_INPUT_FILES)

    knobs = request.verifier.knobs
    agent_files = {name: cp.store.get_object(ref) for name, ref in agent_refs.items()}
    verifier_files = {name: cp.store.get_object(ref) for name, ref in verifier_refs.items()}
    provisioning_spec = request.policy.provisioning or ObjectProvisionMode.ALL

    return await configure_fe_holdout_task(
        cp, backend=backend, agent_files=agent_files, verifier_files=verifier_files,
        accept_policy=str(knobs.get("accept_policy", ACCEPT_IMPROVE_OVER_BEST_PRIOR)),
        margin=float(knobs.get("margin", 0.0)),
        provisioning_spec=provisioning_spec,
        provisioning=with_fe_sandbox_image(provisioning_config_from(request.sandbox)),
    )


def describe_fe_holdout_task() -> TaskTypeDescription:
    """The `fe-holdout` task type's published contract (ADR 0004 (c)) for ``verity catalog``."""
    domain = build_feature_engineering_domain()
    schema = domain.schema
    return TaskTypeDescription(
        type_name=FE_HOLDOUT_TASK_ID,
        artifact_types=tuple(t.name for t in schema.types()),
        gated_types=tuple(t.name for t in schema.types() if domain.gated_types.is_gated(t.name)),
        operations=tuple(
            OperationDescription(o.name, o.inputs, o.output, o.required_payload_keys)
            for o in schema.operations()
        ),
        domain_instructions=domain.domain_instructions,
        verifier_approach=_FE_HOLDOUT_VERIFIER_APPROACH,
        sandbox_notes=_FE_HOLDOUT_SANDBOX_NOTES,
    )


def register(catalog: TaskCatalog) -> None:
    """Register the built-in ``fe-holdout`` task type (a ``verity.task_types`` entry point)."""
    catalog.register(FE_HOLDOUT_TASK_ID, build_fe_holdout_task, describe=describe_fe_holdout_task)
