"""Apply the FE-Kaggle task to a generic control plane (the `fe-kaggle` task type).

Reuses the §12 feature-engineering domain (schema / shape / instructions) and the
worker-provisioning wiring of :mod:`verity.composition.fe`. Since §9.1 the verifier runs as a
**sibling service** (its own image, carrying the gates + the ``kaggle`` extra): this module builds
the per-task data split and ships it — plus the gate knobs — to that verifier as a
:class:`VerifierSetup` (default: a `RemoteVerifier`, launched on demand; see
:func:`launch_fe_kaggle_factory`). **No** gate code or Kaggle client is imported here.

The split the control plane prepares (the 9.2/#74 residual — it still routes the outputs to both
roles): the labeled hold-out of ``train.csv`` (cheap proxy) + the full labeled subsample and the
**real, unlabeled ``test.csv``** (the Kaggle run) go to the verifier; the agent's sandbox sees only
``agent_train`` (hold-out removed) + the real ``test.csv`` — never the reserved labels nor the real
test's labels. The Kaggle creds live on the verifier side, forwarded by name, never in an agent
worker. ``describe_fe_kaggle_task`` publishes the contract for ``verity catalog``.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from verity.composition.dataset import stratified_split, subsample
from verity.composition.description import OperationDescription, TaskTypeDescription
from verity.composition.fe import ProvisioningConfig, provisioning_config_from
from verity.composition.task_request import TaskRequest
from verity.contracts import Artifact, ArtifactStatus, GateUnavailable, Operation, OperationStatus
from verity.contracts.ports import VerifierPort, VerifierSetup
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
    declared_objects,
)
from verity.provisioning.backend import Labels, WorkerBackend, WorkerSpec
from verity.sandbox.backend_driver import BackendSandboxDriver
from verity.sandbox.registration import build_sandbox

__all__ = [
    "FE_KAGGLE_TASK_ID",
    "FE_KAGGLE_GOAL",
    "VerifierFactory",
    "launch_fe_kaggle_factory",
    "configure_fe_kaggle_task",
    "build_fe_kaggle_task",
    "describe_fe_kaggle_task",
]

FE_KAGGLE_TASK_ID = "fe-kaggle"
FE_KAGGLE_GOAL = (
    "Climb the real Kaggle leaderboard for this competition with a fast, self-contained pipeline."
)

_DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"
_DEFAULT_VERIFIER_URL = "http://verifier:8001"
_DEFAULT_VERIFIER_IMAGE = "verity-verifier:latest"
_DEFAULT_VERIFIER_PORT = 8001
# Creds the verifier service needs, forwarded by NAME (values stay in the daemon env, never the CP
# image or an agent worker): the Kaggle credentials the trusted final-test gate submits with.
_VERIFIER_CRED_ENV = ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR")

# How the control plane obtains the verifier for a task: it builds the per-task `VerifierSetup` and
# hands it to a factory that returns a (typically remote) `VerifierPort`. The default ships the
# setup to a sibling verifier service over HTTP (§9.1); offline tests inject a loopback factory.
VerifierFactory = Callable[[VerifierSetup], Awaitable[VerifierPort]]


async def _http_verifier(setup: VerifierSetup) -> VerifierPort:
    """Static seam: a `RemoteVerifier` over HTTP at ``$VERITY_VERIFIER_URL`` (a standing service).

    The HTTP transport import is **lazy** so this module pulls neither the ``service`` extra nor —
    load-bearing for §9.1 — any gate/``kaggle`` code at import time: the gates live in the verifier
    image, reached only over the wire. The per-task data + knobs ride in the setup payload.
    """
    from verity.transport.http import build_remote_verifier

    url = os.environ.get("VERITY_VERIFIER_URL", _DEFAULT_VERIFIER_URL)
    return build_remote_verifier(url, setup_payload=setup)


def _verifier_service_spec(network: str, *, image: str, staging: str | None) -> WorkerSpec:
    """The `WorkerSpec` for an on-demand, **trusted** verifier service container (§9.1).

    Joins ``network`` (so the control plane reaches it by its container name via Docker DNS), mounts
    the docker socket (to launch its own code-runner children) and — when set — the shared staging
    dir at an identical host path (so those children's bind mounts resolve on the host daemon, the
    same sibling-mount trick the control plane uses). The Kaggle creds are forwarded by name; the
    verifier builds its gate stack from the setup payload it is then sent. ``command`` is empty so
    the image ``ENTRYPOINT`` (``python -m verity.verifier``) runs, selected by
    ``VERITY_VERIFIER=fe-kaggle``."""
    name = f"verity-verifier-{uuid.uuid4().hex[:10]}"
    env = {"VERITY_VERIFIER": FE_KAGGLE_TASK_ID}
    host_mounts: tuple[tuple[str, str, str], ...] = ()
    if staging:
        env["VERITY_WORKER_STAGING"] = staging
        host_mounts = ((staging, staging, "rw"),)
    return WorkerSpec(
        image=image, command=(),
        labels=Labels(role="verifier", config=FE_KAGGLE_TASK_ID),
        service_name=name, network_name=network, mount_docker_socket=True,
        host_mounts=host_mounts, env=env, env_passthrough=_VERIFIER_CRED_ENV,
    )


async def _await_verifier_ready(
    verifier: VerifierPort, *, timeout_s: float, poll_interval_s: float
) -> None:
    """Poll the freshly-launched service's ``health`` until it answers, or give up (recoverable)."""
    from verity.transport.base import TransportUnavailable

    waited = 0.0
    while True:
        try:
            if await verifier.health():
                return
        except TransportUnavailable:
            pass  # the container is still booting uvicorn — keep polling
        if waited >= timeout_s:
            raise GateUnavailable(f"verifier service did not become healthy within {timeout_s}s")
        await asyncio.sleep(poll_interval_s)
        waited += poll_interval_s


def launch_fe_kaggle_factory(
    backend: WorkerBackend,
    *,
    network: str,
    image: str = _DEFAULT_VERIFIER_IMAGE,
    staging: str | None = None,
    port: int = _DEFAULT_VERIFIER_PORT,
    health_timeout_s: float = 60.0,
    health_poll_interval_s: float = 1.0,
) -> VerifierFactory:
    """A `VerifierFactory` that **launches** the verifier sibling on demand (the §9.1 default).

    Per task: launch a trusted verifier container on ``network``, reach it by its container name via
    Docker DNS (no address inspection), wait for it to be healthy, and return a `RemoteVerifier`
    that ships the setup payload over HTTP — and whose ``teardown`` destroys the container. This is
    the *minimal* on-demand lifecycle (the right pattern for many verifiers, no idle containers);
    the full fleet lifecycle (concurrency, address inspection, reconciliation) is tracked apart.
    """

    async def make(setup: VerifierSetup) -> VerifierPort:
        from verity.transport.http import build_remote_verifier

        spec = _verifier_service_spec(network, image=image, staging=staging)
        handle = await backend.launch(spec)
        verifier = build_remote_verifier(f"http://{spec.service_name}:{port}", setup_payload=setup)
        verifier.on_teardown = lambda: backend.destroy(handle)
        await _await_verifier_ready(
            verifier, timeout_s=health_timeout_s, poll_interval_s=health_poll_interval_s
        )
        return verifier

    return make


def _verifier_factory_from_env(backend: WorkerBackend) -> VerifierFactory:
    """Pick the verifier seam from the environment: launch a sibling per task when a network is
    configured (``$VERITY_VERIFIER_NETWORK``), else talk to a standing service at
    ``$VERITY_VERIFIER_URL`` (the default). Offline tests bypass this by injecting a factory.
    """
    network = os.environ.get("VERITY_VERIFIER_NETWORK")
    if network:
        return launch_fe_kaggle_factory(
            backend, network=network,
            image=os.environ.get("VERITY_VERIFIER_IMAGE", _DEFAULT_VERIFIER_IMAGE),
            staging=os.environ.get("VERITY_WORKER_STAGING"),
            port=int(os.environ.get("VERITY_VERIFIER_PORT", str(_DEFAULT_VERIFIER_PORT))),
        )
    return _http_verifier

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
    "best ACCEPTED proxy is rejected before any Kaggle call. HARD, rate-limited (the competition's "
    "daily submission cap, read from its Kaggle metadata; overridable): the "
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
    split: Any,
    full_train_csv: bytes,
    real_test_csv: bytes,
    make_verifier: VerifierFactory | None = None,
    competition: str = "",
    daily_submission_limit: int | None = None,
    provisioning: ProvisioningConfig | None = None,
    wait_deadline_s: float = 86_400.0,
    poll_interval_s: float = 60.0,
    score_poll_interval_s: float = 20.0,
    submit_message: str = "verity fe-kaggle",
) -> str:
    """Configure the `fe-kaggle` task onto a generic ``cp`` via its API. Returns the task id.

    The verifier runs as a **sibling service** (§9.1 / #73): this assembles the per-task
    :class:`VerifierSetup` — the split CSVs the gates run on plus the gate knobs — and hands it to
    ``make_verifier`` (default :func:`_http_verifier`, a `RemoteVerifier` over HTTP). The gates,
    the `kaggle` extra, and the Kaggle creds all live in the verifier image; **no** gate code is
    imported here. ``split`` is the labeled hold-out split (the cheap proxy: ``agent_train_csv`` /
    ``reserved_test_csv`` / ``reserved_labels``); ``full_train_csv`` is the full labeled subsample
    and ``real_test_csv`` the competition's real (unlabeled) test set (the Kaggle run).
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

    # The opaque per-task inputs the (remote) verifier builds its gate stack from. The control plane
    # treats these as routed blobs + knobs — it does not interpret them (verifier opacity, §3.6).
    setup = VerifierSetup(
        objects={
            "agent_train": split.agent_train_csv,
            "reserved_test": split.reserved_test_csv,
            "full_train": full_train_csv,
            "real_test": real_test_csv,
        },
        params={
            "competition": competition,
            "reserved_labels": dict(split.reserved_labels),
            "timeout_s": provisioning.code_timeout_s,
            "wait_deadline_s": wait_deadline_s,
            "poll_interval_s": poll_interval_s,
            "score_poll_interval_s": score_poll_interval_s,
            "submit_message": submit_message,
            **(
                {"daily_submission_limit": daily_submission_limit}
                if daily_submission_limit is not None
                else {}
            ),
        },
    )
    verifier = await (make_verifier or _verifier_factory_from_env(backend))(setup)

    cp.register_sandbox(FE_KAGGLE_TASK_ID, lambda: sandbox)
    cp.register_verifier(FE_KAGGLE_TASK_ID, lambda: verifier)
    config = TaskConfig(
        task_id=FE_KAGGLE_TASK_ID, instructions=_FE_KAGGLE_INSTRUCTIONS,
        domain_instructions=domain.domain_instructions, schema=domain.schema,
        gated_types=domain.gated_types, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator, object_namer=declared_objects,
        sandbox_key=FE_KAGGLE_TASK_ID, verifier_key=FE_KAGGLE_TASK_ID,
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

    Needs two ingested inputs: ``data.data_ref`` (the labeled ``train.csv``, subsampled and split
    here into the agent's train + the reserved hold-out for the cheap proxy) and
    ``data.test_ref`` (the real, unlabeled ``test.csv`` for Kaggle). ``verifier.knobs.competition``
    is the competition slug (required); optional knobs tune polling/wait, and
    ``daily_submission_limit`` overrides the per-competition cap that is otherwise read from the
    competition metadata. Routes the slug + knobs to `configure_fe_kaggle_task`, which ships them to
    the verifier sibling; the live `RealKaggleScorer` is built verifier-side (no `kaggle` import).
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

    # The Kaggle scorer + the gates are constructed on the verifier side (#73): we route the
    # competition slug + knobs in the setup payload; no `kaggle` import lives in the control plane.
    limit_knob = knobs.get("daily_submission_limit")
    return await configure_fe_kaggle_task(
        cp, backend=backend, split=split,
        full_train_csv=sub, real_test_csv=real_test,
        competition=str(competition),
        daily_submission_limit=int(limit_knob) if limit_knob is not None else None,
        provisioning=provisioning_config_from(request.sandbox),
        wait_deadline_s=float(knobs.get("wait_deadline_s", 86_400.0)),
        poll_interval_s=float(knobs.get("budget_poll_interval_s", 60.0)),
        score_poll_interval_s=float(knobs.get("score_poll_interval_s", 20.0)),
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
