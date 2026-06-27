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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
from verity.logging import get_logger
from verity.provisioning.backend import Labels, ResourceLimits, WorkerBackend, WorkerSpec
from verity.sandbox.backend_driver import BackendSandboxDriver
from verity.sandbox.registration import build_sandbox

if TYPE_CHECKING:
    from verity.composition.catalog import TaskCatalog
    from verity.contracts.model import VerdictBundle
    from verity.contracts.ports import VerifierRequest

log = get_logger("verity.composition.fe_kaggle")

__all__ = [
    "FE_KAGGLE_TASK_ID",
    "FE_KAGGLE_GOAL",
    "VerifierFactory",
    "launch_fe_kaggle_factory",
    "configure_fe_kaggle_task",
    "build_fe_kaggle_task",
    "describe_fe_kaggle_task",
    "register",
]

FE_KAGGLE_TASK_ID = "fe-kaggle"
FE_KAGGLE_GOAL = (
    "Climb the real Kaggle leaderboard for this competition with a fast, self-contained pipeline."
)

# The canonical role-keyed input files the user's prep produces (ADR 0005; see
# ``tools/prepare_fe_data.py``). The control plane validates only their *presence* — it never reads
# their contents. The verifier image maps these filenames onto its gate inputs.
AGENT_INPUT_FILES = ("train.csv", "test.csv")
VERIFIER_INPUT_FILES = (
    "train.csv", "holdout.csv", "holdout_labels.csv", "full_train.csv", "test.csv",
)

_DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"
_DEFAULT_VERIFIER_URL = "http://verifier:8001"
_DEFAULT_VERIFIER_IMAGE = "verity-verifier:latest"
_DEFAULT_VERIFIER_PORT = 8001
# The CP→verifier dispatch is one blocking HTTP call held open for the WHOLE gate evaluation: the
# cheap proxy pip-installs an ML stack + trains (minutes), and the hard gate regenerates + submits
# to Kaggle + polls the public score. The 30s transport default times out well before that, so be
# generous; override with VERITY_VERIFIER_TIMEOUT. (A cap-blocked Kaggle wait can exceed even this —
# that legitimately long synchronous hold wants the async verify-job redesign, tracked separately.)
_DEFAULT_VERIFIER_TIMEOUT_S = 1800.0
# The verifier sibling is shipped the whole verifier-role dataset as its setup payload over HTTP and
# buffers + parses it in memory, so the cap must clear the dataset size (full-data fe-kaggle is
# ~230 MB of CSV → a few hundred MB peak while receiving). The 512 MB worker default OOM-kills it on
# a real run; default generously and let an operator override with VERITY_VERIFIER_MEMORY. (Removing
# the over-the-wire bulk transfer entirely is the networked-data-plane track, tracked separately.)
_DEFAULT_VERIFIER_MEMORY = "4g"
# Creds the verifier service needs, forwarded by NAME (values stay in the daemon env, never the CP
# image or an agent worker): the Kaggle credentials the trusted final-test gate submits with.
_VERIFIER_CRED_ENV = ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR")
# Code-runner sizing knobs the verifier's own runner reads (it pip-installs the FE ML stack into a
# tmpfs); forwarded by NAME so an operator can override the verifier image's defaults from the CP.
_VERIFIER_RUNNER_ENV = (
    "VERITY_CODE_IMAGE", "VERITY_CODE_MEMORY", "VERITY_CODE_TMPFS", "VERITY_CODE_CPUS"
)

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
    timeout = float(os.environ.get("VERITY_VERIFIER_TIMEOUT", _DEFAULT_VERIFIER_TIMEOUT_S))
    return build_remote_verifier(url, timeout_s=timeout, setup_payload=setup)


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
        host_mounts=host_mounts, env=env,
        env_passthrough=_VERIFIER_CRED_ENV + _VERIFIER_RUNNER_ENV,
        limits=ResourceLimits(
            memory=os.environ.get("VERITY_VERIFIER_MEMORY", _DEFAULT_VERIFIER_MEMORY)
        ),
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
    verify_timeout_s: float = _DEFAULT_VERIFIER_TIMEOUT_S,
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
        verifier = build_remote_verifier(
            f"http://{spec.service_name}:{port}", timeout_s=verify_timeout_s, setup_payload=setup
        )
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
            verify_timeout_s=float(
                os.environ.get("VERITY_VERIFIER_TIMEOUT", str(_DEFAULT_VERIFIER_TIMEOUT_S))
            ),
        )
    return _http_verifier


@dataclass(slots=True)
class _LazyLaunchVerifier:
    """A `VerifierPort` that defers launch to ``provision`` and reaps at ``teardown`` (#96).

    The control plane drives provision/teardown on the **run** boundary (``ControlPlane.run``), not
    the task boundary — so a verifier sibling container lives only for the duration of an active run
    and idle verifiers do not accumulate across a resident task's runs. This adapter holds the
    launch ``factory`` + the per-task ``setup`` and (re)builds a fresh inner `VerifierPort` each
    ``provision`` (launch + ship setup), tearing it down (destroying the container via the inner's
    own ``teardown``) each ``teardown``. Re-launchable across runs of a resident task.

    ``dispatch`` delegates to the live inner; calling it before ``provision`` is a control-plane bug
    (the run path always provisions first), surfaced as a recoverable `GateUnavailable`.
    """

    factory: VerifierFactory
    setup: VerifierSetup
    inner: VerifierPort | None = None
    identity: str = ""

    async def provision(self) -> None:
        if self.inner is None:
            self.inner = await self.factory(self.setup)
        await self.inner.provision()
        self.identity = self.inner.identity

    async def dispatch(self, request: VerifierRequest) -> VerdictBundle:
        if self.inner is None:
            raise GateUnavailable(
                "verifier dispatched before provision (run-scoped lifecycle, #96)"
            )
        return await self.inner.dispatch(request)

    async def teardown(self) -> None:
        inner, self.inner, self.identity = self.inner, None, ""
        if inner is not None:
            await inner.teardown()

    async def health(self) -> bool:
        return self.inner is not None and await self.inner.health()


_FE_KAGGLE_INSTRUCTIONS = (
    "Climb into the TOP TIER of the real Kaggle leaderboard. Improve balanced accuracy by any "
    "means that fits in one script — features, model choice, a small ensemble, calibration, "
    "imbalance handling — and TRAIN ON THE FULL data. If there's no prior work yet, explore the "
    "data with code first; otherwise read your prior/just-revised script under scratch/provided/ "
    "and target its weakness. Each cycle your hold-out score is checked against a competitive "
    "target read live from the leaderboard: below it you're told the gap and asked to REVISE (no "
    "submission spent); at or above it the script is regenerated on the full data and submitted to "
    "the REAL leaderboard, accepted only if its public score beats your best. Predict EVERY row of "
    "test.csv. Anything that leaks the target inflates your own number but fails on the held-out "
    "and real data."
)

_FE_KAGGLE_VERIFIER_APPROACH = (
    "A goal-seeking ladder over the regenerated script (the gate re-runs the script on gold data; "
    "the agent's own CSV is never trusted). CHEAP, per-cycle: the script runs on a labeled "
    "hold-out carved from train.csv and is scored on balanced accuracy; a submission that does not "
    "beat the best ACCEPTED proxy by a noise margin is rejected before any Kaggle call. "
    "COMPETITIVE, per-cycle (no budget spent): the live leaderboard is read and the score at the "
    "top-N% rank computed; if our calibrated hold-out estimate is below it the submission is "
    "REFINED (rests revised, the gap fed back) so a real submission is spent only on a competitive "
    "attempt. HARD, rate-limited (the competition's daily submission cap, read from its Kaggle "
    "metadata; overridable): a survivor is regenerated on the full train + real test set and "
    "submitted to the live competition; ACCEPTED only if the public score beats our best prior "
    "Kaggle-confirmed score (the real leaderboard is the incumbent ladder), recording the (proxy, "
    "public) pair to calibrate future estimates. When the daily cap is spent the gate blocks until "
    "it frees; a Kaggle API failure degrades the cycle (recorded), not crash the run."
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
    agent_files: Mapping[str, bytes],
    verifier_files: Mapping[str, bytes],
    make_verifier: VerifierFactory | None = None,
    competition: str = "",
    daily_submission_limit: int | None = None,
    provisioning_mode: ObjectProvisionMode = ObjectProvisionMode.BEST_REVISED_OR_ACCEPTED,
    provisioning: ProvisioningConfig | None = None,
    wait_deadline_s: float = 86_400.0,
    poll_interval_s: float = 60.0,
    score_poll_interval_s: float = 20.0,
    submit_message: str = "verity fe-kaggle",
    target_fraction: float = 0.10,
    proxy_margin: float = 0.0,
    min_calibration_points: int = 2,
    pessimism: float = 0.0,
) -> str:
    """Configure the `fe-kaggle` task onto a generic ``cp`` via its API. Returns the task id.

    The data arrives **pre-prepared and role-keyed** (ADR 0005 / #74): ``agent_files`` is the
    ``{filename: bytes}`` set routed into the agent's sandbox ``data/`` directory, and
    ``verifier_files`` is the set shipped verbatim to the verifier as the :class:`VerifierSetup`
    objects. The control plane splits nothing and reads no ``target``/``id_column`` — the answer key
    is simply one of the verifier-role files, so it structurally cannot reach the agent.

    The verifier runs as a **sibling service** (§9.1 / #73): this hands the per-task
    :class:`VerifierSetup` (the verifier-role blobs + the gate knobs) to ``make_verifier`` (default
    :func:`_http_verifier`, a `RemoteVerifier` over HTTP). The gates, the `kaggle` extra, and the
    Kaggle creds all live in the verifier image; **no** gate code is imported here.
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
            step_budget=provisioning.step_budget, runtime=provisioning.runtime,
        ),
        proposer_identity=f"deepagents-worker:{model}",
        # The agent role's pre-prepared files, routed verbatim into data/. The user's prep removed
        # the hold-out rows from train.csv, so the agent can never see the hold-out labels.
        static_contents={"data": dict(agent_files)},
    )

    # The opaque per-task inputs the (remote) verifier builds its gate stack from: the verifier
    # role's files, routed by filename. The control plane treats these as blobs + knobs — it does
    # not interpret them (verifier opacity, §3.6); the answer key is one of these files.
    setup = VerifierSetup(
        objects=dict(verifier_files),
        params={
            "competition": competition,
            "timeout_s": provisioning.code_timeout_s,
            "wait_deadline_s": wait_deadline_s,
            "poll_interval_s": poll_interval_s,
            "score_poll_interval_s": score_poll_interval_s,
            "submit_message": submit_message,
            "target_fraction": target_fraction,
            "proxy_margin": proxy_margin,
            "min_calibration_points": min_calibration_points,
            "pessimism": pessimism,
            **(
                {"daily_submission_limit": daily_submission_limit}
                if daily_submission_limit is not None
                else {}
            ),
        },
    )
    # Run-scoped lifecycle (#96): wrap the launch factory in a lazy adapter rather than launching
    # the verifier sibling here at configure-time. The control plane provisions it at run-start and
    # tears it down at run-end (api.ControlPlane.run), so an idle verifier never outlives its run.
    verifier = _LazyLaunchVerifier(make_verifier or _verifier_factory_from_env(backend), setup)

    cp.register_sandbox(FE_KAGGLE_TASK_ID, lambda: sandbox)
    cp.register_verifier(FE_KAGGLE_TASK_ID, lambda: verifier)
    config = TaskConfig(
        task_id=FE_KAGGLE_TASK_ID, instructions=_FE_KAGGLE_INSTRUCTIONS,
        domain_instructions=domain.domain_instructions, schema=domain.schema,
        gated_types=domain.gated_types, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator, object_namer=declared_objects,
        sandbox_key=FE_KAGGLE_TASK_ID, verifier_key=FE_KAGGLE_TASK_ID,
        object_provisioning=ObjectProvisioningPolicy(
            mode=provisioning_mode, type_filter=SUBMISSION
        ),
    )
    await cp.configure(config)
    return FE_KAGGLE_TASK_ID


def _require_role_files(role: str, refs: Mapping[str, str], expected: tuple[str, ...]) -> None:
    """Fail fast (misuse) if the user's prep did not supply ``role``'s expected files.

    A presence-only check — the control plane validates that the named blobs exist, never their
    contents (ADR 0005: prep is the user's responsibility, the CP routes opaque blobs).
    """
    missing = [name for name in expected if name not in refs]
    if missing:
        raise ValueError(
            f"the fe-kaggle {role} role is missing prepared file(s) {missing}; "
            f"run tools/prepare_fe_data.py and ingest the {role}/ bundle "
            f"(create the task with files={{'{role}': {{...}}}})"
        )


def _parse_provision_mode(raw: object, default: ObjectProvisionMode) -> ObjectProvisionMode:
    """Parse a ``provisioning_mode`` knob string to a mode; fall back to ``default`` on anything
    unrecognised (degrade-don't-crash — a typo in a knob should not abort task creation)."""
    if raw is None:
        return default
    try:
        return ObjectProvisionMode(str(raw).strip().lower())
    except ValueError:
        log.warning(
            "fe_kaggle_unknown_provisioning_mode",
            value=str(raw),
            using=default.value,
            known=[m.value for m in ObjectProvisionMode],
        )
        return default


async def build_fe_kaggle_task(
    cp: ControlPlane, *, backend: WorkerBackend, request: TaskRequest
) -> str:
    """The `fe-kaggle` catalog builder: configure the task from a declarative `TaskRequest`.

    Consumes **pre-prepared, role-keyed** inputs (ADR 0005 / #74): ``data.roles['agent']`` and
    ``data.roles['verifier']``, each ``{filename: content_hash}`` of files the user prepared outside
    Verity (see ``tools/prepare_fe_data.py``). The control plane fetches the blobs by hash and
    routes them to the two roles **without interpreting them** — no split, no target/id column.
    ``verifier.knobs.competition`` is the competition slug (required); optional knobs tune
    polling/wait, and ``daily_submission_limit`` overrides the per-competition cap otherwise read
    from the competition metadata. The slug + knobs ride to the verifier sibling; the live
    `RealKaggleScorer` is built verifier-side (no `kaggle` import here).
    """
    agent_refs = request.data.files_for("agent")
    verifier_refs = request.data.files_for("verifier")
    _require_role_files("agent", agent_refs, AGENT_INPUT_FILES)
    _require_role_files("verifier", verifier_refs, VERIFIER_INPUT_FILES)
    knobs = request.verifier.knobs
    competition = knobs.get("competition")
    if not competition:
        raise ValueError("fe-kaggle requires verifier.knobs['competition'] (the competition slug)")

    agent_files = {name: cp.store.get_object(ref) for name, ref in agent_refs.items()}
    verifier_files = {name: cp.store.get_object(ref) for name, ref in verifier_refs.items()}

    # The Kaggle scorer + the gates are constructed on the verifier side (#73): we route the
    # competition slug + knobs in the setup payload; no `kaggle` import lives in the control plane.
    limit_knob = knobs.get("daily_submission_limit")
    # ``target_percentile`` is the human-facing knob (top-N%); the gate works in a fraction.
    target_fraction = float(knobs.get("target_percentile", 10.0)) / 100.0
    # Which prior submission(s) to provision into the next cycle's workspace (registries.py). Rides
    # ``verifier.knobs`` for now (the existing knob channel); its principled home is a typed
    # context-assembly/``PolicyRequest`` field, since provisioning is orchestration, not gating.
    provisioning_mode = _parse_provision_mode(
        knobs.get("provisioning_mode"), ObjectProvisionMode.BEST_REVISED_OR_ACCEPTED
    )
    return await configure_fe_kaggle_task(
        cp, backend=backend, agent_files=agent_files, verifier_files=verifier_files,
        competition=str(competition),
        daily_submission_limit=int(limit_knob) if limit_knob is not None else None,
        provisioning_mode=provisioning_mode,
        provisioning=provisioning_config_from(request.sandbox),
        wait_deadline_s=float(knobs.get("wait_deadline_s", 86_400.0)),
        poll_interval_s=float(knobs.get("budget_poll_interval_s", 60.0)),
        score_poll_interval_s=float(knobs.get("score_poll_interval_s", 20.0)),
        submit_message=str(knobs.get("submit_message", "verity fe-kaggle")),
        target_fraction=target_fraction,
        proxy_margin=float(knobs.get("proxy_margin", 0.0)),
        min_calibration_points=int(knobs.get("min_calibration_points", 2)),
        pessimism=float(knobs.get("pessimism", 0.0)),
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


def register(catalog: TaskCatalog) -> None:
    """Register the built-in ``fe-kaggle`` task type (a ``verity.task_types`` entry point)."""
    catalog.register(FE_KAGGLE_TASK_ID, build_fe_kaggle_task, describe=describe_fe_kaggle_task)
