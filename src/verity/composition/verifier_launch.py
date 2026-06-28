"""Launch a sibling verifier service for a task (§9.1) — the verifier-name-agnostic seam.

Promoted out of :mod:`verity.composition.fe_kaggle` so every task type that runs against a remote
verifier sibling shares one launch lifecycle, parameterized by the **verifier name**
(``VERITY_VERIFIER``) and the env it forwards (creds + code-runner sizing). The control plane stays
dumb: it launches a container and ships the per-task :class:`VerifierSetup` over HTTP — importing no
gate code (those live in the verifier image, reached only over the wire).

Two seams, picked from the environment by :func:`verifier_factory_from_env`:

* launch a sibling **per task** when ``$VERITY_VERIFIER_NETWORK`` is set (the §9.1 default) — a
  trusted container on that network, reached by its container name via Docker DNS, torn down at
  run-end (run-scoped lifecycle, #96, via :class:`LazyLaunchVerifier`);
* otherwise talk to a **standing** service at ``$VERITY_VERIFIER_URL``.

Offline tests bypass both by injecting a factory directly.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from verity.contracts import GateUnavailable
from verity.contracts.ports import VerifierPort, VerifierSetup
from verity.logging import get_logger
from verity.provisioning.backend import Labels, ResourceLimits, WorkerBackend, WorkerSpec

if TYPE_CHECKING:
    from verity.contracts.model import VerdictBundle
    from verity.contracts.ports import VerifierRequest

log = get_logger("verity.composition.verifier_launch")

__all__ = [
    "VerifierFactory",
    "LazyLaunchVerifier",
    "http_verifier_factory",
    "launch_verifier_factory",
    "verifier_factory_from_env",
    "DEFAULT_VERIFIER_URL",
    "DEFAULT_VERIFIER_IMAGE",
    "DEFAULT_VERIFIER_PORT",
    "DEFAULT_VERIFIER_TIMEOUT_S",
    "DEFAULT_VERIFIER_MEMORY",
    "RUNNER_ENV",
]

DEFAULT_VERIFIER_URL = "http://verifier:8001"
DEFAULT_VERIFIER_IMAGE = "verity-verifier:latest"
DEFAULT_VERIFIER_PORT = 8001
# The CP→verifier dispatch is one blocking HTTP call held open for the WHOLE gate evaluation (the
# code-runner pip-installs an ML stack + trains, minutes per run), so the transport cap must be
# generous; override with VERITY_VERIFIER_TIMEOUT. Sized to the fe-kaggle worst case (~10.8h); a
# holdout cycle is far cheaper but the generous default is harmless.
DEFAULT_VERIFIER_TIMEOUT_S = 38880.0
# The verifier sibling buffers + parses the whole verifier-role dataset in memory, so the cap must
# clear the dataset size; default generously, override with VERITY_VERIFIER_MEMORY.
DEFAULT_VERIFIER_MEMORY = "4g"
# Code-runner sizing knobs the verifier's own runner reads (it pip-installs the FE ML stack into a
# tmpfs); forwarded by NAME so an operator can override the verifier image's defaults from the CP.
RUNNER_ENV = ("VERITY_CODE_IMAGE", "VERITY_CODE_MEMORY", "VERITY_CODE_TMPFS", "VERITY_CODE_CPUS")

# How the control plane obtains the verifier for a task: it builds the per-task `VerifierSetup` and
# hands it to a factory that returns a (typically remote) `VerifierPort`.
VerifierFactory = Callable[[VerifierSetup], Awaitable[VerifierPort]]


def http_verifier_factory(
    *,
    default_url: str = DEFAULT_VERIFIER_URL,
    default_timeout_s: float = DEFAULT_VERIFIER_TIMEOUT_S,
) -> VerifierFactory:
    """A factory talking to a **standing** `RemoteVerifier` over HTTP at ``$VERITY_VERIFIER_URL``.

    The HTTP transport import is **lazy** so this module pulls neither the ``service`` extra nor any
    gate/``kaggle`` code at import time: the gates live in the verifier image, reached on the wire.
    """

    async def make(setup: VerifierSetup) -> VerifierPort:
        from verity.transport.http import build_remote_verifier

        url = os.environ.get("VERITY_VERIFIER_URL", default_url)
        timeout = float(os.environ.get("VERITY_VERIFIER_TIMEOUT", default_timeout_s))
        return build_remote_verifier(url, timeout_s=timeout, setup_payload=setup)

    return make


def verifier_service_spec(
    network: str,
    *,
    verifier_name: str,
    image: str,
    staging: str | None,
    env_passthrough: tuple[str, ...],
    memory: str,
) -> WorkerSpec:
    """The `WorkerSpec` for an on-demand, **trusted** verifier service container (§9.1).

    Joins ``network`` (so the control plane reaches it by container name via Docker DNS), mounts the
    docker socket (to launch its own code-runner children) and — when set — the shared staging dir
    at an identical host path (so those children's bind mounts resolve on the host daemon). The
    empty ``command`` lets the image ``ENTRYPOINT`` (``python -m verity.verifier``) run, selected by
    ``VERITY_VERIFIER=<verifier_name>``. ``env_passthrough`` forwards creds + runner sizing by name.
    """
    name = f"verity-verifier-{uuid.uuid4().hex[:10]}"
    env = {"VERITY_VERIFIER": verifier_name}
    host_mounts: tuple[tuple[str, str, str], ...] = ()
    if staging:
        env["VERITY_WORKER_STAGING"] = staging
        host_mounts = ((staging, staging, "rw"),)
    return WorkerSpec(
        image=image, command=(),
        labels=Labels(role="verifier", config=verifier_name),
        service_name=name, network_name=network, mount_docker_socket=True,
        host_mounts=host_mounts, env=env,
        env_passthrough=env_passthrough,
        limits=ResourceLimits(memory=os.environ.get("VERITY_VERIFIER_MEMORY", memory)),
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


def launch_verifier_factory(
    backend: WorkerBackend,
    *,
    verifier_name: str,
    network: str,
    env_passthrough: tuple[str, ...],
    image: str = DEFAULT_VERIFIER_IMAGE,
    staging: str | None = None,
    port: int = DEFAULT_VERIFIER_PORT,
    memory: str = DEFAULT_VERIFIER_MEMORY,
    health_timeout_s: float = 60.0,
    health_poll_interval_s: float = 1.0,
    verify_timeout_s: float = DEFAULT_VERIFIER_TIMEOUT_S,
) -> VerifierFactory:
    """A `VerifierFactory` that **launches** the verifier sibling on demand (the §9.1 default).

    Per task: launch a trusted verifier container on ``network``, reach it by its container name via
    Docker DNS, wait for it to be healthy, and return a `RemoteVerifier` that ships the setup
    payload over HTTP — and whose ``teardown`` destroys the container.
    """

    async def make(setup: VerifierSetup) -> VerifierPort:
        from verity.transport.http import build_remote_verifier

        spec = verifier_service_spec(
            network, verifier_name=verifier_name, image=image, staging=staging,
            env_passthrough=env_passthrough, memory=memory,
        )
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


def verifier_factory_from_env(
    backend: WorkerBackend,
    *,
    verifier_name: str,
    env_passthrough: tuple[str, ...],
    default_url: str = DEFAULT_VERIFIER_URL,
    default_image: str = DEFAULT_VERIFIER_IMAGE,
) -> VerifierFactory:
    """Pick the verifier seam from the environment: launch a sibling per task when a network is
    configured (``$VERITY_VERIFIER_NETWORK``), else talk to a standing service at
    ``$VERITY_VERIFIER_URL``. Offline tests bypass this by injecting a factory.
    """
    network = os.environ.get("VERITY_VERIFIER_NETWORK")
    if network:
        return launch_verifier_factory(
            backend, verifier_name=verifier_name, network=network,
            env_passthrough=env_passthrough,
            image=os.environ.get("VERITY_VERIFIER_IMAGE", default_image),
            staging=os.environ.get("VERITY_WORKER_STAGING"),
            port=int(os.environ.get("VERITY_VERIFIER_PORT", str(DEFAULT_VERIFIER_PORT))),
            verify_timeout_s=float(
                os.environ.get("VERITY_VERIFIER_TIMEOUT", str(DEFAULT_VERIFIER_TIMEOUT_S))
            ),
        )
    return http_verifier_factory(default_url=default_url)


@dataclass(slots=True)
class LazyLaunchVerifier:
    """A `VerifierPort` that defers launch to ``provision`` and reaps at ``teardown`` (#96).

    The control plane drives provision/teardown on the **run** boundary (``ControlPlane.run``), so a
    verifier sibling container lives only for the duration of an active run and idle verifiers don't
    accumulate across a resident task's runs. This adapter holds the launch ``factory`` + the
    per-task ``setup`` and (re)builds a fresh inner `VerifierPort` each ``provision`` (launch + ship
    setup), tearing it down each ``teardown``. Re-launchable across runs of a resident task.

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
