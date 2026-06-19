"""On-demand verifier launch + lifecycle (ROADMAP Phase 9.1 / #73). Offline.

The control plane launches the verifier as a *trusted sibling service* per task and reaches it by
Docker DNS (`launch_fe_kaggle_factory`); the real container bring-up + health handshake is the
manual ``@live`` step, but the spec it launches, the env-driven seam selection, and the
launched-container teardown are all pinned offline here (a `FakeBackend` records the spec; the
readiness poll is short-circuited to assert the recoverable-timeout path).
"""

from __future__ import annotations

import asyncio

import pytest

from verity.composition.fe_kaggle import (
    _http_verifier,
    _verifier_factory_from_env,
    launch_fe_kaggle_factory,
)
from verity.contracts import GateUnavailable
from verity.contracts.ports import VerifierSetup
from verity.provisioning import FakeBackend

pytest.importorskip("fastapi")  # the factory builds a RemoteVerifier over the HTTP transport
pytest.importorskip("httpx")

_SETUP = VerifierSetup(objects={"agent_train": b"x"}, params={"competition": "demo"})


def test_launch_factory_launches_a_trusted_verifier_service_spec() -> None:
    backend = FakeBackend()
    # health_timeout_s=0 → after one (failing) readiness poll the factory gives up with a
    # recoverable GateUnavailable; we don't need a real container to assert *what was launched*.
    make = launch_fe_kaggle_factory(
        backend, network="verity-net", staging="/srv/staging", health_timeout_s=0.0,
        health_poll_interval_s=0.0,
    )
    async def go() -> None:
        await make(_SETUP)

    with pytest.raises(GateUnavailable, match="did not become healthy"):
        asyncio.run(go())

    assert len(backend.launched) == 1
    spec = backend.launched[0]
    assert spec.labels.role == "verifier" and spec.labels.config == "fe-kaggle"
    assert spec.service_name and spec.service_name.startswith("verity-verifier-")
    assert spec.network_name == "verity-net" and spec.mount_docker_socket is True
    assert spec.env["VERITY_VERIFIER"] == "fe-kaggle"
    assert spec.env["VERITY_WORKER_STAGING"] == "/srv/staging"
    assert ("/srv/staging", "/srv/staging", "rw") in spec.host_mounts


def test_env_seam_launches_when_a_network_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERITY_VERIFIER_NETWORK", "verity-net")
    factory = _verifier_factory_from_env(FakeBackend())
    assert factory is not _http_verifier  # a launch closure, not the static-URL seam


def test_env_seam_uses_a_standing_service_when_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERITY_VERIFIER_NETWORK", raising=False)
    assert _verifier_factory_from_env(FakeBackend()) is _http_verifier


def test_teardown_destroys_the_launched_container() -> None:
    # A RemoteVerifier the control plane launched destroys its container at teardown — the standing
    # daemon's verifier.teardown() reaps the on-demand sibling with no control-plane change.
    from verity.transport import LoopbackTransport, RemoteVerifier, VerifierServer

    class _V:
        identity = "fe-kaggle"

        async def dispatch(self, request: object) -> object: ...  # unused here
        async def provision(self) -> None: ...
        async def teardown(self) -> None: ...
        async def health(self) -> bool:
            return True

    destroyed: list[bool] = []

    async def _destroy() -> None:
        destroyed.append(True)

    remote = RemoteVerifier(LoopbackTransport(VerifierServer(_V()).handle))  # type: ignore[arg-type]
    remote.on_teardown = _destroy
    asyncio.run(remote.teardown())
    assert destroyed == [True]  # the container was reaped after the wire teardown
