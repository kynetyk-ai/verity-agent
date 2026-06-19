"""The HTTP transport over a real FastAPI app (ROADMAP Phase 7.2). Offline — httpx drives the ASGI
app in-process (no socket). Auto-skips without the ``service`` extra.

Proves the same seam the loopback test proves, but through the real FastAPI route: dispatch
round-trips, the advisory status codes match the envelope (200 / 503 / 422), and an unreachable peer
(a connection error) maps to ``GateUnavailable`` so the run still degrades rather than crashes.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx  # noqa: E402

from verity.contracts import (  # noqa: E402
    Artifact,
    ArtifactStatus,
    GateDecision,
    GateUnavailable,
    VerdictBundle,
    VerdictKind,
    VerifierRequest,
    VerifierSetup,
)
from verity.contracts.wire import verifier_request_to_dict  # noqa: E402
from verity.transport.client import RemoteVerifier  # noqa: E402
from verity.transport.http import (  # noqa: E402
    HttpTransport,
    build_asgi_app,
    status_for_envelope,
)
from verity.transport.server import VerifierServer  # noqa: E402
from verity.verifier.errors import VerifierError  # noqa: E402

_ACCEPT = VerdictBundle(
    ArtifactStatus.ACCEPTED, (GateDecision("g", VerdictKind.ACCEPT, "ok", score=0.5),)
)
_ARTIFACT = Artifact("a1", "Note", {"text": "hi"}, ArtifactStatus.PROPOSED, "agent", "t1")
_REQUEST = VerifierRequest(proposal=_ARTIFACT)


class _Verifier:
    """An in-test VerifierPort that returns a fixed bundle or raises."""

    identity = "svc-verifier"

    def __init__(self, *, bundle: VerdictBundle = _ACCEPT, raises: Exception | None = None) -> None:
        self.bundle = bundle
        self.raises = raises

    async def dispatch(self, request: VerifierRequest) -> VerdictBundle:
        if self.raises is not None:
            raise self.raises
        return self.bundle

    async def provision(self) -> None: ...
    async def teardown(self) -> None: ...
    async def health(self) -> bool:
        return True


def _remote(verifier: object) -> tuple[RemoteVerifier, httpx.AsyncClient]:
    app = build_asgi_app(VerifierServer(verifier))  # type: ignore[arg-type]
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://verifier")
    return RemoteVerifier(HttpTransport(base_url="http://verifier", client=client)), client


def test_http_dispatch_roundtrips_through_fastapi() -> None:
    remote, client = _remote(_Verifier())

    async def go() -> VerdictBundle:
        try:
            await remote.provision()
            return await remote.dispatch(_REQUEST)
        finally:
            await client.aclose()

    assert asyncio.run(go()) == _ACCEPT


def test_http_setup_builds_a_data_bearing_verifier() -> None:
    # A builder-backed server over real HTTP: the per-task VerifierSetup is shipped at provision and
    # the impl is built server-side from it (§9.1), then dispatch uses it.
    built: list[VerifierSetup] = []

    def builder(setup: VerifierSetup) -> _Verifier:
        built.append(setup)
        return _Verifier()

    app = build_asgi_app(VerifierServer(builder=builder))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://verifier")
    setup = VerifierSetup(objects={"train.csv": b"x,y\n1,2\n"}, params={"competition": "demo"})
    remote = RemoteVerifier(
        HttpTransport(base_url="http://verifier", client=client), setup_payload=setup
    )

    async def go() -> VerdictBundle:
        try:
            await remote.provision()
            return await remote.dispatch(_REQUEST)
        finally:
            await client.aclose()

    assert asyncio.run(go()) == _ACCEPT
    assert built[0].objects["train.csv"] == b"x,y\n1,2\n"  # data survived the real HTTP hop
    assert remote.identity == "svc-verifier"


def test_http_provision_resolves_identity() -> None:
    remote, client = _remote(_Verifier())

    async def go() -> str:
        try:
            await remote.provision()
            return remote.identity
        finally:
            await client.aclose()

    assert asyncio.run(go()) == "svc-verifier"


def test_recoverable_error_is_503_and_reraised() -> None:
    _unused, client = _remote(_Verifier(raises=GateUnavailable("backend down")))
    body = json.dumps(verifier_request_to_dict(_REQUEST)).encode()

    async def go() -> int:
        try:
            resp = await client.post("http://verifier/dispatch", content=body)
            return resp.status_code
        finally:
            await client.aclose()

    assert asyncio.run(go()) == 503  # advisory status for a recoverable error

    remote2, client2 = _remote(_Verifier(raises=GateUnavailable("backend down")))

    async def dispatch() -> None:
        try:
            await remote2.dispatch(_REQUEST)
        finally:
            await client2.aclose()

    with pytest.raises(GateUnavailable, match="backend down"):  # the error survived the HTTP hop
        asyncio.run(dispatch())


def test_status_for_envelope_maps_kinds() -> None:
    from verity.transport.envelope import err_envelope, ok_envelope

    assert status_for_envelope(ok_envelope(None)) == 200
    assert status_for_envelope(err_envelope(GateUnavailable("x"))) == 503
    assert status_for_envelope(err_envelope(VerifierError("x"))) == 422
    assert status_for_envelope(b"not json at all") == 422


def test_connection_failure_maps_to_gate_unavailable() -> None:
    class _BoomClient:
        async def post(self, *args: object, **kwargs: object) -> object:
            raise httpx.ConnectError("connection refused")

        async def aclose(self) -> None: ...

    remote = RemoteVerifier(HttpTransport(base_url="http://down", client=_BoomClient()))  # type: ignore[arg-type]
    with pytest.raises(GateUnavailable, match="verifier unreachable"):
        asyncio.run(remote.dispatch(_REQUEST))
