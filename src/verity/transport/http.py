"""The HTTP transport (ROADMAP Phase 7.2): the same seam with a socket between.

:class:`HttpTransport` POSTs a method's bytes to ``{base_url}/{method}`` and returns the reply's
bytes; :func:`build_asgi_app` wraps a ``VerifierServer`` / ``SandboxServer`` in a FastAPI app whose
single route hands the body to ``server.handle``. The envelope still carries the truth (the client
re-raises the boundary error from the body), so the HTTP status code is **advisory**: success → 200,
a recoverable error (``GateUnavailable`` / ``SandboxError``) → 503, fatal misuse → 422, so a bare
proxy that only sees the status still conveys the degrade-or-fail distinction. A connection that
never reaches the peer becomes :class:`~verity.transport.base.TransportUnavailable`, which the
client maps to its own recoverable error.

Imports FastAPI / httpx — installed via the ``service`` extra. The package ``__init__`` does **not**
import this module, so the core and the loopback path never require the extra (same discipline as
the lazy deepagents import).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, Request, Response

from verity.transport.base import Transport, TransportUnavailable
from verity.transport.client import RemoteVerifier
from verity.transport.envelope import RECOVERABLE_KINDS
from verity.transport.server import SandboxServer, VerifierServer

__all__ = [
    "HttpTransport",
    "build_asgi_app",
    "status_for_envelope",
    "build_remote_verifier",
]

_RECOVERABLE_STATUS = 503
_FATAL_STATUS = 422


@dataclass(slots=True)
class HttpTransport(Transport):
    """A :class:`Transport` over HTTP.

    By default each request uses a **fresh** ``httpx.AsyncClient`` — deliberate, because the control
    plane drives a dispatch on a *different* event loop than it configured on (``configure`` runs in
    one ``asyncio.run``, the dispatch on a background loop), and one async client cannot span loops
    ("Event loop is closed"). A verifier call is once per cycle, so a per-request connection is a
    negligible cost for that robustness. ``client`` may be injected (an ASGI client in tests) — then
    it is reused and the caller owns its lifecycle.
    """

    base_url: str
    client: httpx.AsyncClient | None = None
    timeout_s: float = 30.0

    async def request(self, method: str, body: bytes) -> bytes:
        url = f"{self.base_url}/{method}"
        try:
            if self.client is not None:
                response = await self.client.post(url, content=body)
            else:
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    response = await client.post(url, content=body)
        except httpx.HTTPError as exc:  # connection refused / timed out — the peer is unreachable
            raise TransportUnavailable(f"{type(exc).__name__}: {exc}") from exc
        # The body is the envelope (valid even on a 503 from our own server); the client parses it.
        return response.content

    async def aclose(self) -> None:
        if self.client is not None:
            await self.client.aclose()


def status_for_envelope(reply: bytes) -> int:
    """Advisory HTTP status for a reply envelope: 200 ok, 503 recoverable, 422 fatal."""
    try:
        obj = json.loads(reply)
    except (ValueError, TypeError):
        return _FATAL_STATUS
    if obj.get("ok"):
        return 200
    kind = (obj.get("error") or {}).get("kind", "")
    return _RECOVERABLE_STATUS if kind in RECOVERABLE_KINDS else _FATAL_STATUS


def build_remote_verifier(base_url: str, *, timeout_s: float = 30.0) -> RemoteVerifier:
    """A control-plane-side ``VerifierPort`` over HTTP at ``base_url``.

    Register it as a ``ProviderRegistry`` factory (``verifier_key="remote"``) so a task dispatches
    to a networked verifier with no control-plane change::

        url = os.environ["VERITY_VERIFIER_URL"]
        vp.register("remote", lambda: build_remote_verifier(url))
    """
    return RemoteVerifier(HttpTransport(base_url=base_url.rstrip("/"), timeout_s=timeout_s))


def build_asgi_app(server: VerifierServer | SandboxServer) -> FastAPI:
    """A FastAPI app whose single route forwards ``(method, body)`` to the server's ``handle``."""
    app = FastAPI()

    @app.post("/{method}")
    async def handle(method: str, request: Request) -> Response:
        reply = await server.handle(method, await request.body())
        return Response(
            content=reply, media_type="application/json", status_code=status_for_envelope(reply)
        )

    return app
