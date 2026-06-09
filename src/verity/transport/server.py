"""Service-side request handlers (ROADMAP Phase 7.2).

A ``*Server`` wraps a real in-process :class:`VerifierPort` / :class:`SandboxPort` and turns an
incoming ``(method, body)`` into the impl's coroutine, encoding the result — or any raised boundary
error — into a JSON envelope. Transport-agnostic: the loopback transport calls ``handle`` directly;
the HTTP transport (PR4) calls it from an ASGI route. The codec is :mod:`verity.contracts.wire`; the
envelope is :mod:`verity.transport.envelope`.
"""

from __future__ import annotations

import json
from typing import Any, cast

from verity.contracts.ports import SandboxPort, VerifierPort
from verity.contracts.wire import (
    proposal_envelope_to_dict,
    served_context_from_dict,
    verdict_bundle_to_dict,
    verifier_request_from_dict,
)
from verity.transport.base import TransportError
from verity.transport.envelope import err_envelope, ok_envelope

__all__ = ["VerifierServer", "SandboxServer"]


def _args(body: bytes) -> dict[str, Any]:
    """Decode a method's argument dict (an empty body means a no-arg call)."""
    if not body:
        return {}
    return cast("dict[str, Any]", json.loads(body))


class VerifierServer:
    """Hosts a real :class:`VerifierPort`; dispatches ``dispatch`` / lifecycle / ``provision``."""

    def __init__(self, impl: VerifierPort) -> None:
        self._impl = impl

    async def handle(self, method: str, body: bytes) -> bytes:
        try:
            return await self._dispatch(method, body)
        except Exception as exc:  # any raised boundary error rides the envelope, classified by kind
            return err_envelope(exc)

    async def _dispatch(self, method: str, body: bytes) -> bytes:
        if method == "dispatch":
            request = verifier_request_from_dict(_args(body))
            bundle = await self._impl.dispatch(request)
            return ok_envelope(verdict_bundle_to_dict(bundle))
        if method == "provision":
            await self._impl.provision()
            # The remote learns the verifier's identity here (it is a sync attribute in-process).
            return ok_envelope({"identity": self._impl.identity})
        if method == "teardown":
            await self._impl.teardown()
            return ok_envelope(None)
        if method == "health":
            return ok_envelope(await self._impl.health())
        raise TransportError(f"unknown verifier method: {method!r}")


class SandboxServer:
    """Hosts a real :class:`SandboxPort`; dispatches serve/collect/regenerate + lifecycle."""

    def __init__(self, impl: SandboxPort) -> None:
        self._impl = impl

    async def handle(self, method: str, body: bytes) -> bytes:
        try:
            return await self._dispatch(method, body)
        except Exception as exc:
            return err_envelope(exc)

    async def _dispatch(self, method: str, body: bytes) -> bytes:
        if method == "serve_context":
            await self._impl.serve_context(served_context_from_dict(_args(body)))
            return ok_envelope(None)
        if method == "collect_proposal":
            envelope = await self._impl.collect_proposal()
            return ok_envelope(proposal_envelope_to_dict(envelope))
        if method == "regenerate":
            await self._impl.regenerate()
            return ok_envelope(None)
        if method == "provision":
            await self._impl.provision()
            return ok_envelope(None)
        if method == "teardown":
            await self._impl.teardown()
            return ok_envelope(None)
        if method == "health":
            return ok_envelope(await self._impl.health())
        raise TransportError(f"unknown sandbox method: {method!r}")
