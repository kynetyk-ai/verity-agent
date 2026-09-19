"""Service-side request handlers (ROADMAP Phase 7.2).

A ``*Server`` wraps a real in-process :class:`VerifierPort` / :class:`SandboxPort` and turns an
incoming ``(method, body)`` into the impl's coroutine, encoding the result — or any raised boundary
error — into a JSON envelope. Transport-agnostic: the loopback transport calls ``handle`` directly;
the HTTP transport (PR4) calls it from an ASGI route. The codec is :mod:`verity.contracts.wire`; the
envelope is :mod:`verity.transport.envelope`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

from verity.contracts.ports import SandboxPort, VerifierPort, VerifierSetup
from verity.contracts.wire import (
    proposal_envelope_to_dict,
    served_context_from_dict,
    verdict_bundle_to_dict,
    verifier_request_from_dict,
    verifier_setup_from_dict,
)
from verity.transport.base import TransportError
from verity.transport.envelope import err_envelope, ok_envelope

__all__ = ["VerifierServer", "SandboxServer"]

# A data-bearing verifier is built server-side from the per-task setup payload (§9.1 / #73).
VerifierBuilder = Callable[[VerifierSetup], VerifierPort]


def _args(body: bytes) -> dict[str, Any]:
    """Decode a method's argument dict (an empty body means a no-arg call)."""
    if not body:
        return {}
    return cast("dict[str, Any]", json.loads(body))


class VerifierServer:
    """Hosts a :class:`VerifierPort`; dispatches ``setup`` / ``dispatch`` / lifecycle.

    Two construction modes. A **dataless** verifier (e.g. the fake) is passed directly as ``impl``
    and is ready immediately. A **data-bearing** verifier (feature-engineering's CSV-fed gates) is
    passed as a ``builder``: the control plane ships the per-task :class:`VerifierSetup` once, the
    ``setup`` method builds the impl from it server-side, and only then is the verifier ready. The
    daemon serializes runs (one active task), so a single held impl is sufficient; ``teardown``
    clears a builder-made impl so the next task rebuilds fresh.
    """

    def __init__(
        self, impl: VerifierPort | None = None, *, builder: VerifierBuilder | None = None
    ) -> None:
        if impl is None and builder is None:
            raise ValueError("VerifierServer needs an impl or a builder")
        self._impl = impl
        self._builder = builder

    def _require_impl(self) -> VerifierPort:
        if self._impl is None:
            raise TransportError("verifier not set up yet: send `setup` before `provision`")
        return self._impl

    async def handle(self, method: str, body: bytes) -> bytes:
        try:
            return await self._dispatch(method, body)
        except Exception as exc:  # any raised boundary error rides the envelope, classified by kind
            return err_envelope(exc)

    async def _dispatch(self, method: str, body: bytes) -> bytes:
        if method == "setup":
            if self._builder is None:
                raise TransportError("this verifier takes no setup (built with a fixed impl)")
            self._impl = self._builder(verifier_setup_from_dict(_args(body)))
            return ok_envelope(None)
        if method == "dispatch":
            impl = self._require_impl()  # before parsing, so "not set up yet" is the precise error
            bundle = await impl.dispatch(verifier_request_from_dict(_args(body)))
            return ok_envelope(verdict_bundle_to_dict(bundle))
        if method == "provision":
            impl = self._require_impl()
            await impl.provision()
            # The remote learns the verifier's identity here (it is a sync attribute in-process).
            return ok_envelope({"identity": impl.identity})
        if method == "teardown":
            if self._impl is not None:
                await self._impl.teardown()
                if self._builder is not None:
                    self._impl = None  # a builder-made impl is per-task; rebuild on the next setup
            return ok_envelope(None)
        if method == "health":
            # Healthy as a *service* even before setup — readiness to accept work, not task state.
            return ok_envelope(True if self._impl is None else await self._impl.health())
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
