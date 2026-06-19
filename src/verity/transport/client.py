"""Port clients (ROADMAP Phase 7.2): the async ports implemented over a :class:`Transport`.

:class:`RemoteVerifier` and :class:`RemoteSandbox` satisfy the verifier / sandbox ports by
serializing each call (:mod:`verity.contracts.wire`), handing the bytes to a transport, and decoding
the reply (:mod:`verity.transport.envelope`, which re-raises boundary errors as themselves). They
are registered as ``ProviderRegistry`` factories under a ``remote`` key, so a task selects
in-process vs remote by ``TaskConfig.{sandbox,verifier}_key`` with **no control-plane change** —
every later port call simply crosses the transport.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from verity.contracts.errors import GateUnavailable
from verity.contracts.model import VerdictBundle
from verity.contracts.ports import (
    ProposalEnvelope,
    ServedContext,
    VerifierRequest,
    VerifierSetup,
)
from verity.contracts.wire import (
    proposal_envelope_from_dict,
    served_context_to_dict,
    verdict_bundle_from_dict,
    verifier_request_to_dict,
    verifier_setup_to_dict,
)
from verity.sandbox.errors import SandboxError
from verity.transport.base import Transport, TransportUnavailable
from verity.transport.envelope import parse_result

__all__ = ["RemoteVerifier", "RemoteSandbox"]


@dataclass(slots=True)
class RemoteVerifier:
    """A :class:`VerifierPort` over a transport. ``identity`` is learned in :meth:`provision`.

    A data-bearing verifier (feature-engineering) carries a per-task :class:`VerifierSetup` —
    shipped to the server **once**, at the front of :meth:`provision`, so the remote builds its gate
    stack before any dispatch. ``setup_payload`` is ``None`` for a dataless verifier (the fake),
    which is ready as soon as the service starts.
    """

    transport: Transport
    identity: str = ""
    setup_payload: VerifierSetup | None = None
    # When the control plane *launched* the verifier container, this destroys it at teardown — so
    # the standing daemon's ``verifier.teardown()`` reaps the on-demand sibling with no CP change
    # (the container lifecycle stays on the provisioning side; this is just the injected callback).
    on_teardown: Callable[[], Awaitable[None]] | None = None

    async def dispatch(self, request: VerifierRequest) -> VerdictBundle:
        body = json.dumps(verifier_request_to_dict(request)).encode("utf-8")
        # An unreachable verifier is a recoverable infra failure (GateUnavailable), not a crash —
        # exactly as a local verifier whose backend exhausted its retries (ROADMAP 5.1).
        try:
            result = parse_result(await self.transport.request("dispatch", body))
        except TransportUnavailable as exc:
            raise GateUnavailable(f"verifier unreachable: {exc}") from exc
        return verdict_bundle_from_dict(result)

    async def provision(self) -> None:
        # Ship the per-task data first (if any), then provision: the server builds its impl from the
        # setup payload, then provisions it and returns its identity (a sync attr in-process).
        if self.setup_payload is not None:
            setup = json.dumps(verifier_setup_to_dict(self.setup_payload)).encode("utf-8")
            try:
                parse_result(await self.transport.request("setup", setup))
            except TransportUnavailable as exc:
                raise GateUnavailable(f"verifier unreachable: {exc}") from exc
        result = parse_result(await self.transport.request("provision", b""))
        self.identity = str(result["identity"])

    async def teardown(self) -> None:
        # Best-effort wire teardown (the service may already be gone), then destroy the container if
        # we launched it — the latter must run even if the former can't reach the peer.
        try:
            parse_result(await self.transport.request("teardown", b""))
        except TransportUnavailable:
            pass
        finally:
            if self.on_teardown is not None:
                await self.on_teardown()

    async def health(self) -> bool:
        return bool(parse_result(await self.transport.request("health", b"")))


@dataclass(slots=True)
class RemoteSandbox:
    """A :class:`SandboxPort` backed by a transport."""

    transport: Transport

    async def _call(self, method: str, body: bytes = b"") -> Any:
        # An unreachable sandbox is a recoverable infra failure (SandboxError), not a crash (#21).
        try:
            return parse_result(await self.transport.request(method, body))
        except TransportUnavailable as exc:
            raise SandboxError(f"sandbox unreachable: {exc}") from exc

    async def serve_context(self, context: ServedContext) -> None:
        body = json.dumps(served_context_to_dict(context)).encode("utf-8")
        await self._call("serve_context", body)

    async def collect_proposal(self) -> ProposalEnvelope:
        result = await self._call("collect_proposal")
        return proposal_envelope_from_dict(result)

    async def regenerate(self) -> None:
        await self._call("regenerate")

    async def provision(self) -> None:
        parse_result(await self.transport.request("provision", b""))

    async def teardown(self) -> None:
        parse_result(await self.transport.request("teardown", b""))

    async def health(self) -> bool:
        return bool(parse_result(await self.transport.request("health", b"")))
