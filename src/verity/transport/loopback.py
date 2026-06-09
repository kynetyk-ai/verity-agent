"""The loopback transport (ROADMAP Phase 7.2): the CI workhorse, no socket.

``LoopbackTransport`` wires a client straight to a server's ``handle`` coroutine in-process, so a
``RemoteVerifier(LoopbackTransport(VerifierServer(real_impl).handle))`` exercises the full
serialize → dispatch → serialize → deserialize path — object bytes, rationale segregation, and error
classification included — with no network. HTTP is the same shape with a socket between.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

__all__ = ["LoopbackTransport"]


@dataclass(slots=True)
class LoopbackTransport:
    """A :class:`~verity.transport.base.Transport` that calls a server ``handle`` in-process."""

    handler: Callable[[str, bytes], Awaitable[bytes]]

    async def request(self, method: str, body: bytes) -> bytes:
        return await self.handler(method, body)

    async def aclose(self) -> None:
        return None
