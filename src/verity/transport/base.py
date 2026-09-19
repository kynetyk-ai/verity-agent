"""The transport abstraction (ROADMAP Phase 7.2): a dumb async byte mover.

A :class:`Transport` ships one request's bytes to a named method and returns the reply's bytes. It
knows nothing about the value types or the application errors — those live in the JSON envelope
(:mod:`verity.transport.envelope`), so the same client/server pair runs over loopback (in-process,
for CI) or HTTP (a real network) by swapping only the transport.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Transport(Protocol):
    """Move one method call's bytes to a peer and return the reply's bytes."""

    async def request(self, method: str, body: bytes) -> bytes: ...
    async def aclose(self) -> None: ...


class TransportError(RuntimeError):
    """A transport- or protocol-level failure (a malformed envelope, an unknown error kind).

    Distinct from the *application* errors that ride the envelope intact (``GateUnavailable`` /
    ``SandboxError`` / ``VerifierError``): those are re-raised as themselves client-side so the
    control plane's degrade-don't-crash classification survives the hop. ``TransportError`` is the
    fallback for everything the protocol itself could not carry.
    """


class TransportUnavailable(TransportError):
    """The peer could not be reached (connection refused / timed out before any envelope).

    A client maps this to its own *recoverable* boundary error — a remote verifier to
    ``GateUnavailable``, a remote sandbox to ``SandboxError`` — so an unreachable service degrades
    the cycle rather than crashing the run, exactly as a local infra failure would.
    """
