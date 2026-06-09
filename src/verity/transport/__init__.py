"""The transport seam (ROADMAP Phase 7.2) — the async ports over a byte mover.

The control plane talks to the sandbox and verifier only through the async ports, resolved by key
via a ``ProviderRegistry``. This package lets a *remote* impl slot in as just another registered
factory — **no `ControlPlane` change**: a client (:class:`RemoteVerifier` / :class:`RemoteSandbox`)
implements the port by serializing the call (:mod:`verity.contracts.wire`),
handing it to a :class:`Transport`, and decoding the reply; a server (:class:`VerifierServer` /
:class:`SandboxServer`) hosts the real in-process impl and dispatches incoming calls to it. The
:class:`LoopbackTransport` wires a client straight to a server's ``handle`` in-process, so the whole
seam — serialization, dispatch, error classification — is provable in CI with no socket. The HTTP
transport (PR4) is an added :class:`Transport`; nothing else changes.

Dependency-light: no HTTP libs here. The HTTP transport's deps live in a `service` extra, so the
lean core and the loopback path never import them.
"""

from __future__ import annotations

from verity.transport.base import Transport, TransportError
from verity.transport.client import RemoteSandbox, RemoteVerifier
from verity.transport.loopback import LoopbackTransport
from verity.transport.server import SandboxServer, VerifierServer

__all__ = [
    "Transport",
    "TransportError",
    "LoopbackTransport",
    "RemoteVerifier",
    "RemoteSandbox",
    "VerifierServer",
    "SandboxServer",
]
