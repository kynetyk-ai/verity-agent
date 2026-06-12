"""The result/error envelope (ROADMAP Phase 7.2) — where degrade-don't-crash survives the hop.

Every server reply is a JSON envelope: ``{"v", "ok": true, "result": ...}`` on success, or
``{"v", "ok": false, "error": {"kind", "message"}}`` on a raised error. :func:`parse_result`
re-raises the **mapped typed error** client-side, so an in-process ``except GateUnavailable`` at the
control plane still fires when the verifier is remote — the run records a failed cycle and keeps
going, exactly as in-process (§3.4, ROADMAP 5.1). An unknown kind degrades to ``TransportError``.

The kind→class map is the load-bearing bit: the recoverable boundary errors (``GateUnavailable`` on
the gating side, ``SandboxError`` on the proposal side) and the fatal misuse error
(``VerifierError``) all cross as themselves. :data:`RECOVERABLE_KINDS` lets the HTTP transport (PR4)
also ride the distinction on a status code (503 vs 422) so a bare proxy 5xx still degrades.
"""

from __future__ import annotations

import json
from typing import Any

from verity.contracts.errors import GateUnavailable
from verity.contracts.ports import ProviderError
from verity.contracts.wire import WIRE_VERSION
from verity.sandbox.errors import SandboxError
from verity.transport.base import TransportError
from verity.verifier.errors import VerifierError

__all__ = [
    "ok_envelope",
    "err_envelope",
    "parse_result",
    "ERROR_KINDS",
    "RECOVERABLE_KINDS",
]

# The boundary errors that cross the wire as themselves (rather than collapsing to TransportError).
ERROR_KINDS: dict[str, type[Exception]] = {
    "GateUnavailable": GateUnavailable,
    "SandboxError": SandboxError,
    "VerifierError": VerifierError,
    "ProviderError": ProviderError,
}

# Recoverable (degrade-don't-crash) kinds — the control plane records the cycle and continues. The
# rest are fatal misuse. The HTTP transport maps recoverable→503, fatal→422.
RECOVERABLE_KINDS = frozenset({"GateUnavailable", "SandboxError"})


def ok_envelope(result: Any) -> bytes:
    """Encode a successful method result (a wire dict, a bool, or ``None``)."""
    return json.dumps({"v": WIRE_VERSION, "ok": True, "result": result}).encode("utf-8")


def err_envelope(exc: Exception) -> bytes:
    """Encode a raised error by class name + message, for the client to re-raise."""
    return json.dumps(
        {"v": WIRE_VERSION, "ok": False, "error": {"kind": type(exc).__name__, "message": str(exc)}}
    ).encode("utf-8")


def parse_result(raw: bytes) -> Any:
    """Decode an envelope: return the result, or re-raise the mapped typed error.

    A non-``ok`` envelope re-raises ``ERROR_KINDS[kind]`` (or :class:`TransportError` for an unknown
    kind), so the boundary error survives the hop and the existing control-plane catch fires.
    """
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise TransportError(f"malformed envelope: {exc}") from exc
    if obj.get("ok"):
        return obj.get("result")
    error = obj.get("error") or {}
    kind = error.get("kind", "")
    message = error.get("message", "")
    exc_type = ERROR_KINDS.get(kind, TransportError)
    raise exc_type(message)
