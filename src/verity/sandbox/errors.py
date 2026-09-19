"""Sandbox errors (ROADMAP Phase 3)."""

from __future__ import annotations

__all__ = ["SandboxError"]


class SandboxError(RuntimeError):
    """Raised when a sandbox cycle cannot produce a well-formed proposal (no descriptor, etc.).

    ``transcript`` optionally carries the agent's rendered step transcript (JSON bytes) saved from
    the failed cycle — e.g. a recursion/step-budget limit-end that produced no proposal. The control
    plane content-addresses it into the store and references it from the cycle's report, so a failed
    or no-proposal cycle is still debuggable (it is *exactly* when the trace matters most).

    ``telemetry`` likewise carries the raw ``__telemetry__.json`` bytes (incl. ``stop_reason``) so
    the control plane can record *why* the cycle stopped — without it, the recursion/step-budget
    reason is written to the outbox but discarded host-side on the no-proposal path.
    """

    def __init__(
        self, *args: object, transcript: bytes | None = None, telemetry: bytes | None = None
    ) -> None:
        super().__init__(*args)
        self.transcript = transcript
        self.telemetry = telemetry
