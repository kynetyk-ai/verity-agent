"""Verifier-service errors (spec §3.6)."""

from __future__ import annotations

__all__ = ["VerifierError"]


class VerifierError(RuntimeError):
    """A misuse of the verifier service.

    The verifier-side analogue of the commit path's :class:`NoImplicitAccept`: a gate named in a
    binding that has no registered plugin is a configuration error, surfaced loudly — never a
    silent accept (§3.6, §5.7).
    """
