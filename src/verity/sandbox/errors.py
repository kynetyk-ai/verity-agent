"""Sandbox errors (ROADMAP Phase 3)."""

from __future__ import annotations

__all__ = ["SandboxError"]


class SandboxError(RuntimeError):
    """Raised when a sandbox cycle cannot produce a well-formed proposal (no descriptor, etc.)."""
