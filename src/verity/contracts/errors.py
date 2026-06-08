"""Cross-service boundary errors (spec §3.3, ROADMAP 5.1).

Errors that one service *raises* and another *catches* belong to the shared contract, not to either
service. :class:`GateUnavailable` is the gating-side analogue of the sandbox's ``SandboxError``: it
says the verifier could not render a verdict because of **infrastructure** (a transient model or
container failure that exhausted its retries, or a dispatch that timed out) — *not* that the
verifier ruled reject, and *not* that the proposal or a binding is malformed (those are misuse, and
stay fatal). The control plane treats it as a recoverable failed cycle — recorded, not fatal —
exactly as it already treats a ``SandboxError`` on the proposal side.
"""

from __future__ import annotations

__all__ = ["GateUnavailable"]


class GateUnavailable(RuntimeError):
    """The verifier could not render a verdict due to infrastructure (transient, recoverable).

    Distinct from the commit path's misuse errors (``CommitError`` / ``NoImplicitAccept`` /
    ``ProposerIsGate`` / ``BundleInconsistent``) and from ``VerifierError`` (a gate with no plugin):
    those are bugs or config to fix and remain fatal. ``GateUnavailable`` is the transient case —
    the gate infrastructure was unreachable this cycle — so the run degrades and continues (§3.4).
    """
