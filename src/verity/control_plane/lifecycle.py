"""The lifecycle state machine — legal status transitions and who may trigger them (spec §6).

``proposed → {tentative | accepted | rejected | revised}``; ``tentative → {accepted |
rejected | revised | superseded}``; ``accepted → superseded``. Only the commit path triggers
transitions out of ``proposed`` (§6, §7); that rule is enforced structurally by who holds
the :class:`~verity.control_plane.store.CommitSink` (§4.2), and the legality of each
transition is enforced here, by the table below. Under ADR 0001 the opaque verifier reports the
terminal status it recommends, and the commit path validates it against this table before writing.
"""

from __future__ import annotations

from verity.control_plane.store import ArtifactStatus

__all__ = [
    "LEGAL_TRANSITIONS",
    "TERMINAL_STATUSES",
    "IllegalTransition",
    "is_legal_transition",
    "assert_transition",
    "is_terminal",
]

_S = ArtifactStatus

# The transition table (spec §6, with the §6-prose edge made explicit per ADR 0001). Each key may
# move only to the states in its value set. ``proposed → accepted`` is a legal edge: the opaque
# verifier now owns staging and reports the terminal status directly, so an all-cheap (or
# fully-cleared) proposal is accepted in one transition. The control plane validates the verifier's
# reported status against this table before writing it — an illegal edge is refused even on the
# verifier's say-so.
LEGAL_TRANSITIONS: dict[ArtifactStatus, frozenset[ArtifactStatus]] = {
    _S.PROPOSED: frozenset({_S.TENTATIVE, _S.ACCEPTED, _S.REJECTED, _S.REVISED}),
    _S.TENTATIVE: frozenset({_S.ACCEPTED, _S.REJECTED, _S.REVISED, _S.SUPERSEDED}),
    _S.ACCEPTED: frozenset({_S.SUPERSEDED}),
    _S.REJECTED: frozenset(),
    _S.SUPERSEDED: frozenset(),
    _S.REVISED: frozenset(),
}

# ``rejected``, ``superseded``, and ``revised`` are terminal and retained (spec §6, §5.3).
# ``accepted`` is *not* terminal — it may still be superseded by a better artifact.
TERMINAL_STATUSES: frozenset[ArtifactStatus] = frozenset(
    {_S.REJECTED, _S.SUPERSEDED, _S.REVISED}
)


class IllegalTransition(RuntimeError):
    """Raised when a status transition is not an edge in :data:`LEGAL_TRANSITIONS` (§6)."""

    def __init__(self, frm: ArtifactStatus, to: ArtifactStatus) -> None:
        super().__init__(f"illegal lifecycle transition: {frm.value} → {to.value}")
        self.frm = frm
        self.to = to


def is_legal_transition(frm: ArtifactStatus, to: ArtifactStatus) -> bool:
    """True iff ``frm → to`` is a legal edge in the state machine (spec §6)."""
    return to in LEGAL_TRANSITIONS[frm]


def assert_transition(frm: ArtifactStatus, to: ArtifactStatus) -> None:
    """Raise :class:`IllegalTransition` unless ``frm → to`` is legal (spec §6)."""
    if not is_legal_transition(frm, to):
        raise IllegalTransition(frm, to)


def is_terminal(status: ArtifactStatus) -> bool:
    """True for the retained terminal states ``rejected``/``superseded``/``revised`` (§6)."""
    return status in TERMINAL_STATUSES
