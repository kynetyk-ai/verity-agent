"""The lifecycle state machine — legal status transitions and who may trigger them (spec §6).

``proposed → {tentative | rejected | revised}``; ``tentative → {accepted | rejected |
revised | superseded}``; ``accepted → superseded``. Only the commit path triggers
transitions out of ``proposed`` (§6, §7); that rule is enforced structurally by who holds
the :class:`~verity.control_plane.store.CommitSink` (§4.2), and the legality of each
*atomic* transition is enforced here, by the table below.

A single commit may apply several atomic transitions in sequence — an entirely-cheap
pipeline walks ``proposed → tentative → accepted`` within one commit (§6) — but every step
it takes must be a legal edge in this table.
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

# The transition table (spec §6). Each key may move only to the states in its value set.
# Note there is **no** direct ``proposed → accepted`` edge: acceptance always routes through
# ``tentative`` (§6). An entirely-cheap pipeline still walks ``proposed → tentative →
# accepted`` inside one commit — the cheap stage earns ``tentative``, then the (empty) hard
# stage earns ``accepted`` — so "proposed → accepted in one commit" (§6 prose) is honored as
# two legal atomic edges, not a new one.
LEGAL_TRANSITIONS: dict[ArtifactStatus, frozenset[ArtifactStatus]] = {
    _S.PROPOSED: frozenset({_S.TENTATIVE, _S.REJECTED, _S.REVISED}),
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
