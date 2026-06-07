"""Lifecycle state-machine tests (spec §6) — ROADMAP Phase 1.1."""

from __future__ import annotations

import itertools

import pytest

from verity.control_plane.lifecycle import (
    LEGAL_TRANSITIONS,
    TERMINAL_STATUSES,
    IllegalTransition,
    assert_transition,
    is_legal_transition,
    is_terminal,
)
from verity.control_plane.store import ArtifactStatus as S

LEGAL_EDGES = {
    (S.PROPOSED, S.TENTATIVE),
    (S.PROPOSED, S.REJECTED),
    (S.PROPOSED, S.REVISED),
    (S.TENTATIVE, S.ACCEPTED),
    (S.TENTATIVE, S.REJECTED),
    (S.TENTATIVE, S.REVISED),
    (S.TENTATIVE, S.SUPERSEDED),
    (S.ACCEPTED, S.SUPERSEDED),
}


@pytest.mark.parametrize(("frm", "to"), sorted(LEGAL_EDGES, key=lambda e: (e[0].value, e[1].value)))
def test_legal_edges_are_allowed(frm: S, to: S) -> None:
    assert is_legal_transition(frm, to)
    assert_transition(frm, to)  # does not raise


@pytest.mark.parametrize(
    ("frm", "to"),
    sorted(
        {(f, t) for f in S for t in S} - LEGAL_EDGES,
        key=lambda e: (e[0].value, e[1].value),
    ),
)
def test_illegal_edges_are_rejected(frm: S, to: S) -> None:
    assert not is_legal_transition(frm, to)
    with pytest.raises(IllegalTransition):
        assert_transition(frm, to)


def test_proposed_cannot_jump_straight_to_accepted() -> None:
    # Acceptance always routes through 'tentative' (§6).
    assert not is_legal_transition(S.PROPOSED, S.ACCEPTED)


def test_terminal_statuses_have_no_outgoing_edges() -> None:
    for status in TERMINAL_STATUSES:
        assert LEGAL_TRANSITIONS[status] == frozenset()
        assert is_terminal(status)


def test_accepted_is_not_terminal() -> None:
    # 'accepted' can still be superseded by a better artifact (§6).
    assert not is_terminal(S.ACCEPTED)
    assert S.SUPERSEDED in LEGAL_TRANSITIONS[S.ACCEPTED]


def test_every_status_has_a_transition_entry() -> None:
    assert set(LEGAL_TRANSITIONS) == set(S)


def test_transition_table_matches_declared_edges() -> None:
    derived = {
        (frm, to) for frm, tos in LEGAL_TRANSITIONS.items() for to in tos
    }
    assert derived == LEGAL_EDGES


def test_no_self_loops() -> None:
    for status in itertools.chain(S):
        assert status not in LEGAL_TRANSITIONS[status]
