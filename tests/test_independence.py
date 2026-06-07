"""Independence-contract tests (spec §8.3, §10) — ROADMAP Phase 2.3.

A gate receives only the store-inputs it declared; the control plane assembles exactly those and
withholds everything else, and the resolved slice's sources are asserted ⊆ the declared allowlist
before dispatch (the §10 audit). These tests pin that behaviour and the over-broad-gate guard.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.helpers import make_store, propose
from verity.contracts import Artifact, ArtifactStatus
from verity.control_plane.independence import (
    STORE_INPUT_PROVIDERS,
    IndependenceViolation,
    StoreInput,
    resolve_declared_slice,
)

NOTE = "Note"


def _probe(artifact_type: str = NOTE) -> Artifact:
    """The artifact under test — only its type/id matter for slice resolution."""
    return Artifact("probe", artifact_type, {"text": "x"}, ArtifactStatus.PROPOSED, "agent", "t9")


def _seed(store) -> None:
    """One accepted Note, one rejected Note, one tentative Note, one accepted Other."""
    propose(store, artifact_id="acc", artifact_type=NOTE)
    store.set_status("acc", ArtifactStatus.TENTATIVE)
    store.set_status("acc", ArtifactStatus.ACCEPTED)
    propose(store, artifact_id="rej", artifact_type=NOTE)
    store.set_status("rej", ArtifactStatus.REJECTED)
    propose(store, artifact_id="ten", artifact_type=NOTE)
    store.set_status("ten", ArtifactStatus.TENTATIVE)
    propose(store, artifact_id="other", artifact_type="Other")
    store.set_status("other", ArtifactStatus.TENTATIVE)
    store.set_status("other", ArtifactStatus.ACCEPTED)


# ----------------------------------------------------------------- the declared-slice resolver


def test_no_declared_inputs_yields_an_empty_slice() -> None:
    store = make_store()
    _seed(store)
    # the tightest independence: a gate that declares nothing sees only the artifact under test
    assert resolve_declared_slice(store, _probe(), frozenset()) == ()


def test_incumbents_returns_only_accepted_same_type() -> None:
    store = make_store()
    _seed(store)
    got = resolve_declared_slice(store, _probe(), frozenset({StoreInput.INCUMBENTS}))
    ids = {a.id for a in got}
    assert ids == {"acc"}  # not the rejected, not the tentative, not the other-typed accepted


def test_rejected_log_returns_only_rejected_same_type() -> None:
    store = make_store()
    _seed(store)
    got = resolve_declared_slice(store, _probe(), frozenset({StoreInput.REJECTED_LOG}))
    assert {a.id for a in got} == {"rej"}


def test_both_inputs_union_without_leaking_other_kinds() -> None:
    store = make_store()
    _seed(store)
    got = resolve_declared_slice(
        store, _probe(), frozenset({StoreInput.INCUMBENTS, StoreInput.REJECTED_LOG})
    )
    assert {a.id for a in got} == {"acc", "rej"}  # tentative + other-typed are still withheld


def test_slice_is_artifacts_only_so_no_rationale_can_ride_along() -> None:
    # complements the type-level guarantee: every sliced item is an Artifact (no rationale field)
    store = make_store()
    _seed(store)
    got = resolve_declared_slice(store, _probe(), frozenset(StoreInput))
    assert all(isinstance(a, Artifact) for a in got)
    assert all(not hasattr(a, "rationale") for a in got)


# ----------------------------------------------------------------- the over-broad-gate guard (§10)


def test_every_store_input_has_a_provider() -> None:
    # a declared input with no provider would be a silently-empty slice; require full coverage
    assert set(STORE_INPUT_PROVIDERS) == set(StoreInput)


def test_a_declared_input_without_a_provider_fails_before_dispatch(monkeypatch) -> None:
    store = make_store()
    _seed(store)
    monkeypatch.delitem(STORE_INPUT_PROVIDERS, StoreInput.INCUMBENTS)
    with pytest.raises(IndependenceViolation):
        resolve_declared_slice(store, _probe(), frozenset({StoreInput.INCUMBENTS}))


# ----------------------------------------------------------------- the §10 property


@given(declared=st.sets(st.sampled_from(list(StoreInput))).map(frozenset))
def test_resolved_sources_are_always_within_the_declared_allowlist(
    declared: frozenset[StoreInput],
) -> None:
    store = make_store()
    _seed(store)
    got = resolve_declared_slice(store, _probe(), declared)
    # accepted-same-type may appear only if INCUMBENTS was declared; rejected only if REJECTED_LOG
    if StoreInput.INCUMBENTS not in declared:
        assert "acc" not in {a.id for a in got}
    if StoreInput.REJECTED_LOG not in declared:
        assert "rej" not in {a.id for a in got}
    # nothing outside the two providers' outputs ever appears
    assert {a.id for a in got} <= {"acc", "rej"}
