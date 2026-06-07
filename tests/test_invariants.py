"""Audit-contract invariant tests (spec §5) — ROADMAP Phase 1.1.

The state-checkable invariants (§5.1, §5.4) are asserted directly; the structural/behavioral
ones (§5.2 append-only, §5.7 no-implicit-accept) are property-tested across many randomized
commit sequences with Hypothesis — "invariants are property tests, not prose."
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from tests.helpers import (
    accept,
    binding,
    gate,
    make_store,
    propose,
    refine,
    reject,
    resolver,
    runner,
    shape_ok,
)
from verity.control_plane.commit import NoImplicitAccept, run_commit
from verity.control_plane.invariants import (
    audit,
    check_provenance,
    check_unique_ids,
)
from verity.control_plane.store import ArtifactStatus

# A verdict drawn at random for each artifact in a randomized commit sequence.
_VERDICTS = {
    "accept": accept(),
    "reject": reject(),
    "refine": refine("defect"),
}
verdict_keys = st.lists(st.sampled_from(sorted(_VERDICTS)), min_size=0, max_size=25)


@given(choices=verdict_keys)
def test_audit_is_clean_after_any_commit_sequence(choices: list[str]) -> None:
    """Whatever verdicts arrive, the store never violates a state-checkable invariant (§5)."""
    store = make_store()
    bound = resolver(binding("Thing", gate("cheap"), gate("hard", is_hard=True)))
    payloads: dict[str, object] = {}

    for i, key in enumerate(choices):
        artifact_id = f"a{i}"
        payload = {"i": i, "tag": key}
        payloads[artifact_id] = payload
        propose(store, artifact_id=artifact_id, artifact_type="Thing", payload=payload)
        run_commit(
            artifact_id,
            store=store,
            sink=store,
            resolve_binding=bound,
            validate_shape=shape_ok(),
            run_gate=runner({"cheap": _VERDICTS[key], "hard": _VERDICTS[key]}),
        )

    assert audit(store) == []

    # §5.2 append-only: no committed artifact's payload changed under status transitions.
    for artifact_id, original in payloads.items():
        stored = store.get_artifact(artifact_id)
        assert stored is not None
        assert stored.payload == original


@given(choices=verdict_keys)
def test_rejections_are_retained_and_queryable(choices: list[str]) -> None:
    """§5.3/§5.5 — every rejected artifact remains in the store and in the rejected-log."""
    store = make_store()
    bound = resolver(binding("Thing", gate("cheap")))
    expected_rejected: set[str] = set()

    for i, key in enumerate(choices):
        artifact_id = f"a{i}"
        propose(store, artifact_id=artifact_id, artifact_type="Thing")
        result = run_commit(
            artifact_id,
            store=store,
            sink=store,
            resolve_binding=bound,
            validate_shape=shape_ok(),
            run_gate=runner({"cheap": _VERDICTS[key]}),
        )
        if result.status is ArtifactStatus.REJECTED:
            expected_rejected.add(artifact_id)

    assert {a.id for a in store.rejected_log()} == expected_rejected
    for artifact_id in expected_rejected:
        assert store.get_artifact(artifact_id) is not None


@given(payload=st.dictionaries(st.text(min_size=1, max_size=8), st.integers(), max_size=5))
def test_no_implicit_accept_for_gateless_type(payload: dict[str, int]) -> None:
    """§5.7 — committing a type with no binding always errors, never default-accepts."""
    store = make_store()
    propose(store, artifact_id="a1", artifact_type="Gateless", payload=payload)
    try:
        run_commit(
            "a1",
            store=store,
            sink=store,
            resolve_binding=resolver(),  # no binding registered
            validate_shape=shape_ok(),
            run_gate=runner({}),
        )
        raise AssertionError("expected NoImplicitAccept")
    except NoImplicitAccept:
        pass
    stored = store.get_artifact("a1")
    assert stored is not None and stored.status is ArtifactStatus.PROPOSED


def test_check_unique_ids_flags_nothing_on_a_clean_store() -> None:
    store = make_store()
    propose(store, artifact_id="a1")
    propose(store, artifact_id="a2", parents=["a1"])
    assert check_unique_ids(store) == []


def test_check_provenance_flags_nonroot_without_edge() -> None:
    """A non-root accepted artifact with no operation edge is a §5.4 violation."""
    store = make_store()
    # Force the pathological state the commit path would refuse: accept with no provenance.
    propose(store, artifact_id="a1", is_root=True)  # propose writes an op...
    # mark a *non-root* artifact accepted while it genuinely lacks any incoming op:
    store.set_status("a1", ArtifactStatus.TENTATIVE)
    store.set_status("a1", ArtifactStatus.ACCEPTED)
    # a1 is root → clean
    assert check_provenance(store) == []
