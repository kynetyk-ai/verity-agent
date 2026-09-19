"""The holdout-experiment verifier builder + entry point (verifier-side, ablation Exp 1–4b).

Offline: builds the verifier from a control-plane-shipped :class:`VerifierSetup` and inspects the
resulting pipeline — no Docker, no dispatch. Pins that ``accept_policy`` selects the hard step, that
the 3-file setup contract is enforced, and that the entry point registers under the expected name.
"""

from __future__ import annotations

import pytest

from verity.contracts.ports import VerifierSetup
from verity.verifier.holdout_service import (
    HOLDOUT_CONFIG,
    build_holdout_verifier_from_setup,
    register_verifier,
)
from verity.verifier.registry import VerifierRegistry

_OBJECTS = {
    "train.csv": b"id,class\n0,STAR\n",
    "holdout.csv": b"id\n1\n2\n",
    "holdout_labels.csv": b"id,class\n1,STAR\n2,GALAXY\n",
}


def _hard_gate(setup: VerifierSetup) -> str:
    verifier = build_holdout_verifier_from_setup(setup)
    steps = verifier.pipelines["Submission"]
    return next(s.gate for s in steps if s.is_hard)


def test_improve_policy_builds_the_selection_step() -> None:
    setup = VerifierSetup(objects=_OBJECTS, params={"accept_policy": "improve_over_best_prior"})
    assert _hard_gate(setup) == "selection"


def test_always_policy_builds_the_score_and_accept_step() -> None:
    setup = VerifierSetup(objects=_OBJECTS, params={"accept_policy": "always"})
    assert _hard_gate(setup) == "score-and-accept"


def test_default_policy_is_improve_over_best_prior() -> None:
    assert _hard_gate(VerifierSetup(objects=_OBJECTS, params={})) == "selection"


def test_missing_object_is_a_loud_error() -> None:
    incomplete = {k: v for k, v in _OBJECTS.items() if k != "holdout_labels.csv"}
    with pytest.raises(ValueError, match="holdout_labels.csv"):
        build_holdout_verifier_from_setup(VerifierSetup(objects=incomplete, params={}))


def test_entry_point_registers_the_setup_builder() -> None:
    registry = VerifierRegistry()
    register_verifier(registry)
    assert HOLDOUT_CONFIG in registry.names()
    assert registry.setup(HOLDOUT_CONFIG) is not None
    assert registry.impl(HOLDOUT_CONFIG) is None  # data-bearing setup, not a dataless impl
