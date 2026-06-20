"""The FE-Kaggle two-tier gate, offline (the `fe-kaggle` task type).

Drives the full read→propose→gate→commit loop over a `FakeBackend` (the agent + code-runner
doubles) and a `FakeKaggleScorer` (the real-world final test, scripted) — no Docker, model, or
network. The gate re-runs the agent's script on gold data (never the agent's CSV): a cheap
local-proxy filter on a hold-out, then the hard Kaggle gate that accepts iff the public score beats
best prior Kaggle-confirmed score. Asserts every branch the design promises.
"""

from __future__ import annotations

import asyncio

from tools.harness.dataset import stratified_split, subsample

from tests._fe_offline import FeWorkers, loopback_fe_kaggle_factory, role_files_from_raw
from verity.composition.fe_kaggle import FE_KAGGLE_TASK_ID, configure_fe_kaggle_task
from verity.contracts import ArtifactStatus, GateUnavailable
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.store import SqliteStore
from verity.domains.feature_engineering import SUBMISSION
from verity.provisioning import FakeBackend
from verity.verifier.kaggle import FakeKaggleScorer

_RAW = b"id,a,class\n0,0,X\n1,2,X\n2,4,X\n3,6,Y\n4,8,Y\n5,10,Y\n"
_REAL_TEST = b"id\n100\n101\n102\n103\n"  # unlabeled real test; the fake runner ignores its content

Step = tuple[str, tuple[str, ...], bytes, list[str]]


def _drive(
    *, steps: list[Step], scorer: FakeKaggleScorer, max_cycles: int, poll_interval_s: float = 0.0,
    proxy: list[str] | None = None,
):
    """Configure + run the fe-kaggle task offline. ``proxy`` sets each cycle's local proxy accuracy
    via the fake runner's predictions (``"perfect"`` -> 1.0, ``"half"`` -> 0.5 by predicting one
    class), since the broadened domain no longer derives proxy score from a feature count. Defaults
    to all-perfect when omitted."""
    # The user prepares the role-keyed inputs (ADR 0005); the deterministic split also gives us the
    # reserved labels we use to script the fake runner's proxy accuracy.
    rf = role_files_from_raw(_RAW, _REAL_TEST, per_class=100, reserved_fraction=0.5)
    sub = subsample(_RAW, per_class=100, target="class")
    split = stratified_split(sub, target="class", id_column="id", reserved_fraction=0.5)
    reserved = dict(split.reserved_labels)
    one_class = sorted(set(reserved.values()))[0]
    quality = proxy or ["perfect"] * len(steps)
    by_marker = {
        marker: (reserved if q == "perfect" else dict.fromkeys(reserved, one_class))
        for (_, _, marker, _), q in zip(steps, quality, strict=True)
    }
    backend = FakeBackend(script=FeWorkers(steps=steps, by_marker=by_marker))
    store = SqliteStore()
    cp = ControlPlane(store, policy=OrchestrationPolicy(max_cycles=max_cycles))
    asyncio.run(
        configure_fe_kaggle_task(
            cp, backend=backend,
            make_verifier=loopback_fe_kaggle_factory(backend, scorer),
            agent_files=rf["agent"], verifier_files=rf["verifier"],
            poll_interval_s=poll_interval_s, wait_deadline_s=5.0,
        )
    )
    asyncio.run(cp.run(FE_KAGGLE_TASK_ID, goal="climb the leaderboard"))
    return store


def _accepted(store: SqliteStore) -> list:
    return store.query_artifacts(type=SUBMISSION, status=ArtifactStatus.ACCEPTED)


def test_accept_and_supersede_on_the_real_leaderboard() -> None:
    # cycle1: proxy 0.5, Kaggle 0.80 -> accepted. cycle2: proxy 1.0 > 0.5 -> passes the filter,
    # Kaggle 0.85 > 0.80 -> accepted and SUPERSEDES cycle1 on the real score.
    scorer = FakeKaggleScorer(scores=[0.80, 0.85], budget=5)
    store = _drive(
        steps=[("submit", ("ds",), b"a", ["f0"]), ("submit", ("ds",), b"b", ["f0"])],
        proxy=["half", "perfect"], scorer=scorer, max_cycles=2,
    )
    assert len(_accepted(store)) == 1  # cycle2 is the live incumbent
    assert len(store.query_artifacts(type=SUBMISSION, status=ArtifactStatus.SUPERSEDED)) == 1
    assert len(scorer.submitted) == 2  # both cleared the proxy and reached Kaggle


def test_loses_on_the_real_score_is_rejected() -> None:
    # cycle2 clears the proxy (1.0 > 0.5) so it DOES submit, but its real score loses -> rejected.
    scorer = FakeKaggleScorer(scores=[0.80, 0.70], budget=5)
    store = _drive(
        steps=[("submit", ("ds",), b"a", ["f0"]), ("submit", ("ds",), b"b", ["f0"])],
        proxy=["half", "perfect"], scorer=scorer, max_cycles=2,
    )
    assert len(_accepted(store)) == 1  # only cycle1; cycle2 lost on the leaderboard
    assert len(store.query_artifacts(type=SUBMISSION, status=ArtifactStatus.REJECTED)) >= 1
    assert len(scorer.submitted) == 2  # cycle2 was submitted (it passed the proxy) and lost


def test_proxy_worse_is_a_cheap_reject_without_a_kaggle_call() -> None:
    # cycle1: proxy 1.0, accepted. cycle2: proxy 0.5 <= 1.0 -> cheap reject; the budget-saving
    # filter means NO Kaggle submission is spent on it.
    scorer = FakeKaggleScorer(scores=[0.80, 0.99], budget=5)
    store = _drive(
        steps=[("submit", ("ds",), b"a", ["f0"]), ("submit", ("ds",), b"b", ["f0"])],
        proxy=["perfect", "half"], scorer=scorer, max_cycles=2,
    )
    assert len(_accepted(store)) == 1
    assert len(scorer.submitted) == 1  # only cycle1 reached Kaggle


def test_blocks_until_the_daily_budget_frees() -> None:
    # remaining_budget reads 0, 0, then 3: the gate blocks (polls) and proceeds once a slot frees.
    scorer = FakeKaggleScorer(scores=[0.80], budget_readings=[0, 0, 3])
    store = _drive(steps=[("submit", ("ds",), b"a", ["f0"])], scorer=scorer, max_cycles=1)
    assert len(_accepted(store)) == 1
    assert len(scorer.submitted) == 1
    assert scorer._budget_i >= 3  # it polled through the two 0s before the slot freed


def test_kaggle_api_failure_degrades_the_cycle() -> None:
    # submit raises -> recoverable GateUnavailable -> recorded failed cycle; run ok, no commit
    scorer = FakeKaggleScorer(fail_with=GateUnavailable("kaggle unreachable"))
    store = _drive(steps=[("submit", ("ds",), b"a", ["f0"])], scorer=scorer, max_cycles=1)
    assert _accepted(store) == []  # nothing committed; the run did not crash (we reached here)
