"""The FE-Kaggle two-tier gate, offline (the `fe-kaggle` task type).

Drives the full read→propose→gate→commit loop over a `FakeBackend` (the agent + code-runner
doubles) and a `FakeKaggleScorer` (the real-world final test, scripted) — no Docker, model, or
network. The gate re-runs the agent's script on gold data (never the agent's CSV): a cheap
local-proxy filter on a hold-out, then the hard Kaggle gate that accepts iff the public score beats
best prior Kaggle-confirmed score. Asserts every branch the design promises.
"""

from __future__ import annotations

import asyncio

from tests._fe_offline import FeWorkers
from verity.composition.dataset import stratified_split, subsample
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
    *, steps: list[Step], scorer: FakeKaggleScorer, max_cycles: int, poll_interval_s: float = 0.0
):
    """Configure + run the fe-kaggle task offline; the fake runner predicts the reserved labels
    exactly (proxy accuracy 1.0), so a cycle's proxy score is set purely by its feature count."""
    sub = subsample(_RAW, per_class=100, target="class")
    split = stratified_split(sub, target="class", id_column="id", reserved_fraction=0.5)
    by_marker = {marker: dict(split.reserved_labels) for (_, _, marker, _) in steps}
    backend = FakeBackend(script=FeWorkers(steps=steps, by_marker=by_marker))
    store = SqliteStore()
    cp = ControlPlane(store, policy=OrchestrationPolicy(max_cycles=max_cycles))
    asyncio.run(
        configure_fe_kaggle_task(
            cp, backend=backend, scorer=scorer, split=split,
            full_train_csv=sub, real_test_csv=_REAL_TEST,
            poll_interval_s=poll_interval_s, wait_deadline_s=5.0,
        )
    )
    asyncio.run(cp.run(FE_KAGGLE_TASK_ID, goal="climb the leaderboard"))
    return store


def _accepted(store: SqliteStore) -> list:
    return store.query_artifacts(type=SUBMISSION, status=ArtifactStatus.ACCEPTED)


def test_accept_and_supersede_on_the_real_leaderboard() -> None:
    # cycle1: 2 features (proxy net 0.98), Kaggle 0.80 -> accepted. cycle2: 1 feature (proxy 0.99 >
    # 0.98 -> passes the filter), Kaggle 0.85 > 0.80 -> accepted and SUPERSEDES cycle1 on the score.
    scorer = FakeKaggleScorer(scores=[0.80, 0.85], budget=5)
    store = _drive(
        steps=[("submit", ("ds",), b"a", ["f0", "f1"]), ("submit", ("ds",), b"b", ["f0"])],
        scorer=scorer, max_cycles=2,
    )
    assert len(_accepted(store)) == 1  # cycle2 is the live incumbent
    assert len(store.query_artifacts(type=SUBMISSION, status=ArtifactStatus.SUPERSEDED)) == 1
    assert len(scorer.submitted) == 2  # both cleared the proxy and reached Kaggle


def test_loses_on_the_real_score_is_rejected() -> None:
    # cycle2 clears the proxy (0.99 > 0.98) so it DOES submit, but its real score loses -> rejected.
    scorer = FakeKaggleScorer(scores=[0.80, 0.70], budget=5)
    store = _drive(
        steps=[("submit", ("ds",), b"a", ["f0", "f1"]), ("submit", ("ds",), b"b", ["f0"])],
        scorer=scorer, max_cycles=2,
    )
    assert len(_accepted(store)) == 1  # only cycle1; cycle2 lost on the leaderboard
    assert len(store.query_artifacts(type=SUBMISSION, status=ArtifactStatus.REJECTED)) >= 1
    assert len(scorer.submitted) == 2  # cycle2 was submitted (it passed the proxy) and lost


def test_proxy_worse_is_a_cheap_reject_without_a_kaggle_call() -> None:
    # cycle1: 1 feature (proxy 0.99), accepted. cycle2: 3 features (proxy 0.97 <= 0.99) -> cheap
    # reject; the budget-saving filter means NO Kaggle submission is spent on it.
    scorer = FakeKaggleScorer(scores=[0.80, 0.99], budget=5)
    store = _drive(
        steps=[("submit", ("ds",), b"a", ["f0"]), ("submit", ("ds",), b"b", ["f0", "f1", "f2"])],
        scorer=scorer, max_cycles=2,
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
