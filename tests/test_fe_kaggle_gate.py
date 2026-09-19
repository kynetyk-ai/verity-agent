"""The FE-Kaggle gate ladder, offline (the `fe-kaggle` task type).

Drives the full read→propose→gate→commit loop over a `FakeBackend` (the agent + code-runner
doubles) and a `FakeKaggleScorer` (the real-world final test + leaderboard, scripted) — no Docker,
model, or network. The gate re-runs the agent's script on gold data (never the agent's CSV): the
cheap local-proxy filter on a hold-out, the **competitive bar** (only submit an attempt we believe
is top-N% on the live leaderboard — else *refine*, spend nothing), then the hard Kaggle gate that
accepts iff the public score beats our best prior Kaggle-confirmed score. Asserts every branch the
design promises, spending **zero** real submissions.
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
    proxy: list[str] | None = None, target_fraction: float = 0.10, proxy_margin: float = 0.0,
    min_calibration_points: int = 2, pessimism: float = 0.0,
):
    """Configure + run the fe-kaggle task offline. ``proxy`` sets each cycle's local proxy accuracy
    via the fake runner's predictions (``"perfect"`` -> 1.0, ``"half"`` -> 0.5 by predicting one
    class), since the broadened domain no longer derives proxy score from a feature count. Defaults
    to all-perfect when omitted. The competitive gate's bar comes from ``scorer.leaderboard`` (empty
    -> bar skipped); ``target_fraction`` / ``proxy_margin`` / calibration knobs tune the gates."""
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
            target_fraction=target_fraction, proxy_margin=proxy_margin,
            min_calibration_points=min_calibration_points, pessimism=pessimism,
        )
    )
    asyncio.run(cp.run(FE_KAGGLE_TASK_ID, goal="climb the leaderboard"))
    return store


def _accepted(store: SqliteStore) -> list:
    return store.query_artifacts(type=SUBMISSION, status=ArtifactStatus.ACCEPTED)


def _revised(store: SqliteStore) -> list:
    return store.query_artifacts(type=SUBMISSION, status=ArtifactStatus.REVISED)


def _rejected(store: SqliteStore) -> list:
    return store.query_artifacts(type=SUBMISSION, status=ArtifactStatus.REJECTED)


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


# --------------------------------------------------------------- the competitive (top-N%) bar


def test_below_competitive_bar_refines_without_a_submission() -> None:
    # The top-10% bar is 0.99; our hold-out estimate is 0.5 -> we are not competitive, so the gate
    # *refines* (the attempt rests 'revised', the gap is fed back) and spends NO Kaggle submission.
    scorer = FakeKaggleScorer(scores=[0.80], budget=5, leaderboard=[0.99])
    store = _drive(
        steps=[("submit", ("ds",), b"a", ["f0"])],
        proxy=["half"], scorer=scorer, max_cycles=1, target_fraction=0.10,
    )
    assert scorer.submitted == []  # the budget-conserving point: no submission below the bar
    assert _accepted(store) == []
    assert len(_revised(store)) == 1  # the refine target the agent edits next cycle


def test_clears_competitive_bar_submits_and_accepts() -> None:
    # Estimate 1.0 clears the top-10% bar (0.40) -> submit; Kaggle 0.80 beats nothing -> accept.
    scorer = FakeKaggleScorer(scores=[0.80], budget=5, leaderboard=[0.40, 0.30, 0.20])
    store = _drive(
        steps=[("submit", ("ds",), b"a", ["f0"])],
        proxy=["perfect"], scorer=scorer, max_cycles=1, target_fraction=0.10,
    )
    assert len(scorer.submitted) == 1
    assert len(_accepted(store)) == 1
    # the scores that document the climb are recorded on the decisions (so they reach the RunReport
    # the trajectory readout reads): the competitive estimate and the public Kaggle score.
    scores = {d.gate: d.score for d in store.decisions_for(_accepted(store)[0].id)}
    assert scores["competitive"] is not None
    assert scores["kaggle"] == 0.80


def test_refine_then_clear_climbs_to_a_submission() -> None:
    # The goal-seeking loop: cycle1 below the bar -> refine (no submission); cycle2 improves past
    # the bar -> submit + accept. The run *documents the climb* from revised to accepted.
    scorer = FakeKaggleScorer(scores=[0.80], budget=5, leaderboard=[0.60])
    store = _drive(
        steps=[("submit", ("ds",), b"a", ["f0"]), ("submit", ("ds",), b"b", ["f0"])],
        proxy=["half", "perfect"], scorer=scorer, max_cycles=2, target_fraction=0.10,
    )
    assert len(_revised(store)) == 1  # cycle1 rested revised
    assert len(scorer.submitted) == 1  # only cycle2, the competitive attempt, reached Kaggle
    assert len(_accepted(store)) == 1


def test_proxy_margin_rejects_below_the_noise_floor() -> None:
    # A 0.6 noise margin rejects a hold-out proxy of 0.5 as not a real signal (no incumbent yet, so
    # the bar is 0) -> a cheap reject before the competitive/Kaggle rungs. With margin 0 the same
    # 0.5 proxy passes (the other tests), so this isolates the margin's effect.
    scorer = FakeKaggleScorer(scores=[0.80], budget=5)
    store = _drive(
        steps=[("submit", ("ds",), b"a", ["f0"])],
        proxy=["half"], scorer=scorer, max_cycles=1, proxy_margin=0.6,
    )
    assert scorer.submitted == []
    assert _accepted(store) == []
    assert len(_rejected(store)) == 1


def test_calibration_shifts_the_competitive_estimate() -> None:
    # The proxy->public calibration: below min_calibration_points pairs the estimate is the raw
    # proxy minus pessimism; at/above it, the mean proxy->public offset is applied (self-improving
    # as real submissions accrue). A focused unit on the gate's estimator.
    from verity.domains.feature_engineering_kaggle import _KaggleGates

    g = _KaggleGates(
        runner=None,  # type: ignore[arg-type]  # _calibrated_estimate never touches the runner
        scorer=FakeKaggleScorer(),
        agent_train_csv=b"", reserved_test_csv=b"", reserved_labels={},
        full_train_csv=b"", real_test_csv=b"",
        min_calibration_points=2, pessimism=0.1,
    )
    assert abs(g._calibrated_estimate(0.90) - 0.80) < 1e-9  # no pairs -> 0.90 - pessimism 0.1
    g._proxy_scores.update({"p1": 0.80, "p2": 0.90})
    g._real_scores.update({"p1": 0.75, "p2": 0.85})  # public ran ~0.05 below proxy
    assert abs(g._calibrated_estimate(0.90) - 0.85) < 1e-9  # 0.90 + offset(-0.05)
