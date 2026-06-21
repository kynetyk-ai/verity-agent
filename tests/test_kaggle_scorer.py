"""The Kaggle scorer seam's pure pieces — the top-N% leaderboard threshold + the fake's read.

Offline, no network/creds: ``top_fraction_threshold`` is the competitive gate's bar (the score you
must beat-or-match to sit in the top fraction of the standings), and ``FakeKaggleScorer`` scripts
the leaderboard the gate reads. The live ``competition_leaderboard_view`` path is exercised only by
the read-only ``@kaggle`` test in ``test_fe_kaggle_live.py``.
"""

from __future__ import annotations

import asyncio

import pytest

from verity.verifier.kaggle import FakeKaggleScorer, top_fraction_threshold


def test_threshold_picks_the_top_rank_boundary() -> None:
    scores = [0.5, 0.9, 0.7, 0.8, 0.6]  # unsorted on purpose — the helper ranks descending
    assert top_fraction_threshold(scores, 0.10) == 0.9  # ceil(0.5)=1 -> the single best
    assert top_fraction_threshold(scores, 0.40) == 0.8  # ceil(2.0)=2 -> 2nd best
    assert top_fraction_threshold(scores, 0.50) == 0.7  # ceil(2.5)=3 -> 3rd best
    assert top_fraction_threshold(scores, 1.00) == 0.5  # the whole field -> the worst score


def test_threshold_degrades_on_an_empty_leaderboard() -> None:
    assert top_fraction_threshold([], 0.10) is None


def test_threshold_clamps_a_singleton() -> None:
    assert top_fraction_threshold([0.42], 0.10) == 0.42


def test_fake_leaderboard_scores_round_trip() -> None:
    scorer = FakeKaggleScorer(leaderboard=[0.95, 0.91, 0.80])
    assert asyncio.run(scorer.leaderboard_scores()) == [0.95, 0.91, 0.80]
    # default: no leaderboard scripted -> empty (the gate then skips the competitive bar)
    assert asyncio.run(FakeKaggleScorer().leaderboard_scores()) == []


@pytest.mark.parametrize("frac", [0.10, 0.25, 0.9])
def test_being_at_threshold_means_in_the_fraction(frac: float) -> None:
    # Property: the count of scores >= the top-`frac` threshold is at least ceil(frac*N) -> a model
    # that reaches the threshold genuinely sits within the top `frac` of the standings.
    import math

    scores = [i / 100 for i in range(100)]  # 0.00 .. 0.99
    thr = top_fraction_threshold(scores, frac)
    assert thr is not None
    in_band = sum(1 for s in scores if s >= thr)
    assert in_band >= math.ceil(frac * len(scores))
