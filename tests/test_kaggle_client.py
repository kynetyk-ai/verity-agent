"""Offline checks for the live Kaggle scorer's budget math — no network, no creds.

`RealKaggleScorer` talks to the real `kaggle` client, but the daily-cap logic is pure: the limit is
read once from competition metadata (cached), the used-count from the API, and ``remaining_budget``
is their difference. We exercise that by unit-testing the ``daily_limit_from`` helper and by
injecting a stub into the scorer's ``_api`` (which bypasses ``_client()``/auth entirely).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from verity.contracts import GateUnavailable
from verity.verifier.kaggle import DAILY_SUBMISSION_LIMIT, daily_limit_from
from verity.verifier.kaggle_client import RealKaggleScorer


def _comp(ref: str, max_daily: object) -> SimpleNamespace:
    """A duck-typed stand-in for the SDK's ``ApiCompetition`` (ref + max_daily_submissions)."""
    return SimpleNamespace(ref=ref, max_daily_submissions=max_daily)


# --- daily_limit_from (pure) ------------------------------------------------------------------


def test_daily_limit_from_uses_matching_competition_metadata() -> None:
    comps = [_comp("other", 9), _comp("playground-s6e6", 2)]
    assert daily_limit_from(comps, "playground-s6e6", DAILY_SUBMISSION_LIMIT) == 2


def test_daily_limit_from_falls_back_when_metadata_is_zero_or_missing() -> None:
    assert daily_limit_from([_comp("c", 0)], "c", 5) == 5  # zero is "unset" → fallback
    assert daily_limit_from([_comp("c", None)], "c", 5) == 5  # non-int → fallback


def test_daily_limit_from_falls_back_when_no_competition_matches() -> None:
    assert daily_limit_from([_comp("other", 3)], "wanted", 5) == 5
    assert daily_limit_from(None, "wanted", 5) == 5  # empty/None result


# --- RealKaggleScorer.remaining_budget (stubbed API, no network) -------------------------------


class _StubApi:
    """A minimal stand-in for the authenticated Kaggle client, recording call counts."""

    def __init__(
        self, *, max_daily: int, submitted_today: int, list_error: Exception | None = None
    ):
        self._max_daily = max_daily
        self._list_error = list_error
        now = datetime.now(UTC)
        self._subs = [
            SimpleNamespace(date=now - timedelta(hours=i)) for i in range(submitted_today)
        ]
        self.list_calls = 0
        self.submissions_calls = 0

    def competitions_list(self, search: str | None = None):
        self.list_calls += 1
        if self._list_error is not None:
            raise self._list_error
        return SimpleNamespace(competitions=[_comp(search or "", self._max_daily)])

    def competition_submissions(self, _competition: str):
        self.submissions_calls += 1
        return list(self._subs)


def _scorer(api: _StubApi, **kw: object) -> RealKaggleScorer:
    s = RealKaggleScorer(competition="comp", **kw)  # type: ignore[arg-type]
    s._api = api  # inject the stub → _client() short-circuits, never authenticates
    return s


def test_remaining_budget_is_metadata_limit_minus_used() -> None:
    api = _StubApi(max_daily=2, submitted_today=1)
    scorer = _scorer(api)
    assert asyncio.run(scorer.remaining_budget()) == 1  # cap 2 − 1 used today


def test_daily_limit_is_resolved_once_and_cached() -> None:
    api = _StubApi(max_daily=4, submitted_today=0)
    scorer = _scorer(api)
    for _ in range(3):  # the budget-poll loop calls remaining_budget repeatedly
        assert asyncio.run(scorer.remaining_budget()) == 4
    assert api.list_calls == 1  # metadata read exactly once despite three budget reads
    assert api.submissions_calls == 3  # used-count IS re-read each time (it changes)


def test_override_wins_over_metadata_and_skips_the_lookup() -> None:
    api = _StubApi(max_daily=2, submitted_today=0)
    scorer = _scorer(api, daily_limit_override=7)
    assert asyncio.run(scorer.remaining_budget()) == 7
    assert api.list_calls == 0  # override → no competitions_list call at all


def test_metadata_read_failure_degrades_to_the_fallback() -> None:
    api = _StubApi(max_daily=2, submitted_today=0, list_error=RuntimeError("kaggle down"))
    scorer = _scorer(api)
    # The metadata read fails, but the budget read still succeeds on the fallback cap (not a crash).
    assert asyncio.run(scorer.remaining_budget()) == DAILY_SUBMISSION_LIMIT


def test_submissions_read_failure_is_recoverable() -> None:
    api = _StubApi(max_daily=2, submitted_today=0)

    def _boom(_competition: str):
        raise RuntimeError("transient")

    api.competition_submissions = _boom  # type: ignore[method-assign]
    scorer = _scorer(api)
    with pytest.raises(GateUnavailable):
        asyncio.run(scorer.remaining_budget())
