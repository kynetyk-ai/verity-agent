"""The Kaggle scorer seam — the verifier's real-world final test (FE-Kaggle task).

The feature-engineering task's authoritative gate submits the regenerated predictions to a live
Kaggle competition and reads back the public-leaderboard score. That submit + poll lives behind this
narrow :class:`KaggleScorer` port so:

* the offline suite + CI run deterministically against :class:`FakeKaggleScorer` (no network, no
  creds), and
* the real ``RealKaggleScorer`` (the `kaggle` client) is exercised only by a ``@kaggle``/``@live``
  test — the daily submission cap is a shared, rate-limited resource.

Two methods, mirroring the two things the gate must know: how much submission budget remains today
(the cap is per team — the competition's own daily limit, read from its metadata), and the public
score of a submission once Kaggle has graded it. The
submit happens on the **trusted** side (the in-process verifier inside the control plane), never in
a worker — credentials must not reach untrusted agent code.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "KaggleScorer",
    "FakeKaggleScorer",
    "DAILY_SUBMISSION_LIMIT",
    "daily_limit_from",
    "top_fraction_threshold",
]

# Default fallback only: the real per-competition cap is read from the competition metadata
# (``max_daily_submissions``, see ``daily_limit_from`` / ``RealKaggleScorer._resolve_daily_limit``).
# Used when the metadata can't be read. The cap is per team, shared across all runs/days.
DAILY_SUBMISSION_LIMIT = 5


def daily_limit_from(competitions: Iterable[Any] | None, competition: str, default: int) -> int:
    """The competition's daily submission cap from a ``competitions_list`` result, else ``default``.

    Kaggle's ``ApiCompetition`` carries ``ref`` (the slug) and ``max_daily_submissions``. We match
    the slug exactly and trust the metadata only when it's a positive int; anything else (no match,
    missing, zero) falls back to ``default`` (the safe over-estimate — a too-high cap is
    self-correcting since Kaggle rejects over-cap submits).
    """
    for c in competitions or ():
        if getattr(c, "ref", None) == competition:
            limit = getattr(c, "max_daily_submissions", 0)
            return limit if isinstance(limit, int) and limit > 0 else default
    return default


def top_fraction_threshold(scores: Sequence[float], fraction: float) -> float | None:
    """The leaderboard score at the top-``fraction`` boundary (higher score = better standing).

    The "are we competitive?" bar: with ``fraction=0.10`` this is the score you must beat-or-match
    to sit in the **top 10%** of the standings. Scores are ranked descending and the boundary is the
    score at rank ``ceil(fraction * N)`` (1-indexed, clamped to ``[1, N]``). Returns ``None`` for an
    empty leaderboard so the caller can degrade (skip the bar) rather than invent a target. Assumes
    higher-is-better (true for balanced accuracy, the FE-Kaggle metric).
    """
    vals = sorted((float(s) for s in scores), reverse=True)
    n = len(vals)
    if n == 0:
        return None
    k = max(1, min(n, math.ceil(fraction * n)))
    return vals[k - 1]


@runtime_checkable
class KaggleScorer(Protocol):
    """The narrow port the Kaggle gate depends on. Bound to one competition at construction."""

    async def remaining_budget(self) -> int:
        """Submissions still allowed today (``limit - today's used``), read from the Kaggle API."""
        ...

    async def leaderboard_scores(self) -> list[float]:
        """Every public-leaderboard score (read-only — spends **no** submission budget).

        The competitive gate turns these into the top-N% target (:func:`top_fraction_threshold`).
        Returns an empty list when the leaderboard can't be read so the gate degrades (skips the
        competitive bar) rather than crashing; a genuine API/infra failure may also raise
        :class:`~verity.contracts.errors.GateUnavailable`.
        """
        ...

    async def submit_and_score(
        self, submission_csv: bytes, *, message: str, wait_deadline_s: float
    ) -> float:
        """Submit ``submission_csv`` and return the public-leaderboard score once graded.

        Polls until Kaggle finishes scoring (≈1–5 min). Raises
        :class:`~verity.contracts.errors.GateUnavailable` on an API/infra failure so the cycle
        degrades (recorded, fed back), not crashes.
        """
        ...


@dataclass
class FakeKaggleScorer:
    """A deterministic, offline :class:`KaggleScorer` for the suite (NON-PRODUCT).

    ``scores`` are returned in order, one per ``submit_and_score`` (the last repeats). ``budget`` is
    the standing remaining budget (decremented per submit); set ``budget_readings`` to script the
    successive ``remaining_budget`` returns instead (e.g. ``[0, 0, 3]`` to exercise the
    block-until-budget-frees path). ``fail_with`` makes ``submit_and_score`` raise (the degrade
    case). ``submitted`` records the bytes submitted so a test can assert how many calls were made.
    ``leaderboard`` scripts the standings :meth:`leaderboard_scores` returns (the competitive bar's
    raw input).
    """

    scores: Sequence[float] = (1.0,)
    budget: int = DAILY_SUBMISSION_LIMIT
    budget_readings: Sequence[int] | None = None
    fail_with: Exception | None = None
    leaderboard: Sequence[float] = ()
    submitted: list[bytes] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    _score_i: int = 0
    _budget_i: int = 0

    async def remaining_budget(self) -> int:
        if self.budget_readings is not None:
            reading = self.budget_readings[min(self._budget_i, len(self.budget_readings) - 1)]
            self._budget_i += 1
            return reading
        return self.budget

    async def leaderboard_scores(self) -> list[float]:
        return [float(s) for s in self.leaderboard]

    async def submit_and_score(
        self, submission_csv: bytes, *, message: str, wait_deadline_s: float
    ) -> float:
        if self.fail_with is not None:
            raise self.fail_with
        self.submitted.append(submission_csv)
        self.messages.append(message)
        self.budget = max(0, self.budget - 1)
        score = self.scores[min(self._score_i, len(self.scores) - 1)]
        self._score_i += 1
        return score
