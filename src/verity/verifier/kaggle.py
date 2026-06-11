"""The Kaggle scorer seam — the verifier's real-world final test (FE-Kaggle task).

The feature-engineering task's authoritative gate submits the regenerated predictions to a live
Kaggle competition and reads back the public-leaderboard score. That submit + poll lives behind this
narrow :class:`KaggleScorer` port so:

* the offline suite + CI run deterministically against :class:`FakeKaggleScorer` (no network, no
  creds), and
* the real ``RealKaggleScorer`` (the `kaggle` client) is exercised only by a ``@kaggle``/``@live``
  test — the daily 5-submission cap is a shared, rate-limited resource.

Two methods, mirroring the two things the gate must know: how much submission budget remains today
(the cap is per team, ~5/day), and the public score of a submission once Kaggle has graded it. The
submit happens on the **trusted** side (the in-process verifier inside the control plane), never in
a worker — credentials must not reach untrusted agent code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["KaggleScorer", "FakeKaggleScorer", "DAILY_SUBMISSION_LIMIT"]

# Kaggle Playground Series cap: ~5 submissions/day per team, shared across all runs/days.
DAILY_SUBMISSION_LIMIT = 5


@runtime_checkable
class KaggleScorer(Protocol):
    """The narrow port the Kaggle gate depends on. Bound to one competition at construction."""

    async def remaining_budget(self) -> int:
        """Submissions still allowed today (``limit - today's used``), read from the Kaggle API."""
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
    """

    scores: Sequence[float] = (1.0,)
    budget: int = DAILY_SUBMISSION_LIMIT
    budget_readings: Sequence[int] | None = None
    fail_with: Exception | None = None
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
