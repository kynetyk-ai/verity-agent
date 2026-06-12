"""Live Kaggle scorer checks (``@kaggle``/``@live``) — auto-skip without the client + creds.

Two levels, both real-network:
* **budget (safe, read-only):** authenticate and read the remaining daily budget from the API. This
  also confirms the competition's rules are accepted (the API 403s otherwise). Spends nothing.
* **submit (opt-in, spends 1 of the ~5/day):** gated behind ``VERITY_KAGGLE_SUBMIT=1`` and the real
  ``test.csv`` — submits a trivial constant-class submission and reads back its public score,
  proving the full submit→poll path end to end.

The full fe-kaggle task through the daemon on a local model is a manual run (see the package
PROTOCOL), not a unit test. Set creds in ``.env`` (``KAGGLE_USERNAME``/``KAGGLE_KEY`` or
``~/.kaggle/kaggle.json``) and accept the competition rules on its Kaggle page first.
"""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import io
import os
from pathlib import Path

import pytest

from verity.contracts import GateUnavailable
from verity.verifier.kaggle import DAILY_SUBMISSION_LIMIT

_COMPETITION = os.environ.get("KAGGLE_COMPETITION", "playground-series-s6e6")
_DATASET = Path("feature-engineering-test/test.csv")


def _kaggle_ready() -> tuple[bool, str]:
    if importlib.util.find_spec("kaggle") is None:  # check without importing (no import-time auth)
        return False, "the `kaggle` client is not installed (the `kaggle` extra)"
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True, ""
    if (Path.home() / ".kaggle" / "kaggle.json").is_file():
        return True, ""
    return False, "no Kaggle creds (KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json)"


_READY, _WHY = _kaggle_ready()


@pytest.mark.kaggle
@pytest.mark.live
@pytest.mark.skipif(not _READY, reason=_WHY)
def test_remaining_budget_is_readable() -> None:
    """Authenticate + read the daily budget (no submission spent). Skips if the API is unavailable —
    most often because the competition's rules have not been accepted yet."""
    from verity.verifier.kaggle_client import RealKaggleScorer

    scorer = RealKaggleScorer(competition=_COMPETITION)
    try:
        budget = asyncio.run(scorer.remaining_budget())
    except GateUnavailable as exc:
        pytest.skip(f"Kaggle API unavailable (accepted the competition rules?): {exc}")
    assert isinstance(budget, int)
    assert 0 <= budget <= DAILY_SUBMISSION_LIMIT


@pytest.mark.kaggle
@pytest.mark.live
@pytest.mark.skipif(not _READY, reason=_WHY)
@pytest.mark.skipif(
    os.environ.get("VERITY_KAGGLE_SUBMIT") != "1",
    reason="opt-in: set VERITY_KAGGLE_SUBMIT=1 to spend one of the ~5/day submissions",
)
@pytest.mark.skipif(not _DATASET.exists(), reason="the real test.csv is not present")
def test_submit_and_score_round_trip() -> None:
    """Submit a trivial constant-class submission and read back its public score (spends 1/day)."""
    from verity.verifier.kaggle_client import RealKaggleScorer

    reader = csv.DictReader(io.StringIO(_DATASET.read_bytes().decode("utf-8")))
    submission = "id,class\n" + "\n".join(f"{row['id']},GALAXY" for row in reader) + "\n"

    scorer = RealKaggleScorer(competition=_COMPETITION, poll_interval_s=15.0)
    score = asyncio.run(
        scorer.submit_and_score(
            submission.encode("utf-8"),
            message="verity fe-kaggle live test",
            wait_deadline_s=600.0,
        )
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0  # balanced accuracy
