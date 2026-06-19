"""The live :class:`KaggleScorer` — submit to a real Kaggle competition + read the public score.

Isolated in its own module so the `kaggle` client is imported **lazily** (only when a real
submission happens): the package import stays clean for the lean core / CI, which never install the
`kaggle` extra. The client is synchronous, so each call runs in a worker thread; scoring is polled
(≈1–5 min). Every Kaggle/IO failure is surfaced as a recoverable :class:`GateUnavailable` so a cycle
degrades (recorded, fed back) rather than crashing the run.

Auth comes from the standard Kaggle locations (``~/.kaggle/kaggle.json`` or
``KAGGLE_USERNAME``+``KAGGLE_KEY``), read on the **trusted** control-plane side — never a worker.
The competition's rules must be accepted on its Kaggle page once, or the API returns 403.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from verity.contracts import GateUnavailable
from verity.logging import get_logger
from verity.verifier.kaggle import DAILY_SUBMISSION_LIMIT, daily_limit_from

__all__ = ["RealKaggleScorer"]

log = get_logger("verity.verifier.kaggle_client")


@dataclass
class RealKaggleScorer:
    """Submit to ``competition`` and read back the public leaderboard score (the fe-kaggle gate)."""

    competition: str
    submission_filename: str = "submission.csv"
    poll_interval_s: float = 20.0
    daily_limit_override: int | None = None  # explicit cap; wins over competition metadata
    _api: Any = field(default=None, init=False, repr=False)
    _daily_limit: int | None = field(default=None, init=False, repr=False)

    def _client(self) -> Any:
        if self._api is None:
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi  # lazy: only the live path
            except ImportError as exc:  # the `kaggle` extra is not installed
                raise GateUnavailable("the Kaggle client is not installed (kaggle extra)") from exc
            api = KaggleApi()
            try:
                api.authenticate()  # reads ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY
            except Exception as exc:  # noqa: BLE001 — any auth failure is infra, degrade
                raise GateUnavailable(f"Kaggle authentication failed: {exc}") from exc
            self._api = api
        return self._api

    async def remaining_budget(self) -> int:
        """``limit - today's submissions`` from the API (the daily cap is shared per team).

        The cap is the competition's own ``max_daily_submissions`` (read once, then cached); the
        used-count is Kaggle's real submission history, so this never drifts from the truth.
        """
        api = self._client()
        try:
            subs = await asyncio.to_thread(api.competition_submissions, self.competition)
        except Exception as exc:  # noqa: BLE001 — API/IO failure → recoverable
            raise GateUnavailable(f"could not read Kaggle submissions: {exc}") from exc
        today = datetime.now(UTC).date()
        used = sum(1 for s in subs if _submitted_on(s) == today)
        limit = await self._resolve_daily_limit()
        remaining = max(0, limit - used)
        log.info(
            "kaggle_budget", competition=self.competition, used=used,
            limit=limit, remaining=remaining,
        )
        return remaining

    async def _resolve_daily_limit(self) -> int:
        """The competition's daily submission cap, resolved once and cached.

        Precedence: explicit ``daily_limit_override`` → competition metadata
        (``max_daily_submissions``) → the ``DAILY_SUBMISSION_LIMIT`` fallback. A metadata-read
        failure degrades to the fallback (logged), never raises — the budget poll calls this every
        tick, so it must be cheap (cached) and must not turn a transient API blip into a hard error.
        """
        if self._daily_limit is not None:
            return self._daily_limit
        if self.daily_limit_override is not None:
            self._daily_limit = self.daily_limit_override
            log.info(
                "kaggle_daily_limit", competition=self.competition,
                limit=self._daily_limit, source="override",
            )
            return self._daily_limit
        api = self._client()
        try:
            resp = await asyncio.to_thread(api.competitions_list, search=self.competition)
            competitions = getattr(resp, "competitions", None)
            limit = daily_limit_from(competitions, self.competition, DAILY_SUBMISSION_LIMIT)
            source = "metadata" if limit != DAILY_SUBMISSION_LIMIT else "metadata-or-fallback"
        except Exception as exc:  # noqa: BLE001 — metadata read is best-effort; degrade to fallback
            limit = DAILY_SUBMISSION_LIMIT
            source = "fallback"
            log.warning(
                "kaggle_daily_limit_unavailable", competition=self.competition,
                error=str(exc), limit=limit,
            )
        self._daily_limit = limit
        log.info("kaggle_daily_limit", competition=self.competition, limit=limit, source=source)
        return limit

    async def submit_and_score(
        self, submission_csv: bytes, *, message: str, wait_deadline_s: float
    ) -> float:
        api = self._client()
        with tempfile.TemporaryDirectory(prefix="verity-kaggle-") as tmp:
            path = Path(tmp) / self.submission_filename
            path.write_bytes(submission_csv)
            try:
                await asyncio.to_thread(
                    api.competition_submit, str(path), message, self.competition
                )
            except Exception as exc:  # noqa: BLE001 — submit failure (incl. 403/429) → recoverable
                raise GateUnavailable(f"Kaggle submission failed: {exc}") from exc
        log.info("kaggle_submitted", competition=self.competition, message=message)
        return await self._poll_score(api, wait_deadline_s)

    async def _poll_score(self, api: Any, wait_deadline_s: float) -> float:
        """Poll until our (newest) submission is graded; the run lock serializes our submits, so the
        most recent submission is ours."""
        waited = 0.0
        while True:
            try:
                subs = await asyncio.to_thread(api.competition_submissions, self.competition)
            except Exception as exc:  # noqa: BLE001
                raise GateUnavailable(f"could not poll the Kaggle score: {exc}") from exc
            latest = _newest(subs)
            score = _public_score(latest)
            if score is not None:
                log.info("kaggle_scored", competition=self.competition, public_score=score)
                return score
            status = str(getattr(latest, "status", "")).lower()
            if "error" in status or "fail" in status:
                raise GateUnavailable(f"Kaggle scoring failed (status={status!r})")
            if waited >= wait_deadline_s:
                raise GateUnavailable(f"Kaggle did not score within {wait_deadline_s}s")
            await asyncio.sleep(self.poll_interval_s)
            waited += self.poll_interval_s


def _submitted_on(submission: Any) -> Any:
    d = getattr(submission, "date", None)
    return d.date() if isinstance(d, datetime) else None


def _newest(subs: list[Any]) -> Any | None:
    if not subs:
        return None
    dated = [s for s in subs if isinstance(getattr(s, "date", None), datetime)]
    if dated:
        return max(dated, key=lambda s: s.date)
    return subs[0]  # the API returns newest-first when dates are unavailable


def _public_score(submission: Any) -> float | None:
    if submission is None:
        return None
    raw: Any = getattr(submission, "publicScore", None)
    if raw is None or raw == "":
        raw = getattr(submission, "public_score", None)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
