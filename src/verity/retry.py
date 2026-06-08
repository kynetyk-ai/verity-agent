"""A small async retry helper — bounded exponential backoff with jitter (ROADMAP 5.1).

Transient failures (a model rate-limit/5xx, a flaky daemon spawn) should be retried a few times
before they become a recovered failed cycle, not abort on the first blip. This is the one shared
retry primitive both services use; it takes no dependency on either, and ``sleep`` / ``jitter`` are
injectable so tests are deterministic and instant (no real waiting, no randomness).

Deliberately tiny: no ``tenacity`` dependency, no decorator magic — a function you wrap a coroutine
factory with. ``retry_on`` names exactly which exception types are transient; anything else raises
straight through (a bug or misconfiguration is not something to retry).
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from verity.logging import get_logger

__all__ = ["RetryPolicy", "retry_async"]

log = get_logger("verity.retry")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many times and how long to back off. ``attempts`` is the *total* tries (1 = no retry)."""

    attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    multiplier: float = 2.0
    jitter_s: float = 0.25

    def delay_for(self, attempt: int, *, jitter: float) -> float:
        """The backoff before retry ``attempt`` (1-based): exponential, capped, plus ``jitter``."""
        base = self.base_delay_s * (self.multiplier ** (attempt - 1))
        return min(self.max_delay_s, base) + jitter


def _default_jitter(jitter_s: float) -> float:
    return random.uniform(0.0, jitter_s)


async def retry_async[T](
    factory: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    retry_on: tuple[type[BaseException], ...],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    jitter: Callable[[float], float] = _default_jitter,
    label: str = "operation",
) -> T:
    """Call ``factory()`` with retries on ``retry_on``; re-raise the last error once tries run out.

    ``factory`` is a no-arg coroutine *factory* (called afresh each attempt), not a coroutine, so
    the awaitable is rebuilt per try. Only ``retry_on`` exceptions are retried; everything else
    propagates immediately. On exhaustion the final ``retry_on`` exception is re-raised unchanged,
    so the caller can wrap it in a domain error (e.g. ``GateUnavailable``).
    """
    last_exc: BaseException | None = None
    for attempt in range(1, policy.attempts + 1):
        try:
            return await factory()
        except retry_on as exc:
            last_exc = exc
            if attempt >= policy.attempts:
                break
            delay = policy.delay_for(attempt, jitter=jitter(policy.jitter_s))
            log.warning(
                "retrying", label=label, attempt=attempt, attempts=policy.attempts,
                error=str(exc), delay_s=round(delay, 3),
            )
            await sleep(delay)
    assert last_exc is not None  # the loop only breaks after catching at least one retry_on
    raise last_exc
