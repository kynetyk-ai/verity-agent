"""The shared async retry helper (ROADMAP 5.1).

Deterministic and instant: ``sleep`` is a recorder (no real waiting) and ``jitter`` is fixed, so the
backoff schedule is asserted exactly. Covers the success, retry-then-succeed, exhaustion, and
don't-retry-the-wrong-error paths.
"""

from __future__ import annotations

import asyncio

import pytest

from verity.retry import RetryPolicy, retry_async


class _Transient(Exception):
    pass


class _Fatal(Exception):
    pass


def _recorder() -> tuple[list[float], object]:
    delays: list[float] = []

    async def sleep(d: float) -> None:
        delays.append(d)

    return delays, sleep


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def test_returns_immediately_on_success_without_sleeping() -> None:
    delays, sleep = _recorder()
    calls = 0

    async def factory() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    out = _run(retry_async(
        factory, policy=RetryPolicy(), retry_on=(_Transient,), sleep=sleep, jitter=lambda _j: 0.0,
    ))
    assert out == "ok" and calls == 1 and delays == []


def test_retries_transient_then_succeeds_with_capped_exponential_backoff() -> None:
    delays, sleep = _recorder()
    attempts = 0

    async def factory() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            raise _Transient(f"blip {attempts}")
        return "recovered"

    policy = RetryPolicy(attempts=5, base_delay_s=1.0, max_delay_s=3.0, multiplier=2.0)
    out = _run(retry_async(
        factory, policy=policy, retry_on=(_Transient,), sleep=sleep, jitter=lambda _j: 0.0,
    ))
    # delays for attempts 1,2,3: 1, 2, min(4,3)=3 -> capped at max_delay_s
    assert out == "recovered" and attempts == 4 and delays == [1.0, 2.0, 3.0]


def test_jitter_is_added_to_each_backoff() -> None:
    delays, sleep = _recorder()

    async def factory() -> str:
        raise _Transient("always")

    policy = RetryPolicy(attempts=3, base_delay_s=1.0, multiplier=2.0)
    with pytest.raises(_Transient):
        _run(retry_async(
            factory, policy=policy, retry_on=(_Transient,), sleep=sleep, jitter=lambda _j: 0.5,
        ))
    assert delays == [1.5, 2.5]  # base 1 + 0.5, base 2 + 0.5; no sleep after the final failure


def test_exhaustion_reraises_the_last_transient_error() -> None:
    _delays, sleep = _recorder()
    seen = 0

    async def factory() -> str:
        nonlocal seen
        seen += 1
        raise _Transient(f"fail {seen}")

    with pytest.raises(_Transient, match="fail 3"):
        _run(retry_async(
            factory, policy=RetryPolicy(attempts=3), retry_on=(_Transient,),
            sleep=sleep, jitter=lambda _j: 0.0,
        ))
    assert seen == 3  # exactly `attempts` tries


def test_a_non_retryable_error_propagates_immediately() -> None:
    delays, sleep = _recorder()
    seen = 0

    async def factory() -> str:
        nonlocal seen
        seen += 1
        raise _Fatal("not transient")

    with pytest.raises(_Fatal):
        _run(retry_async(
            factory, policy=RetryPolicy(attempts=5), retry_on=(_Transient,),
            sleep=sleep, jitter=lambda _j: 0.0,
        ))
    assert seen == 1 and delays == []  # tried once, never retried, never slept
