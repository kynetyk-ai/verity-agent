"""Bounded subprocess output capture (audit V3)."""

from __future__ import annotations

import asyncio
import sys

from verity.proc import communicate_capped


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


async def _spawn(script: str) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable, "-c", script,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )


def test_capped_capture_truncates_a_flood_and_still_completes() -> None:
    # A child that floods stdout far past the cap: we retain ~cap bytes (plus a marker), drain the
    # rest so the child never blocks, and the call returns normally (no OOM, no deadlock).
    async def go() -> tuple[bytes, bytes]:
        proc = await _spawn("import sys; sys.stdout.write('x' * 5_000_000)")
        out, err = await communicate_capped(proc, cap=10_000)
        return out, err

    out, err = _run(go())
    assert b"[stdout truncated at 10000 bytes]" in out
    assert len(out) < 11_000  # retained ~cap, not the full 5 MB
    assert err == b""


def test_capped_capture_passes_short_output_through_untouched() -> None:
    async def go() -> tuple[bytes, bytes]:
        proc = await _spawn("import sys; sys.stdout.write('hello'); sys.stderr.write('warn')")
        return await communicate_capped(proc, cap=10_000)

    out, err = _run(go())
    assert out == b"hello" and err == b"warn"  # under the cap → verbatim, no marker
