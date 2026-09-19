"""Bounded subprocess output capture (audit V3).

``asyncio.subprocess`` ``communicate()`` reads a child's entire stdout/stderr into memory. The
verifier runs *agent-submitted* code, so a runaway submission (``while True: print('x'*10**6)``)
could stream gigabytes through the pipe and OOM the verifier host mid-sweep — the container memory
cap bounds the *container*, not the parent's read buffer. :func:`communicate_capped` retains at most
a fixed budget per stream while still **draining** the rest (so the child never blocks on a full
pipe), appending a truncation marker. The tail is what the gates' error messages need anyway.
"""

from __future__ import annotations

import asyncio

__all__ = ["communicate_capped", "drain_capped", "DEFAULT_OUTPUT_CAP"]

# Per-stream retained-byte budget. Generous for legitimate pip/training chatter, but a hard ceiling
# on what a runaway child can make the host hold in memory.
DEFAULT_OUTPUT_CAP = 1_000_000
_READ_CHUNK = 65_536


async def drain_capped(stream: asyncio.StreamReader | None, cap: int) -> tuple[bytes, bool]:
    """Read ``stream`` to EOF, retaining at most ``cap`` bytes; report whether any were dropped.

    Public so a caller that needs timeout-safe capture (e.g. the worker backend, which must retain
    whatever was read *before* a wall-clock kill) can own the drain tasks itself rather than wrap
    :func:`communicate_capped` — wrapping it in ``wait_for`` would cancel the drains and lose the
    bytes already buffered, exactly the partial output a timed-out worker most needs.
    """
    if stream is None:
        return b"", False
    buf = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(_READ_CHUNK)
        if not chunk:
            return bytes(buf), truncated
        room = cap - len(buf)
        if room > 0:
            buf.extend(chunk[:room])
        if len(chunk) > max(room, 0):
            truncated = True  # bytes past the cap were read + discarded (the pipe stays drained)


async def communicate_capped(
    proc: asyncio.subprocess.Process, *, cap: int = DEFAULT_OUTPUT_CAP
) -> tuple[bytes, bytes]:
    """Like ``proc.communicate()`` but retains at most ``cap`` bytes per stream (draining the rest).

    Drains both pipes concurrently then awaits the child's exit, so a child that floods its output
    cannot OOM the host or deadlock on a full pipe. A truncation marker is appended to a capped
    stream so the caller can tell it was clipped.
    """
    out_task = asyncio.ensure_future(drain_capped(proc.stdout, cap))
    err_task = asyncio.ensure_future(drain_capped(proc.stderr, cap))
    out, out_truncated = await out_task
    err, err_truncated = await err_task
    await proc.wait()
    if out_truncated:
        out += b"\n...[stdout truncated at %d bytes]" % cap
    if err_truncated:
        err += b"\n...[stderr truncated at %d bytes]" % cap
    return out, err
