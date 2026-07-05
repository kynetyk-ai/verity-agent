"""The agent transcript/telemetry as **structured log events**, not agent-visible files.

Why this exists (integrity): the worker runs the agent loop and its shell tool as the **same uid in
the same container** (a single non-root uid, ``--cap-drop=ALL --no-new-privileges`` — no second
identity to drop to). So any *file* the driver writes, the agent's shell can read, edit, or delete —
the diagnostics record would not be trustworthy. Instead the driver **logs each turn** (+ a final
telemetry event) to the process's stderr, which the host captures *outside* the container. The agent
can't enumerate, read, or erase a log line already streamed out (it could, deliberately, *inject*
fake lines via ``/proc/<pid>/fd/2`` — pollution, not erasure; a documented residual). No transcript
or telemetry file ever exists in the agent's filesystem.

This module is the shared contract between the **producer** (``deepagents_driver`` in the worker,
which renders + logs each message) and the **consumer** (``backend_driver`` on the host, which
reconstructs the transcript/telemetry from the worker's captured stderr and writes them into the
*host* workspace outbox — after the container has exited — so ``AgentSandbox``'s harvest is
unchanged). It imports no ``deepagents`` code, so the host side stays light.
"""

from __future__ import annotations

import json
from typing import Any, cast

__all__ = [
    "TURN_EVENT",
    "TELEMETRY_EVENT",
    "TRANSCRIPT_FIELD_CAP",
    "cap_field",
    "render_message",
    "parse_log_events",
    "reconstruct_from_logs",
    "telemetry_from_turns",
]

#: structlog ``event`` names the producer emits and the consumer filters on.
TURN_EVENT = "agent_turn"
TELEMETRY_EVENT = "agent_telemetry"

#: Per-field truncation so a runaway tool dump can't bloat a single turn's log line.
TRANSCRIPT_FIELD_CAP = 20_000

# structlog/log-framework keys that ride on every line — not part of the transcript/telemetry data.
_LOG_META = frozenset({"event", "level", "timestamp", "logger", "logger_name"})
# The transcript-entry fields the producer renders (everything else on a turn line is log meta).
_ENTRY_FIELDS = ("index", "type", "content", "tool_calls", "tool_name", "usage", "model")


def cap_field(value: Any, limit: int = TRANSCRIPT_FIELD_CAP) -> Any:
    """Truncate an overlong rendered field; pass short values through unchanged."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= limit:
        return value
    return text[:limit] + f"…[truncated {len(text) - limit} chars]"


def render_message(index: int, message: Any) -> dict[str, Any]:
    """Render one loop message to a JSON-safe transcript entry (duck-typed; no deepagents import).

    Captures the message ``index``, its ``type``, the ``content``, any tool calls (name **+**
    capped args), and — for a ``ToolMessage`` — the tool ``name`` (so a tool *result* is legible).
    An AI message additionally carries its per-turn ``usage`` (input/output tokens) and ``model``,
    so the end-of-run telemetry aggregate can be **reconstructed from the streamed turns** when a
    hard kill (timeout / OOM) prevents the worker from ever emitting it (#125). The whole entry is
    round-tripped through ``json`` with ``default=str`` so the structured logger can never choke
    on a non-serializable field.
    """
    tool_calls = [
        {"name": tc.get("name"), "args": cap_field(tc.get("args"))}
        for tc in (getattr(message, "tool_calls", None) or [])
    ]
    entry: dict[str, Any] = {
        "index": index,
        "type": type(message).__name__,
        "content": cap_field(getattr(message, "content", None)),
    }
    if tool_calls:
        entry["tool_calls"] = tool_calls
    if type(message).__name__ == "ToolMessage" and getattr(message, "name", None) is not None:
        entry["tool_name"] = message.name
    if type(message).__name__ == "AIMessage":
        usage = getattr(message, "usage_metadata", None)
        if usage:
            entry["usage"] = {
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
            }
        meta = getattr(message, "response_metadata", None) or {}
        model = meta.get("model_name") or meta.get("model")
        if model:
            entry["model"] = model
    # Guarantee JSON-safety for the structured logger (and for the host's later re-serialization).
    return cast("dict[str, Any]", json.loads(json.dumps(entry, default=str)))


def parse_log_events(stderr: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Parse a worker's captured stderr into ``(turns, telemetry)``.

    Collects every :data:`TURN_EVENT` entry (ordered by ``index``) and the last
    :data:`TELEMETRY_EVENT`. Non-JSON lines and unrelated events are ignored, so this is robust to
    interleaved framework logs. ``telemetry`` is ``None`` when the worker never emitted its
    end-of-run event (a hard kill) — the caller decides whether to reconstruct one from the turns
    (:func:`telemetry_from_turns`).
    """
    turns: list[dict[str, Any]] = []
    telemetry: dict[str, Any] | None = None
    for line in stderr.splitlines():
        line = line.strip()
        if not line or line[0] != "{":
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        event = obj.get("event")
        if event == TURN_EVENT:
            entry = {k: obj[k] for k in _ENTRY_FIELDS if k in obj}
            if "index" in entry:
                turns.append(entry)
        elif event == TELEMETRY_EVENT:
            telemetry = {k: v for k, v in obj.items() if k not in _LOG_META}
    turns.sort(key=lambda e: e.get("index", 0))
    return turns, telemetry


def telemetry_from_turns(turns: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Reconstruct the telemetry aggregate from streamed turn entries (#125).

    Mirrors the worker's ``_extract_telemetry`` field contract (input/output/total tokens,
    ``model_steps``, ``tool_calls``, ``model``) by summing the per-turn ``usage`` captured by
    :func:`render_message` — tagged ``"source": "reconstructed_from_turns"`` so a rebuilt record
    is never mistaken for a worker-emitted measurement. Token sums undercount for turns produced
    before per-turn usage existed (or by models that report none); steps/tool-calls are exact.
    ``None`` when there are no turns to reconstruct from.
    """
    if not turns:
        return None
    input_tokens = output_tokens = model_steps = tool_calls = 0
    model: Any = None
    for entry in turns:
        if entry.get("type") != "AIMessage":
            continue
        model_steps += 1
        tool_calls += len(entry.get("tool_calls") or [])
        usage = entry.get("usage") or {}
        input_tokens += int(usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("output_tokens", 0) or 0)
        model = entry.get("model") or model
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "model_steps": model_steps,
        "tool_calls": tool_calls,
        "model": model,
        "source": "reconstructed_from_turns",
    }


def reconstruct_from_logs(stderr: str) -> tuple[bytes | None, bytes | None]:
    """Rebuild ``(transcript_bytes, telemetry_bytes)`` from a worker's captured stderr log stream.

    A thin serializer over :func:`parse_log_events`. Returns ``(None, None)`` for whatever is
    absent — a run that never logged a turn (e.g. died before the first model step) yields no
    transcript, and the caller treats that like any other missing-diagnostics case.
    """
    turns, telemetry = parse_log_events(stderr)
    transcript_bytes = (
        json.dumps(turns, default=str).encode("utf-8") if turns else None
    )
    telemetry_bytes = (
        json.dumps(telemetry, default=str).encode("utf-8") if telemetry is not None else None
    )
    return transcript_bytes, telemetry_bytes
