"""The transcript-via-logs contract: render → log-line → host reconstruction (no agent file).

These exercise :mod:`verity.sandbox.log_transcript`, the deepagents-free seam shared by the worker
(produces ``agent_turn`` / ``agent_telemetry`` events) and the host (reconstructs both from captured
stderr). The integrity property under test: the record survives entirely outside the agent's
filesystem, and reconstruction is robust to the noise — and the *injection* — an agent could add.
"""

from __future__ import annotations

import json

from verity.sandbox.log_transcript import (
    TELEMETRY_EVENT,
    TRANSCRIPT_FIELD_CAP,
    TURN_EVENT,
    reconstruct_from_logs,
    render_message,
)


class _Msg:
    """A duck-typed loop message (no langchain import needed for the contract test)."""

    def __init__(self, content: str, *, tool_calls: list[dict] | None = None,
                 name: str | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        if name is not None:
            self.name = name


def _log_line(event: str, **fields: object) -> str:
    """Mimic one structlog JSON line (the producer's fields + the framework's meta keys)."""
    return json.dumps({"event": event, "level": "info", "timestamp": "t", **fields})


def test_round_trip_reconstructs_transcript_and_telemetry() -> None:
    # Producer renders each message; the host reconstructs the ordered transcript from the lines.
    msgs = [
        render_message(0, _Msg("plan", tool_calls=[{"name": "execute", "args": {"cmd": "go"}}])),
        render_message(1, _Msg("ok", name="execute")),
    ]
    stderr = "\n".join(
        [
            "some framework startup line (not JSON)",
            _log_line(TURN_EVENT, **msgs[0]),
            _log_line("unrelated_event", foo="bar"),
            _log_line(TURN_EVENT, **msgs[1]),
            _log_line(TELEMETRY_EVENT, input_tokens=10, output_tokens=2, model_steps=2,
                      stop_reason="StepBudgetExceeded: ..."),
        ]
    )
    transcript_bytes, telemetry_bytes = reconstruct_from_logs(stderr)
    transcript = json.loads(transcript_bytes)
    telemetry = json.loads(telemetry_bytes)

    assert [e["index"] for e in transcript] == [0, 1]
    assert transcript[0]["tool_calls"][0] == {"name": "execute", "args": {"cmd": "go"}}
    assert telemetry["model_steps"] == 2 and telemetry["stop_reason"].startswith("StepBudget")
    # The log-framework meta keys never leak into the reconstructed records.
    assert "timestamp" not in telemetry and all("event" not in e for e in transcript)


def test_reconstruction_orders_by_index_and_ignores_noise() -> None:
    # Out-of-order lines + plain stdout noise + a non-JSON line must not corrupt the result.
    stderr = "\n".join(
        [
            "Traceback (most recent call last):",
            _log_line(TURN_EVENT, index=1, type="AIMessage", content="second"),
            _log_line(TURN_EVENT, index=0, type="HumanMessage", content="first"),
            "{ not valid json",
        ]
    )
    transcript_bytes, telemetry_bytes = reconstruct_from_logs(stderr)
    transcript = json.loads(transcript_bytes)
    assert [e["content"] for e in transcript] == ["first", "second"]
    assert telemetry_bytes is None  # no telemetry event present


def test_missing_events_yield_none() -> None:
    transcript_bytes, telemetry_bytes = reconstruct_from_logs("no structured events here\n")
    assert transcript_bytes is None and telemetry_bytes is None


def test_render_message_caps_overlong_fields_and_is_json_safe() -> None:
    big = "y" * (TRANSCRIPT_FIELD_CAP + 1000)
    entry = render_message(3, _Msg(big))
    assert entry["index"] == 3
    assert "truncated" in entry["content"] and len(entry["content"]) < len(big)
    json.dumps(entry)  # must not raise — the entry is guaranteed JSON-safe for the logger


# ------------------------------------------------- telemetry reconstruction from turns (#125)


class AIMessage:  # noqa: N801 — duck-typed: render_message keys usage capture on the class NAME
    """A duck-typed AI message carrying per-turn usage (no langchain import needed)."""

    def __init__(self, content: str, *, tool_calls: list[dict] | None = None,
                 usage: dict | None = None, model: str | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage_metadata = usage
        self.response_metadata = {"model_name": model} if model else {}


def test_render_message_captures_per_turn_usage_on_ai_messages() -> None:
    entry = render_message(1, AIMessage(
        "plan", tool_calls=[{"name": "execute", "args": {}}],
        usage={"input_tokens": 120, "output_tokens": 30}, model="claude-sonnet-4-6",
    ))
    assert entry["usage"] == {"input_tokens": 120, "output_tokens": 30}
    assert entry["model"] == "claude-sonnet-4-6"
    # non-AI messages carry no usage key at all
    assert "usage" not in render_message(2, _Msg("tool result", name="execute"))


def test_telemetry_from_turns_mirrors_the_worker_aggregate() -> None:
    from verity.sandbox.log_transcript import telemetry_from_turns

    turns = [
        render_message(0, _Msg("go")),
        render_message(1, AIMessage("a", tool_calls=[{"name": "execute", "args": {}}],
                                    usage={"input_tokens": 100, "output_tokens": 20},
                                    model="m1")),
        render_message(2, _Msg("ok", name="execute")),
        render_message(3, AIMessage("b", usage={"input_tokens": 200, "output_tokens": 40})),
    ]
    telemetry = telemetry_from_turns(turns)
    assert telemetry == {
        "input_tokens": 300, "output_tokens": 60, "total_tokens": 360,
        "model_steps": 2, "tool_calls": 1, "model": "m1",
        "source": "reconstructed_from_turns",
    }
    assert telemetry_from_turns([]) is None
