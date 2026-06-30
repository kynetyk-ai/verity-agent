#!/usr/bin/env python3
"""Render a harvested agent transcript (JSON) into readable Markdown.

Standalone helper — not wired into the harness. The transcript is emitted by the worker as
structured stderr **log events** each cycle (the agent never writes it — it shares the worker uid,
so a file would be tamperable), reconstructed host-side, and content-addressed into the per-task
object store under the reserved ``__transcript__.json`` name. Each entry is one agent message:
``{index, type, content, tool_calls?, tool_name?}``.
``content`` is a string (Human/Tool messages) or a list of content blocks (AIMessage), and each
``tool_calls`` item is ``{name, args}``.

Usage:
    python tools/render_transcript.py <transcript.json> [--max N] [--full] [-o out.md]
    # from the store, over stdin:
    docker exec verity-cp cat /var/lib/verity/tasks/<task>/objects/<hash> \\
        | python tools/render_transcript.py -

Examples:
    python tools/render_transcript.py run.json
    python tools/render_transcript.py run.json --full -o run.md
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

_DEFAULT_MAX = 2000  # per-field display cap (chars); the stored fields are already capped at 8 KiB


def _text_of(content: Any) -> str:
    """Extract human-readable text from a message ``content`` (str, or a list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(str(block["text"]))
                # tool_use blocks are rendered separately from the entry's `tool_calls`; skip here.
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, indent=2, default=str)


def _clip(text: str, limit: int | None) -> str:
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated {len(text) - limit} more chars]"


def _fence(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text.rstrip()}\n```"


def _render_args(args: Any, limit: int | None) -> str:
    # Show a shell command bare; everything else as pretty JSON. Both clipped + fenced.
    if isinstance(args, dict) and set(args) <= {"command", "cmd"} and len(args) == 1:
        return _fence(_clip(str(next(iter(args.values()))), limit), "bash")
    pretty = args if isinstance(args, str) else json.dumps(args, indent=2, default=str)
    return _fence(_clip(pretty, limit))


def render(entries: list[dict[str, Any]], limit: int | None) -> str:
    out: list[str] = [f"# Transcript — {len(entries)} messages\n"]
    for e in entries:
        idx, mtype = e.get("index", "?"), e.get("type", "?")
        tool = f" ({e['tool_name']})" if e.get("tool_name") else ""
        out.append(f"## [{idx}] {mtype}{tool}")
        text = _text_of(e.get("content")).strip()
        if text:
            # Tool results (ToolMessage) read best fenced; model/human prose as plain text.
            out.append(_fence(_clip(text, limit)) if mtype == "ToolMessage" else _clip(text, limit))
        for tc in e.get("tool_calls", []) or []:
            out.append(f"🔧 **{tc.get('name')}**")
            out.append(_render_args(tc.get("args"), limit))
        out.append("")  # blank line between entries
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render an agent transcript JSON as Markdown.")
    ap.add_argument("path", help="transcript JSON file, or '-' for stdin")
    ap.add_argument("--max", type=int, default=_DEFAULT_MAX,
                    help=f"per-field display cap in chars (default {_DEFAULT_MAX})")
    ap.add_argument("--full", action="store_true", help="no truncation (overrides --max)")
    ap.add_argument("-o", "--out", help="write to this file instead of stdout")
    args = ap.parse_args(argv)

    raw = sys.stdin.read() if args.path == "-" else open(args.path, encoding="utf-8").read()
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: not valid JSON ({exc})", file=sys.stderr)
        return 2
    if not isinstance(entries, list):
        print("error: transcript must be a JSON array of message entries", file=sys.stderr)
        return 2

    md = render(entries, None if args.full else args.max)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"wrote {args.out} ({len(entries)} messages)", file=sys.stderr)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
