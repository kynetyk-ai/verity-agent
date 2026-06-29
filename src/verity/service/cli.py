"""The ``verity`` console script (ROADMAP 8.2, ADR 0004 (d)).

One binary, two roles: ``verity serve`` runs the daemon (the executor); every other verb is a thin
**client** over the daemon's Unix socket — `docker exec verity-cp verity run <task_id>`. Mirrors the
`verity-reaper` precedent (argparse + ``asyncio.run``, 0 / non-zero exit).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import replace
from typing import Any

from verity.composition import TaskRequest
from verity.composition.task_request import DataRequest, PolicyRequest, SandboxRequest

__all__ = ["main"]

DEFAULT_SOCKET = "/run/verity.sock"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verity", description="Verity control-plane CLI.")
    parser.add_argument("--socket", help="daemon socket path ($VERITY_SOCKET or /run/verity.sock)")
    parser.add_argument("--url", help="daemon network URL, e.g. http://host:8080 ($VERITY_URL)")
    parser.add_argument("--token", help="bearer token for a network daemon ($VERITY_API_TOKEN)")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the control-plane daemon")
    serve.add_argument(
        "--http", metavar="HOST:PORT",
        help="serve the authenticated network API on this address (needs $VERITY_API_TOKEN); "
             "default is a local Unix socket",
    )
    sub.add_parser("tasks", help="list created tasks")

    cat = sub.add_parser("catalog", help="list task types and their published contracts")
    cat.add_argument("--type", help="describe just this task type")

    ing = sub.add_parser("ingest", help="ingest a file from the exchange; returns a data handle")
    ing.add_argument("file", help="path under the exchange's in/ directory (subdirs allowed)")

    create = sub.add_parser("create", help="create a task instance from a declarative request")
    create.add_argument("--type", help="task type (see `verity catalog`, e.g. code/fe-holdout/"
                        "fe-kaggle); or from --request-file")
    create.add_argument("--goal", default="", help="the run goal")
    create.add_argument(
        "--file", action="append", default=[], metavar="ROLE:NAME=HANDLE", dest="files",
        help="route a pre-prepared input file (from `verity ingest`) to a worker role, e.g. "
             "`--file agent:train.csv=<handle>`; repeatable. Data prep is done by you (ADR 0005)."
    )
    create.add_argument(
        "--request-file", help="path to a full TaskRequest JSON (e.g. a task package's task.json); "
        "supersedes the request-shaping flags below (only --file/--goal still apply)"
    )
    create.add_argument(
        "--model",
        help="native `provider:model` (e.g. anthropic:claude-sonnet-4-6); OR, WITH --base-url, a "
             "literal OpenAI-compatible model name that may contain colons (e.g. "
             "qwen3.6:27b-coding-mxfp8) — it is NOT split on ':' when --base-url is set",
    )
    create.add_argument(
        "--base-url",
        help="OpenAI-compatible endpoint for a local/hosted model (e.g. "
             "http://host.docker.internal:11434/v1 for host Ollama); makes --model a literal name",
    )
    create.add_argument("--max-cycles", type=int, default=4)
    create.add_argument("--stop-on-accept", action="store_true")

    run = sub.add_parser("run", help="start a run (background); returns a run id")
    run.add_argument("task_id")
    run.add_argument("--goal")

    for verb, help_text in (
        ("status", "a run's scheduling status + summary"),
        ("results", "a run's full results (RunReport JSON)"),
        ("export", "write a run's durable artifacts to the exchange out/ directory"),
    ):
        p = sub.add_parser(verb, help=help_text)
        p.add_argument("run_id")

    return parser


def _task_request(args: argparse.Namespace) -> TaskRequest:
    """Build the `TaskRequest`: from ``--request-file`` (a full JSON request, e.g. a package's
    task.json) if given, else from the request-shaping flags. ``--goal`` overrides either."""
    if args.request_file:
        with open(args.request_file, encoding="utf-8") as fh:
            request = TaskRequest.from_dict(json.load(fh))
        return replace(request, goal=args.goal) if args.goal else request
    if not args.type:
        raise SystemExit("verity create: one of --type or --request-file is required")
    return TaskRequest(
        type_name=args.type,
        goal=args.goal,
        sandbox=SandboxRequest(model=args.model, base_url=args.base_url),
        policy=PolicyRequest(max_cycles=args.max_cycles, stop_on_accept=args.stop_on_accept),
        data=DataRequest(),
    )


def _parse_file_args(entries: list[str]) -> dict[str, dict[str, str]]:
    """Parse repeated ``--file ROLE:NAME=HANDLE`` into ``{role: {name: handle}}`` (ADR 0005)."""
    files: dict[str, dict[str, str]] = {}
    for entry in entries:
        role_name, sep, handle = entry.partition("=")
        role, role_sep, name = role_name.partition(":")
        if not (sep and role_sep and role and name and handle):
            raise SystemExit(f"verity create: --file must be ROLE:NAME=HANDLE, got {entry!r}")
        files.setdefault(role, {})[name] = handle
    return files


async def _dispatch(client: Any, args: argparse.Namespace) -> Any:
    if args.command == "catalog":
        return await client.catalog(args.type)
    if args.command == "tasks":
        return await client.list_tasks()
    if args.command == "ingest":
        return await client.ingest(args.file)
    if args.command == "create":
        return await client.create_task(
            _task_request(args).to_dict(), files=_parse_file_args(args.files)
        )
    if args.command == "run":
        return await client.run(args.task_id, goal=args.goal)
    if args.command == "status":
        return await client.status(args.run_id)
    if args.command == "results":
        return await client.results(args.run_id)
    if args.command == "export":
        return await client.export(args.run_id)
    raise AssertionError(f"unhandled command: {args.command}")  # pragma: no cover


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "serve":
        from verity.service.daemon import main as serve_main

        serve_main(http=args.http)
        return 0

    from verity.service.client import Client, ControlServiceError

    token = args.token or os.environ.get("VERITY_API_TOKEN")
    url = args.url or os.environ.get("VERITY_URL")
    if url:  # the network/TCP daemon (8.4) — bearer-authenticated
        client = Client(base_url=url, token=token)
    else:  # the local Unix socket (the default `docker exec` path)
        socket_path = args.socket or os.environ.get("VERITY_SOCKET", DEFAULT_SOCKET)
        client = Client(socket_path, token=token)
    try:
        result = asyncio.run(_dispatch(client, args))
    except ControlServiceError as exc:
        print(f"verity: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
