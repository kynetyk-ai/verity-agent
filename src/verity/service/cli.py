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
from typing import Any

from verity.composition import TaskRequest
from verity.composition.task_request import DataRequest, PolicyRequest, SandboxRequest

__all__ = ["main"]

DEFAULT_SOCKET = "/run/verity.sock"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verity", description="Verity control-plane CLI.")
    parser.add_argument("--socket", help="daemon socket path ($VERITY_SOCKET or /run/verity.sock)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve", help="run the control-plane daemon")
    sub.add_parser("tasks", help="list created tasks")

    cat = sub.add_parser("catalog", help="list task types and their published contracts")
    cat.add_argument("--type", help="describe just this task type")

    ing = sub.add_parser("ingest", help="ingest a file from the exchange; returns a data handle")
    ing.add_argument("file", help="filename under the exchange's in/ directory")

    create = sub.add_parser("create", help="create a task instance from a declarative request")
    create.add_argument("--type", required=True, help="task type (e.g. fe, code)")
    create.add_argument("--goal", default="", help="the run goal")
    create.add_argument("--data", help="a data handle from `verity ingest`")
    create.add_argument("--model", help="provider:model string (e.g. anthropic:claude-sonnet-4-6)")
    create.add_argument("--base-url", help="OpenAI-compatible endpoint for a local/hosted model")
    create.add_argument("--max-cycles", type=int, default=4)
    create.add_argument("--stop-on-accept", action="store_true")
    create.add_argument("--per-class", type=int, default=300, help="FE: rows/class to subsample")
    create.add_argument("--reserved-fraction", type=float, default=0.5, help="FE: held-out share")

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
    return TaskRequest(
        type_name=args.type,
        goal=args.goal,
        sandbox=SandboxRequest(model=args.model, base_url=args.base_url),
        policy=PolicyRequest(max_cycles=args.max_cycles, stop_on_accept=args.stop_on_accept),
        data=DataRequest(per_class=args.per_class, reserved_fraction=args.reserved_fraction),
    )


async def _dispatch(client: Any, args: argparse.Namespace) -> Any:
    if args.command == "catalog":
        return await client.catalog(args.type)
    if args.command == "tasks":
        return await client.list_tasks()
    if args.command == "ingest":
        return await client.ingest(args.file)
    if args.command == "create":
        return await client.create_task(_task_request(args).to_dict(), data=args.data)
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

        serve_main()
        return 0

    from verity.service.client import Client, ControlServiceError

    socket_path = args.socket or os.environ.get("VERITY_SOCKET", DEFAULT_SOCKET)
    client = Client(socket_path)
    try:
        result = asyncio.run(_dispatch(client, args))
    except ControlServiceError as exc:
        print(f"verity: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
