"""In-container entrypoint (ROADMAP Phase 3, Sprint 2): build + run the agent in isolation.

Runs inside the sandbox container (``python -m verity.sandbox.container_entry``). Reads the mounted
``CycleInput``, builds the YOLO coding agent over ``/work`` via the **shared** builders (same
tool-binding + descriptor contract as the in-process driver), and runs it. The propose tool leaves
the descriptor in ``/work/outbox``, harvested by the host after the container exits. Imports Deep
Agents — quarantined behind the mypy override, runs only in the image.
"""

from __future__ import annotations

import os
from pathlib import Path

from verity.logging import configure_logging, get_logger
from verity.sandbox.container_io import (
    CONTAINER_INPUT_PATH,
    CONTAINER_OUTBOX,
    CONTAINER_WORKSPACE,
    CycleInput,
)
from verity.sandbox.deepagents_driver import build_deepagents_agent, run_agent

log = get_logger("verity.sandbox.container_entry")

_DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"


def main() -> None:
    configure_logging(json_output=True)
    cycle = CycleInput.from_json(Path(CONTAINER_INPUT_PATH).read_bytes())
    model = os.environ.get("VERITY_SANDBOX_MODEL", _DEFAULT_MODEL)
    agent = build_deepagents_agent(
        model=model,
        operations=cycle.operations,
        root=Path(CONTAINER_WORKSPACE),
        outbox=Path(CONTAINER_OUTBOX),
        system_prompt=cycle.system_prompt,
    )
    log.info("container_entry_run", model=model, ops=[s.name for s in cycle.operations])
    run_agent(agent, cycle.user_message, recursion_limit=cycle.recursion_limit)
    log.info("container_entry_done")


if __name__ == "__main__":
    main()
