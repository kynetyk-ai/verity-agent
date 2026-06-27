"""In-container entrypoint (ROADMAP Phase 3, Sprint 2): build + run the agent in isolation.

Runs inside the sandbox container (``python -m verity.sandbox.container_entry``). Reads the mounted
``CycleInput``, builds the YOLO coding agent over ``/work`` via the **shared** builders (same
tool-binding + descriptor contract as the in-process driver), and runs it. The propose tool leaves
the descriptor in ``/work/outbox``, harvested by the host after the container exits. Imports Deep
Agents — quarantined behind the mypy override, runs only in the image.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from deepagents.backends.local_shell import LocalShellBackend
from langgraph.errors import GraphRecursionError

from verity.logging import configure_logging, get_logger
from verity.sandbox.container_io import (
    CONTAINER_INPUT_PATH,
    CONTAINER_OUTBOX,
    CONTAINER_WORKSPACE,
    CycleInput,
)
from verity.sandbox.deepagents_driver import (
    StepBudgetExceeded,
    build_deepagents_agent,
    run_agent,
)
from verity.sandbox.descriptor import RESERVED_TELEMETRY_NAME
from verity.sandbox.model_spec import ModelSpec, resolve_model
from verity.sandbox.tools import resolve_tools

log = get_logger("verity.sandbox.container_entry")

_EXECUTE_TIMEOUT_S = 1500


def main() -> None:
    configure_logging(json_output=True)
    cycle = CycleInput.from_json(Path(CONTAINER_INPUT_PATH).read_bytes())
    # The worker's /work is a fresh writable mount (empty); the propose tool writes the descriptor
    # to outbox/, so it must exist before the agent runs. Idempotent (the old per-cycle-container
    # path provisions it via the workspace layout).
    Path(CONTAINER_OUTBOX).mkdir(parents=True, exist_ok=True)
    # Reconstruct the model target from env (a bare provider string for Anthropic, or an
    # OpenAI-compatible endpoint with a base_url) and resolve it for create_deep_agent (Phase 6).
    spec = ModelSpec.from_env(os.environ)
    model = resolve_model(spec)
    # Shell-capable backend: the agent's native `execute` runs code IN the container (the container
    # is the isolation). This replaces the old custom run_shell tool.
    backend = LocalShellBackend(
        root_dir=Path(CONTAINER_WORKSPACE), virtual_mode=False, timeout=_EXECUTE_TIMEOUT_S
    )
    agent = build_deepagents_agent(
        model=model,
        operations=cycle.operations,
        outbox=Path(CONTAINER_OUTBOX),
        system_prompt=cycle.system_prompt,
        backend=backend,
        deadline_s=cycle.deadline_s,
        step_budget=cycle.step_budget,
        extra_tools=resolve_tools(cycle.tool_names),
    )
    log.info(
        "container_entry_run",
        model=spec.provider_string(),
        base_url=spec.base_url,
        ops=[s.name for s in cycle.operations],
    )
    try:
        run_agent(
            agent, cycle.user_message,
            recursion_limit=cycle.recursion_limit, outbox=Path(CONTAINER_OUTBOX),
        )
    except (StepBudgetExceeded, GraphRecursionError) as exc:
        # The agent ran out of its step/recursion budget (#103) — a graceful, EXPECTED stop. The
        # soft step nudge biases the agent to propose first, so the propose tool has usually already
        # written its descriptor to the outbox. Salvage it: record the stop reason and exit 0 so the
        # host HARVESTS the (possibly partial) proposal, instead of crashing exit 1 and losing the
        # whole cycle. If nothing was proposed, the host records a "no proposal" cycle and feeds it
        # back (degrade-don't-crash) — still never a fatal exit-1 loss. A genuine crash (OOM, tool
        # error) is NOT caught here, so it still surfaces as a non-zero exit with diagnostics.
        reason = f"{type(exc).__name__}: {exc}"
        log.warning("container_entry_budget_exhausted", reason=reason)
        _annotate_stop_reason(Path(CONTAINER_OUTBOX), reason)
    log.info("container_entry_done")


def _annotate_stop_reason(outbox: Path, reason: str) -> None:
    """Record the graceful stop reason in the harvested telemetry (best-effort, never raises).

    ``run_agent`` only writes ``__telemetry__`` on a clean return, so on a budget-exhausted stop we
    write (or augment) it here with ``stop_reason`` — surfacing the cause in the RunReport rather
    than salvaging silently.
    """
    path = outbox / RESERVED_TELEMETRY_NAME
    try:
        telemetry = json.loads(path.read_bytes()) if path.exists() else {}
    except (OSError, ValueError):
        telemetry = {}
    telemetry["stop_reason"] = reason
    try:
        path.write_bytes(json.dumps(telemetry).encode("utf-8"))
    except OSError as exc:
        log.warning("container_entry_telemetry_write_failed", error=str(exc))


if __name__ == "__main__":
    main()
