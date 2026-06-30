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

from deepagents.backends.local_shell import LocalShellBackend

from verity.logging import configure_logging, get_logger
from verity.sandbox.container_io import (
    CONTAINER_INPUT_PATH,
    CONTAINER_OUTBOX,
    CONTAINER_WORKSPACE,
    CycleInput,
)
from verity.sandbox.deepagents_driver import (
    build_deepagents_agent,
    run_agent,
)
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
    # `inherit_env=True` so `execute` sees the container env — PATH + the image's pip config
    # (PIP_TARGET=/work/pylib, TMPDIR=/work, PIP_NO_CACHE_DIR) — so a bare `pip install` works under
    # the read-only mount. Without it the backend runs commands with an EMPTY env and pip dies on a
    # read-only venv/cache (Errno 30) or the tiny /tmp (Errno 28). Blank the model API key so it is
    # NOT exposed to the LLM-controlled shell (the runtime reads it from os.environ directly).
    key_env = spec.key_env()
    backend = LocalShellBackend(
        root_dir=Path(CONTAINER_WORKSPACE), virtual_mode=False, timeout=_EXECUTE_TIMEOUT_S,
        inherit_env=True, env={key_env: ""} if key_env else None,
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
    # run_agent logs each turn + a final telemetry event (incl. stop_reason on a budget/recursion
    # limit-end) to STDERR, which the host captures and reconstructs the transcript/telemetry from —
    # the agent never sees a diagnostics file (it shares this worker's uid). A budget end is handled
    # gracefully inside run_agent (no raise); only a genuine crash propagates to a non-zero exit.
    # The returned (transcript, telemetry) is ignored here — already streamed out via logs.
    run_agent(
        agent, cycle.user_message,
        recursion_limit=cycle.recursion_limit, outbox=Path(CONTAINER_OUTBOX),
    )
    log.info("container_entry_done")


if __name__ == "__main__":
    main()
