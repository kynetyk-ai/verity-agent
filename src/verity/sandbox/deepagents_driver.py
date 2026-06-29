"""Deep Agents agent build + the in-process driver (ROADMAP Phase 3).

The Deep Agents / LangChain import is quarantined to this module and
:mod:`~verity.sandbox.container_entry` (both behind the mypy override). The agent is a general
coding agent in **YOLO mode** — Deep Agents' native coding toolset (file ops + ``execute`` shell +
planning + subagents), **no** permission prompts / human-in-the-loop, plus the generated propose
tools — whose only task-specific inputs come through the control plane (the system prompt, the
served context, and the operations from the schema). Isolation, not in-loop gating, is the boundary.

Shell execution is the **backend's** native ``execute``: the container entrypoint uses a
shell-capable backend (``LocalShellBackend``) so ``execute`` runs code in the isolated container.
The in-process driver uses a plain ``FilesystemBackend`` (file tools only — it is the dev/test
harness and must not run untrusted code on the host).

:func:`build_deepagents_agent` + :func:`run_agent` are the shared build/run used by both drivers, so
the tool-binding and descriptor contract are single-sourced. Each domain ``OperationSignature``
becomes one propose tool (name = op name) that writes the descriptor to the outbox; the host core
harvests and mints (see :mod:`verity.sandbox.descriptor`).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError

from verity.control_plane.registries import OperationSignature
from verity.control_plane.workspace import ProvisionedWorkspace
from verity.logging import get_logger
from verity.sandbox.container_io import effective_recursion_limit, effective_step_budget
from verity.sandbox.descriptor import (
    RESERVED_PROPOSAL_NAME,
    RESERVED_TELEMETRY_NAME,
    RESERVED_TRANSCRIPT_NAME,
    ProposalDescriptor,
)
from verity.sandbox.errors import SandboxError
from verity.sandbox.model_spec import coerce_model
from verity.sandbox.tools import resolve_tools

__all__ = [
    "DeepAgentsInProcessDriver",
    "DeadlineMiddleware",
    "StepBudgetMiddleware",
    "StepBudgetExceeded",
    "build_deepagents_agent",
    "effective_recursion_limit",
    "effective_step_budget",
    "run_agent",
]

log = get_logger("verity.sandbox.deepagents_driver")

_DEFAULT_RECURSION_LIMIT = 80
_DEADLINE_WARN_FRACTION = 0.8  # warn the agent once it is this far into its time/step budget
_DEADLINE_REMIND_FRACTION = 0.5  # start the recurring time-remaining awareness at the halfway mark

# Finalize guard (#: end-of-loop, not time/step): a ReAct agent ends the moment the model returns a
# message with no tool call — nothing structurally requires a proposal first. If the loop ends with
# no proposal in the outbox, re-invoke with a salient "submit now" nudge a bounded number of times,
# so a model that simply stopped (thinking it was "done") gets a clear chance to deliver. The two
# DeadlineMiddleware/StepBudget nudges only fire under time/step PRESSURE; this catches the agent
# that converges early and never submits.
_FINALIZE_RETRIES = 2
_FINALIZE_NUDGE = (
    "NO proposal submitted yet — nothing is recorded and this cycle is wasted without one. The one "
    "deliverable this cycle is a proposal. Do it NOW: write every declared object file to "
    "outbox/ and CALL THE PROPOSE/SUBMIT TOOL. Do not stop until that tool returns success."
)
# The rendered message transcript is written to the outbox EVERY cycle (not only on a no-proposal
# failure) under the reserved name, harvested like the telemetry, and content-addressed into the
# store — the step-level record the experiments read back. See ``RESERVED_TRANSCRIPT_NAME``.
_TRANSCRIPT_NAME = RESERVED_TRANSCRIPT_NAME
# Per-field byte cap in the transcript (message content / tool args / tool result) — a runaway tool
# dump (a CSV echo, a stack trace) is truncated so it can't bloat the content-addressed store.
_TRANSCRIPT_FIELD_CAP = 8192


class DeadlineMiddleware(AgentMiddleware):
    """A two-tier soft-deadline ladder (ROADMAP 5.2, #102): pace early, then finalize before death.

    The hard sandbox timeout is the backstop; this biases the agent to *budget its time* and then
    *finalize* first. Two additive tiers, both wall-clock against ``deadline_s``:

    * **``reminder_fraction``→``warn_fraction`` (50%→80%, recurring, EPHEMERAL):**
      ``wrap_model_call`` augments the system prompt *for that one model call* with a transient
      "≈Xs left — budget accordingly" note. It is **never written to durable message state** (no
      context bloat — the structural-context-hygiene thesis), and is throttled to a coarse cadence
      so it does not fire on every step.
    * **``warn_fraction`` (80%, one-time, PERSISTED):** ``before_model`` injects a single sharp
      "finalize and submit NOW" :class:`HumanMessage` — a salient interrupt for the hard pivot.

    100% is the hard stop (worker timeout), handled outside this middleware. ``now`` injectable for
    tests.
    """

    def __init__(
        self,
        deadline_s: float,
        *,
        now: Callable[[], float] = time.monotonic,
        warn_fraction: float = _DEADLINE_WARN_FRACTION,
        reminder_fraction: float = _DEADLINE_REMIND_FRACTION,
    ) -> None:
        super().__init__()
        self._deadline_s = deadline_s
        self._now = now
        self._warn_at = warn_fraction * deadline_s
        self._remind_at = reminder_fraction * deadline_s
        # Throttle the recurring note to ~once per 10% of the budget (a 50/40/30/20% coarse
        # countdown), so it paces the agent without nagging every step or bloating tokens.
        self._remind_interval_s = max(1.0, 0.1 * deadline_s)
        self._start: float | None = None
        self._warned = False
        self._last_remind_at: float | None = None

    def _elapsed(self) -> float:
        if self._start is None:
            self._start = self._now()
        return self._now() - self._start

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self._start = self._now()
        return None

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        # The one-time, persisted 80% finalize-now pivot.
        elapsed = self._elapsed()
        if self._warned or elapsed < self._warn_at:
            return None
        self._warned = True
        remaining = max(0, int(self._deadline_s - elapsed))
        return {
            "messages": [
                HumanMessage(content=(
                    f"Time almost up: ~{remaining}s of {int(self._deadline_s)}s left. STOP "
                    "exploring and finalize NOW, even if imperfect: build the best, "
                    "properly-shaped proposal and submit it by CALLING THE PROPOSE/SUBMIT "
                    "TOOL. Doing the work but never calling the tool records nothing — submit now."
                ))
            ]
        }

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # The recurring, EPHEMERAL 50%→80% pacing note. Augment the system prompt for this single
        # call only; the durable request/state is untouched, so nothing accumulates across steps.
        note = self._time_note()
        if note is None:
            return handler(request)
        base = request.system_prompt or ""
        augmented = SystemMessage(content=f"{base}\n\n{note}".strip())
        return handler(request.override(system_message=augmented))

    def _time_note(self) -> str | None:
        """The transient time-remaining note to inject this call, or ``None`` (before 50%, after
        80%, or throttled). Coarsely throttled so it paces rather than nags."""
        elapsed = self._elapsed()
        if elapsed < self._remind_at or elapsed >= self._warn_at:
            return None
        now = self._now()
        if self._last_remind_at is not None:
            if now - self._last_remind_at < self._remind_interval_s:
                return None
        self._last_remind_at = now
        remaining = max(0, int(self._deadline_s - elapsed))
        return (
            f"[time budget] ~{remaining}s of {int(self._deadline_s)}s remaining this cycle. "
            "Budget the remaining time; do not start work (long trainings, broad searches) that "
            "cannot finish and submit before the deadline."
        )


class StepBudgetExceeded(RuntimeError):
    """The agent burned its per-cycle model-step budget without converging on a proposal."""


class StepBudgetMiddleware(AgentMiddleware):
    """A per-cycle model-step budget (ROADMAP 5.1): nudge to wrap up, then hard-stop if ignored.

    Each ``before_model`` is one model step. At ``warn_fraction`` of ``max_steps`` it injects a
    one-time wrap-up nudge (like :class:`DeadlineMiddleware`, but counting steps not wall-clock);
    once the budget is spent it raises :class:`StepBudgetExceeded`, which the drivers turn into a
    cycle-level ``SandboxError`` so a non-converging agent yields a *recorded, fed-back* failed
    cycle with a clear reason instead of an opaque ``GraphRecursionError`` at the framework limit.
    """

    def __init__(self, max_steps: int, *, warn_fraction: float = _DEADLINE_WARN_FRACTION) -> None:
        super().__init__()
        self._max = max_steps
        self._warn_at = max(1, int(warn_fraction * max_steps))
        self._steps = 0
        self._warned = False

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self._steps += 1
        if self._steps > self._max:
            raise StepBudgetExceeded(
                f"model-step budget exhausted ({self._max} steps); the agent did not converge on a "
                "proposal within its budget"
            )
        if not self._warned and self._steps >= self._warn_at:
            self._warned = True
            return {
                "messages": [
                    HumanMessage(content=(
                        f"Step budget almost spent: {self._steps} of {self._max} model steps used. "
                        "Stop exploring now — finalize and submit the best proposal immediately."
                    ))
                ]
            }
        return None


def build_deepagents_agent(
    *,
    model: Any,
    operations: tuple[OperationSignature, ...],
    outbox: Path,
    system_prompt: str,
    backend: Any,
    extra_tools: Sequence[Any] = (),
    middleware: Sequence[Any] = (),
    deadline_s: float | None = None,
    step_budget: int | None = None,
) -> Any:
    """Build the YOLO coding agent: Deep Agents' native tools + a propose tool per operation.

    ``backend`` decides the coding capability: a shell-capable backend (``LocalShellBackend``) gives
    a working ``execute``; a plain ``FilesystemBackend`` gives file tools only. ``outbox`` is where
    the propose tools write the descriptor. ``extra_tools`` are domain/task tools the adapter binds
    (§8.2 seam, #6). ``middleware`` forwards to ``create_deep_agent``; when ``deadline_s`` is set a
    :class:`DeadlineMiddleware` is appended, and when ``step_budget`` is set a
    :class:`StepBudgetMiddleware` is appended (ROADMAP 5.1). (Deep Agents auto-injects
    **summarization/compaction** middleware into its base stack, so compaction is already on — 5.2.)
    """
    tools = [_make_propose_tool(sig, outbox) for sig in operations] + list(extra_tools)
    stack = list(middleware)
    if deadline_s is not None:
        stack.append(DeadlineMiddleware(deadline_s))
    if step_budget is not None:
        stack.append(StepBudgetMiddleware(step_budget))
    # The framework is opaque to us: treat the compiled agent as Any so its overloaded `invoke`
    # bridges cleanly through asyncio.to_thread / a direct call.
    agent: Any = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        backend=backend,
        middleware=stack,
    )
    return agent


def run_agent(
    agent: Any,
    user_message: str,
    *,
    recursion_limit: int = _DEFAULT_RECURSION_LIMIT,
    outbox: Path | None = None,
) -> None:
    """Run the agent loop to completion (synchronous), with the end-of-loop finalize guard.

    When ``outbox`` is given, the loop's telemetry (tokens / steps / tool-calls / model) is written
    to the reserved ``__telemetry__.json`` there, harvested back to the host like the proposal
    descriptor (ROADMAP 5.3b). Best-effort: a model without ``usage_metadata`` yields zeros / nulls.

    The run is **streamed**, not ``invoke``-d, so the accumulated message state is recoverable even
    when the run hits a recursion / step-budget limit: ``GraphRecursionError`` does not return the
    messages, so an ``invoke`` that raises would lose the whole trace — the case we most want
    to inspect. We keep the last streamed state and persist diagnostics from it before re-raising.

    The **finalize guard**: a ReAct agent ends as soon as the model returns a message with no tool
    call, with no structural requirement that it proposed first. If the loop ends with no proposal
    in the outbox, re-invoke with a salient "submit now" nudge up to ``_FINALIZE_RETRIES`` times
    (the accumulated history carried forward), so an agent that simply converged without delivering
    gets a clear, bounded chance to. A guard re-invoke is best-effort — if it raises (e.g. a step or
    recursion limit), we stop and let the no-proposal outcome stand. The telemetry + rendered
    transcript are persisted EVERY cycle (success, normal no-proposal, AND a recursion/budget
    limit-end), harvested + content-addressed by the host like the proposal descriptor (5.3b).
    """
    inputs = {"messages": [{"role": "user", "content": user_message}]}
    config = {"recursion_limit": recursion_limit}
    result: Any = {"messages": []}
    try:
        for state in agent.stream(inputs, config, stream_mode="values"):
            result = state  # keep the latest full state (messages survive an exhaustion below)
            # Persist after every step so the trace survives even a HARD wall-clock timeout that
            # SIGKILLs the worker mid-run (#103 / F4): the workspace is bind-mounted, so the last
            # write lands on the host and the control plane salvages it. Cheap + best-effort.
            if outbox is not None:
                _write_diagnostics(result, outbox)
    except (GraphRecursionError, StepBudgetExceeded):
        # Budget/recursion exhausted — a graceful, EXPECTED stop (#103). We still have the streamed
        # state, so persist diagnostics from it BEFORE re-raising, so a no-proposal limit-end cycle
        # is still debuggable. container_entry catches the re-raise, annotates the stop reason, and
        # exits 0; the host then harvests this transcript like any other.
        if outbox is not None:
            _write_diagnostics(result, outbox)
        raise
    if outbox is not None:
        result = _finalize_guard(agent, result, outbox, recursion_limit=recursion_limit)
        _write_diagnostics(result, outbox)


def _write_diagnostics(result: Any, outbox: Path) -> None:
    """Persist the loop's telemetry + step transcript to the outbox (best-effort; never fatal)."""
    try:
        telemetry = _extract_telemetry(result)
        (outbox / RESERVED_TELEMETRY_NAME).write_bytes(json.dumps(telemetry).encode("utf-8"))
    except OSError as exc:  # diagnostics are best-effort — never let them fail/mask the cycle
        log.warning("telemetry_persist_failed", error=str(exc))
    _persist_transcript(result, outbox)  # already OSError-guarded


def _finalize_guard(
    agent: Any, result: Any, outbox: Path, *, recursion_limit: int
) -> Any:
    """Re-invoke the agent to deliver when the loop ended with no proposal (bounded retries)."""
    for attempt in range(1, _FINALIZE_RETRIES + 1):
        if (outbox / RESERVED_PROPOSAL_NAME).is_file():
            return result
        log.warning("finalize_guard_nudge", attempt=attempt, max_attempts=_FINALIZE_RETRIES)
        prior = result.get("messages", []) if isinstance(result, dict) else []
        try:
            result = agent.invoke(
                {"messages": [*prior, HumanMessage(content=_FINALIZE_NUDGE)]},
                {"recursion_limit": recursion_limit},
            )
        except Exception as exc:  # noqa: BLE001 — a guard re-invoke is best-effort; let the
            # no-proposal outcome stand (the host records a recoverable failed cycle, #21).
            log.warning("finalize_guard_invoke_failed", attempt=attempt, error=str(exc))
            return result
    return result


def _persist_transcript(result: Any, outbox: Path) -> None:
    """Write the rendered step transcript to the outbox — the per-cycle instrumentation record.

    Captures, per message: an ``index``, the message ``type`` and ``content``, the tool calls
    (name **+ arguments**), and any tool ``result`` — enough to reconstruct the agent's trajectory
    (what it did, in what order, with what inputs/outputs). Large ``content``/args/results are
    truncated to ``_TRANSCRIPT_FIELD_CAP`` bytes (a runaway tool dump can't bloat the store); the
    write is best-effort (a transcript must never fail a cycle). The host harvests it like the
    telemetry and content-addresses it into the object store.
    """
    messages = result.get("messages", []) if isinstance(result, dict) else []
    rendered: list[dict[str, Any]] = []
    for index, m in enumerate(messages):
        tool_calls = [
            {"name": tc.get("name"), "args": _cap(tc.get("args"))}
            for tc in (getattr(m, "tool_calls", None) or [])
        ]
        entry: dict[str, Any] = {
            "index": index,
            "type": type(m).__name__,
            "content": _cap(getattr(m, "content", None)),
        }
        if tool_calls:
            entry["tool_calls"] = tool_calls
        # A ToolMessage carries the tool's *result*; record it (capped) so the loop is legible.
        if type(m).__name__ == "ToolMessage" and getattr(m, "name", None) is not None:
            entry["tool_name"] = m.name
        rendered.append(entry)
    try:
        (outbox / _TRANSCRIPT_NAME).write_bytes(json.dumps(rendered, default=str).encode("utf-8"))
    except OSError as exc:  # instrumentation is best-effort — never let it fail the cycle
        log.warning("transcript_persist_failed", error=str(exc))


def _cap(value: Any, limit: int = _TRANSCRIPT_FIELD_CAP) -> Any:
    """Truncate an overlong rendered field so a runaway tool dump can't bloat the transcript."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= limit:
        return value
    return text[:limit] + f"…[truncated {len(text) - limit} chars]"


def _extract_telemetry(result: Any) -> dict[str, Any]:
    """Sum token usage + count steps / tool-calls across the run's ``AIMessage``s (5.3b)."""
    messages = result.get("messages", []) if isinstance(result, dict) else []
    input_tokens = output_tokens = model_steps = tool_calls = 0
    model_name: str | None = None
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        model_steps += 1
        tool_calls += len(message.tool_calls or [])
        usage = message.usage_metadata
        if usage:
            input_tokens += int(usage.get("input_tokens", 0) or 0)
            output_tokens += int(usage.get("output_tokens", 0) or 0)
        meta = message.response_metadata or {}
        model_name = meta.get("model_name") or meta.get("model") or model_name
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "model_steps": model_steps,
        "tool_calls": tool_calls,
        "model": model_name,
    }


class DeepAgentsInProcessDriver:
    """Runs the Deep Agents loop in this process (tests/dev). ``model`` is a provider string/model.

    YOLO: no ``interrupt_on`` / HITL. Uses a plain ``FilesystemBackend`` (file tools, **no**
    working ``execute``) — in-process must not run untrusted code on the host, so this driver is for
    the fake-model offline tests and the code-execution-free live smoke; arbitrary-code work belongs
    in the container driver.
    """

    def __init__(
        self,
        *,
        model: Any,
        recursion_limit: int = _DEFAULT_RECURSION_LIMIT,
        tool_names: tuple[str, ...] = (),
        step_budget: int | None = None,
    ) -> None:
        self._model = model
        self._recursion_limit = recursion_limit
        self._tool_names = tool_names
        self._step_budget = step_budget

    async def run(
        self,
        *,
        system_prompt: str,
        user_message: str,
        operations: tuple[OperationSignature, ...],
        workspace: ProvisionedWorkspace,
    ) -> None:
        backend = FilesystemBackend(root_dir=workspace.root, virtual_mode=False)
        agent = build_deepagents_agent(
            # Resolve a ModelSpec at the in-process edge (parity with the container path); a bare
            # string or a fake chat model passes through untouched (Phase 6).
            model=coerce_model(self._model),
            operations=operations,
            outbox=workspace.outbox(),
            system_prompt=system_prompt,
            backend=backend,
            extra_tools=resolve_tools(self._tool_names),
            step_budget=effective_step_budget(self._recursion_limit, self._step_budget),
        )
        log.info("deepagents_run", ops=[s.name for s in operations], root=str(workspace.root))
        try:
            await asyncio.to_thread(
                run_agent, agent, user_message,
                recursion_limit=effective_recursion_limit(self._recursion_limit, self._step_budget),
                outbox=workspace.outbox(),
            )
        except Exception as exc:
            # A runaway loop (GraphRecursionError) or any in-loop failure becomes a cycle-level
            # SandboxError, so the control plane can skip the cycle rather than abort the run (#21).
            raise SandboxError(
                f"in-process agent loop failed: {type(exc).__name__}: {exc}"
            ) from exc


def _make_propose_tool(signature: OperationSignature, outbox: Path) -> Callable[..., str]:
    """Bind one :class:`OperationSignature` to a propose tool that writes the outbox descriptor."""
    op_name = signature.name
    out_type = signature.output
    in_types = ", ".join(signature.inputs) or "(none)"
    object_keys = signature.object_payload_keys

    def _missing_objects(payload: dict[str, Any]) -> list[str]:
        """Declared object files (by the op's `object_payload_keys`) absent from outbox/."""
        missing = []
        for key in object_keys:
            name = payload.get(key)
            if isinstance(name, str) and name and not (outbox / name).is_file():
                missing.append(f"{key}={name!r}")
        return missing

    def propose(
        parents: list[str], payload: dict[str, Any], rationale: str = "", tested: bool = False
    ) -> str:
        if not isinstance(payload, dict) or not payload:
            return "rejected: payload must be a non-empty JSON object"
        if not parents:
            return "rejected: parents must include the input artifact id(s) from the context"
        # Enforceable: every object the payload declares must actually be in outbox/ — catch a
        # missing/misplaced file in-cycle (a fixable rejected, no gate cycle spent) instead of a
        # runs-clean reject downstream. Generic: the keys come from the op, not hardcoded names.
        if missing := _missing_objects(payload):
            present = sorted(
                p.name for p in outbox.iterdir()
                if p.is_file() and p.name != RESERVED_PROPOSAL_NAME
            ) if outbox.is_dir() else []
            return (
                f"rejected: declared {', '.join(missing)} but no such file is in outbox/. "
                f"Write the file(s) to outbox/ then call again. "
                f"outbox/ now has: {present or 'nothing'}."
            )
        descriptor = ProposalDescriptor(
            op_name=op_name, parents=tuple(parents), payload=payload, metadata=rationale
        )
        (outbox / RESERVED_PROPOSAL_NAME).write_bytes(descriptor.to_json())
        # `tested` is a self-attestation (we cannot verify it) — a nudge, not a gate. Submitting an
        # untested script is the top cause of an outright reject, so we warn when it is not set.
        warn = "" if tested else (
            " NOTE: tested=false — no attestation that this was RUN end-to-end. If it has not been "
            "run (via the execute tool), it may fail the gate — run it first next time."
        )
        return (
            f"proposal recorded ({op_name} -> {out_type}); "
            f"the harness will harvest and gate it.{warn}"
        )

    propose.__name__ = op_name
    propose.__qualname__ = op_name
    propose.__doc__ = (
        f"Propose a {out_type} via {op_name} (inputs: {in_types}). Call once, last.\n\n"
        f"Before calling this:\n"
        f"  1. Write every declared object file to outbox/ (this tool rejects the call if a file\n"
        f"     the payload names is not there).\n"
        f"  2. Actually RUN the script end-to-end with the execute tool and confirm its output\n"
        f"     — a crash, timeout, or missing output is an outright reject, no partial credit.\n\n"
        f"Args:\n"
        f"    parents: ids of the input artifact(s) of type {in_types}, from the context above.\n"
        f"    payload: the {out_type} payload as a JSON object (see domain instructions / spec).\n"
        f"    rationale: brief note on how/why (audit trail; never shown to the verifier).\n"
        f"    tested: set true ONLY after the script was run end-to-end and its output seen.\n"
        f"        Attestation only (not verified) — but untested is what keeps failing."
    )
    return propose
