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
import re
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
from verity.sandbox.log_transcript import TELEMETRY_EVENT, TURN_EVENT, render_message
from verity.sandbox.model_spec import coerce_model
from verity.sandbox.tools import resolve_tools

__all__ = [
    "DeepAgentsInProcessDriver",
    "DeadlineMiddleware",
    "ToolCallRecoveryMiddleware",
    "StepBudgetMiddleware",
    "StepBudgetExceeded",
    "AgentLoopError",
    "build_deepagents_agent",
    "effective_recursion_limit",
    "effective_step_budget",
    "run_agent",
]

log = get_logger("verity.sandbox.deepagents_driver")

_DEFAULT_RECURSION_LIMIT = 80
_DEADLINE_WARN_FRACTION = 0.8  # warn the agent once it is this far into its time/step budget
_DEADLINE_REMIND_FRACTION = 0.5  # start the recurring time-remaining awareness at the halfway mark

# Tool-call parse recovery. An OpenAI-compatible server that cannot parse the model's emitted tool
# call into its response format returns a 500 whose body message begins with ``error parsing tool
# call`` and quotes the offending ``raw='…'`` (Ollama's exact wording). This is a DETERMINISTIC
# content fault dressed as a transient 5xx: the OpenAI SDK's built-in retries (max_retries=2)
# reproduce it verbatim — the same Ollama seed regenerates the same malformed tool call — so it
# escapes as a fatal ``AgentLoopError`` and wastes the whole cycle. The only recovery is to CHANGE
# THE INPUT: re-run the model call with the rejected payload fed back as a correction, so the next
# generation differs. Bounded per model step; anything that is not this exact signature propagates
# unchanged (a genuine transient 500 stays on the SDK's retry-then-crash path). NARROW BY DESIGN —
# other servers (vLLM, llama.cpp) word this failure differently and will NOT match until their
# signature is added to the check (tracked in #141).
_TOOL_PARSE_SIGNATURE = "error parsing tool call"
_TOOL_PARSE_RAW_RE = re.compile(r"raw='(.*)', err=", re.DOTALL)
_TOOL_PARSE_MAX_RETRIES = 2
_TOOL_PARSE_CORRECTION = (
    "The previous tool call was REJECTED by the model server as invalid JSON and was NOT executed: "
    "{raw}. Re-issue the intended tool call as a SINGLE well-formed JSON object with quoted keys "
    'and values (for example {{"ignore_cache": true}}, never {{"ignore_cache=True"}}). Emit only '
    "strictly valid JSON for every tool call."
)
_TOOL_PARSE_CORRECTION_GENERIC = (
    "The previous tool call was REJECTED by the model server as invalid JSON and was NOT executed. "
    "Re-issue the intended tool call as a SINGLE well-formed JSON object with quoted keys and "
    "values. Emit only strictly valid JSON for every tool call."
)

# Finalize guard (#: end-of-loop, not time/step): a ReAct agent ends the moment the model returns a
# message with no tool call — nothing structurally requires a proposal first. If the loop ends with
# no proposal in the outbox, re-invoke with a salient "submit now" nudge a bounded number of times,
# so a model that simply stopped (thinking it was "done") gets a clear chance to deliver. The two
# DeadlineMiddleware/StepBudget nudges only fire under time/step PRESSURE; this catches the agent
# that converges early and never submits.
_FINALIZE_RETRIES = 2
# ``{outbox}`` is filled with the real absolute outbox path at use (so the nudge names the exact
# directory — never a bare ``outbox/`` a model can misread as the read-only root ``/outbox``).
_FINALIZE_NUDGE = (
    "NO proposal submitted yet — nothing is recorded and this cycle is wasted without one. The one "
    "deliverable this cycle is a proposal. Do it NOW: write every declared object file to "
    "{outbox}/ and CALL THE PROPOSE/SUBMIT TOOL. Do not stop until that tool returns success."
)
# The transcript + telemetry are emitted as STRUCTURED LOG EVENTS (one per turn, plus a final
# telemetry line) to stderr — NOT written as files in the agent's outbox. The worker runs the agent
# loop and its shell tool as the same uid in the same container, so any outbox file the driver wrote
# the agent could read/edit/delete; logging streams the record out of the container where the agent
# can't reach it. The host (backend_driver) reconstructs both from the captured stderr. The
# in-process driver (trusted, no agent shell) instead writes the returned data to its local outbox.
# See ``verity.sandbox.log_transcript`` for the shared producer/consumer contract.


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


def _tool_parse_rejection(exc: BaseException) -> str | None:
    """If ``exc`` is a provider-side tool-call *parse* rejection, return the offending raw payload
    (best-effort; ``""`` when the payload cannot be extracted). ``None`` means "not this case" — the
    caller MUST re-raise untouched, so a transient 500 or any other error keeps today's behavior.

    Duck-typed (no hard ``openai`` import): a 500 status whose body/message carries the exact
    ``error parsing tool call`` signature. Narrow by design — raw extraction is best-effort and
    never gates the match (a format tweak degrades to the generic correction, not a miss).
    """
    if getattr(exc, "status_code", None) != 500:
        return None
    message = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)  # some servers nest the message under "error"
        if isinstance(inner, dict):
            message = str(inner.get("message", ""))
    if not message:  # fall back to the rendered "500 - {…}" string
        message = str(exc)
    if _TOOL_PARSE_SIGNATURE not in message:
        return None
    match = _TOOL_PARSE_RAW_RE.search(message)
    return match.group(1) if match else ""


class ToolCallRecoveryMiddleware(AgentMiddleware):
    """Recover from a provider-side tool-call PARSE rejection instead of crashing the cycle.

    Some OpenAI-compatible servers (Ollama today) return a 500 when they cannot parse the model's
    emitted tool call into their response format. It is a DETERMINISTIC fault the SDK's transient
    retry cannot fix (the same seed regenerates the same malformed call), so left alone it escapes
    as a fatal :class:`AgentLoopError` and wastes the whole cycle. This wraps the model call and, on
    that exact signature only, RE-RUNS it with the rejected payload appended to the system prompt as
    a correction — perturbing the input so the next generation differs and (usually) emits valid
    JSON. The correction is EPHEMERAL (it augments only the retried call, never durable state, like
    the deadline note) and bounded to ``max_retries`` re-runs per model step; every other error
    propagates unchanged so transient 500s stay on the SDK's retry-then-crash path.
    """

    def __init__(self, *, max_retries: int = _TOOL_PARSE_MAX_RETRIES) -> None:
        super().__init__()
        self._max_retries = max_retries

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        attempt = 0
        current = request
        while True:
            try:
                return handler(current)
            except Exception as exc:  # noqa: BLE001 — re-raised at once unless it is our narrow case
                raw = _tool_parse_rejection(exc)
                if raw is None:
                    raise
                attempt += 1
                if attempt > self._max_retries:
                    log.warning("tool_call_parse_recovery_exhausted", max_retries=self._max_retries)
                    raise
                log.warning(
                    "tool_call_parse_recovery", attempt=attempt, max_retries=self._max_retries
                )
                current = self._with_correction(request, raw)  # always augment the CLEAN base

    @staticmethod
    def _with_correction(request: ModelRequest, raw: str) -> ModelRequest:
        correction = (
            _TOOL_PARSE_CORRECTION.format(raw=raw) if raw else _TOOL_PARSE_CORRECTION_GENERIC
        )
        base = request.system_prompt or ""
        augmented = SystemMessage(content=f"{base}\n\n{correction}".strip())
        return request.override(system_message=augmented)


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
    :class:`StepBudgetMiddleware` is appended (ROADMAP 5.1). A :class:`ToolCallRecoveryMiddleware`
    is **always** appended — a no-op unless a provider raises a tool-call parse 500 (Ollama et al.).
    (Deep Agents auto-injects **summarization/compaction** middleware into its base stack, so
    compaction is already on — 5.2.)
    """
    tools = [_make_propose_tool(sig, outbox) for sig in operations] + list(extra_tools)
    stack = list(middleware)
    # Always-on, self-bounding, no-op unless a provider emits a tool-call parse 500 (Ollama et al.);
    # harmless on native providers that never raise it. Prepended so it wraps the pacing middleware
    # below — a recovery re-run still re-applies their deadline/step notes.
    stack.append(ToolCallRecoveryMiddleware())
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


class AgentLoopError(RuntimeError):
    """A genuine in-loop crash, carrying the diagnostics streamed before it (#125).

    ``transcript``/``telemetry`` are the rendered-turn and (partial, crash-stop-reason) telemetry
    JSON bytes at the moment of the crash. The container path doesn't need them (they were already
    logged to stderr and are host-reconstructed); the trusted in-process driver converts this into
    a ``SandboxError(transcript=…, telemetry=…)`` so a crash there stays debuggable too.
    """

    def __init__(self, *, transcript: bytes, telemetry: bytes) -> None:
        super().__init__("agent loop crashed (see __cause__)")
        self.transcript = transcript
        self.telemetry = telemetry


def run_agent(
    agent: Any,
    user_message: str,
    *,
    recursion_limit: int = _DEFAULT_RECURSION_LIMIT,
    outbox: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the agent loop to completion (synchronous); return ``(transcript, telemetry)``.

    Each turn is rendered and **logged** as a structured ``agent_turn`` event AS IT STREAMS, and a
    final ``agent_telemetry`` event (tokens / steps / tool-calls / model, plus ``stop_reason`` on a
    budget end) is logged at the end. The host reconstructs both from the worker's captured stderr —
    the agent never sees a transcript/telemetry file (it shares the worker's uid, so a file would be
    tamperable). Per-turn logging also means a HARD wall-clock kill still leaves every turn already
    streamed out on the host (subsumes F4). The same data is returned for the trusted in-process
    driver, which writes it to its local outbox.

    The run is **streamed**, not ``invoke``-d, so the message state survives a recursion / step
    limit (``GraphRecursionError`` does not return messages). A budget/recursion exhaustion is a
    graceful, EXPECTED stop (#103): we record ``stop_reason`` and return normally (no re-raise), so
    the host records a no-proposal cycle and feeds it back — never a fatal loss. A genuine crash
    (model/tool error) still propagates — but only after the telemetry event (with a
    ``crash: …`` stop_reason) is logged and the diagnostics are attached to the raised
    :class:`AgentLoopError`, so a crashed cycle's telemetry is never lost (#125).

    The **finalize guard** (clean completion only): a ReAct agent ends as soon as the model returns
    a message with no tool call, with no structural requirement that it proposed first. If the loop
    ended with no proposal in the outbox, re-invoke with a salient "submit now" nudge up to
    ``_FINALIZE_RETRIES`` times (history carried forward); best-effort (a raise lets the no-proposal
    outcome stand).
    """
    inputs = {"messages": [{"role": "user", "content": user_message}]}
    config = {"recursion_limit": recursion_limit}
    result: Any = {"messages": []}
    transcript: list[dict[str, Any]] = []

    def emit_new(res: Any) -> None:
        """Render + log every message appended since the last emit (append-only `values` stream)."""
        messages = res.get("messages", []) if isinstance(res, dict) else []
        for index in range(len(transcript), len(messages)):
            entry = render_message(index, messages[index])
            transcript.append(entry)
            log.info(TURN_EVENT, **entry)

    stop_reason: str | None = None
    try:
        for state in agent.stream(inputs, config, stream_mode="values"):
            result = state  # keep the latest full state (messages survive an exhaustion below)
            emit_new(result)
    except (GraphRecursionError, StepBudgetExceeded) as exc:
        emit_new(result)
        # Graceful limit-end (#103): record the reason, return normally (no re-raise).
        stop_reason = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        # A genuine in-loop crash (fatal model error, tool crash). The turns streamed so far are
        # already out; ALSO log the telemetry aggregate — partial but real — with the crash as
        # its stop_reason, so a dead worker's stderr still yields telemetry (#125/#126). Then
        # re-raise wrapped with the diagnostics attached, so the trusted in-process path (which
        # has no stderr capture to reconstruct from) can carry them on its SandboxError.
        emit_new(result)
        telemetry = _extract_telemetry(result)
        telemetry["stop_reason"] = f"crash: {type(exc).__name__}: {exc}"
        log.info(TELEMETRY_EVENT, **telemetry)
        raise AgentLoopError(
            transcript=json.dumps(transcript, default=str).encode("utf-8"),
            telemetry=json.dumps(telemetry, default=str).encode("utf-8"),
        ) from exc
    else:
        if outbox is not None:
            result = _finalize_guard(agent, result, outbox, recursion_limit=recursion_limit)
            emit_new(result)

    telemetry = _extract_telemetry(result)
    if stop_reason is not None:
        telemetry["stop_reason"] = stop_reason
    log.info(TELEMETRY_EVENT, **telemetry)
    return transcript, telemetry


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
                {"messages": [*prior, HumanMessage(content=_FINALIZE_NUDGE.format(outbox=outbox))]},
                {"recursion_limit": recursion_limit},
            )
        except Exception as exc:  # noqa: BLE001 — a guard re-invoke is best-effort; let the
            # no-proposal outcome stand (the host records a recoverable failed cycle, #21).
            log.warning("finalize_guard_invoke_failed", attempt=attempt, error=str(exc))
            return result
    return result


def _write_diagnostics_files(
    transcript: list[dict[str, Any]], telemetry: dict[str, Any], outbox: Path
) -> None:
    """In-process (trusted) path only: persist the returned transcript + telemetry to the local
    outbox under the reserved names, so ``AgentSandbox`` harvests them like any other cycle. Never
    used by the container path (the agent could edit a file there — diagnostics go out as logs)."""
    try:
        (outbox / RESERVED_TELEMETRY_NAME).write_bytes(json.dumps(telemetry).encode("utf-8"))
        (outbox / RESERVED_TRANSCRIPT_NAME).write_bytes(
            json.dumps(transcript, default=str).encode("utf-8")
        )
    except OSError as exc:  # diagnostics are best-effort — never let them fail/mask the cycle
        log.warning("diagnostics_write_failed", error=str(exc))


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
            transcript, telemetry = await asyncio.to_thread(
                run_agent, agent, user_message,
                recursion_limit=effective_recursion_limit(self._recursion_limit, self._step_budget),
                outbox=workspace.outbox(),
            )
        except AgentLoopError as exc:
            # A genuine in-loop failure (model/tool crash) becomes a cycle-level SandboxError, so
            # the control plane can skip the cycle rather than abort the run (#21). The diagnostics
            # ride along (#125): in-process has no stderr capture to reconstruct from, so the
            # transcript + crash telemetry carried on the AgentLoopError are the only copy.
            cause = exc.__cause__
            raise SandboxError(
                f"in-process agent loop failed: {type(cause).__name__}: {cause}",
                transcript=exc.transcript,
                telemetry=exc.telemetry,
            ) from exc
        except Exception as exc:
            # A failure OUTSIDE the loop (agent build, thread bridge) — nothing streamed to carry.
            raise SandboxError(
                f"in-process agent loop failed: {type(exc).__name__}: {exc}"
            ) from exc
        # In-process is trusted (no untrusted agent shell), so write the returned diagnostics to the
        # local outbox for AgentSandbox.harvest. The container path never does this — there the
        # agent shares the worker uid, so diagnostics leave as logs (host-reconstructed), not files.
        _write_diagnostics_files(transcript, telemetry, workspace.outbox())


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
                f"rejected: declared {', '.join(missing)} but no such file is in {outbox}/. "
                f"Write the file(s) to {outbox}/ then call again. "
                f"{outbox}/ now has: {present or 'nothing'}."
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
        f"  1. Write every declared object file to {outbox}/ (this tool rejects the call if a\n"
        f"     file the payload names is not there).\n"
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
