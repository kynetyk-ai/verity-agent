"""The Deep Agents driver binding (ROADMAP Phase 3) — requires the ``sandbox`` extra.

Proves the real :class:`DeepAgentsInProcessDriver` drives a Deep Agents loop whose tool call is the
generated per-operation propose tool, producing a typed envelope and (live) an accepted Note. The
offline tests use a scripted fake chat model — no network; the ``@pytest.mark.live`` smoke needs a
real key and is auto-skipped without one.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    Operation,
    OperationStatus,
    ProviderRegistry,
    SandboxPort,
    ServedContext,
    VerifierPort,
)
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.commit import CommitOutcome
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import (
    ArtifactTypeDef,
    DefaultRetrievalPolicy,
    OperationSignature,
    SchemaRegistry,
)
from verity.control_plane.store import SqliteStore
from verity.domains.fake import NOTE, SOURCE, build_fake_domain, build_fake_verifier
from verity.sandbox import AgentSandbox

pytest.importorskip("deepagents")

from deepagents.backends.filesystem import FilesystemBackend  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from verity.sandbox.deepagents_driver import (  # noqa: E402
    DeadlineMiddleware,
    DeepAgentsInProcessDriver,
    StepBudgetExceeded,
    StepBudgetMiddleware,
    _extract_telemetry,
    _make_propose_tool,
    build_deepagents_agent,
)
from verity.sandbox.descriptor import RESERVED_PROPOSAL_NAME  # noqa: E402


class ToolCallingFakeModel(GenericFakeChatModel):
    """A scripted fake chat model that survives Deep Agents' ``bind_tools`` (it ignores tools)."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def _author_call(text: str, parent: str = "src-1") -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "author",
                    "args": {"parents": [parent], "payload": {"text": text}, "rationale": "fits"},
                    "id": "call-1",
                }
            ],
        ),
        AIMessage(content="done"),
    ]


def _author_payload(payload: dict[str, Any], parent: str = "src-1") -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "author", "args": {"parents": [parent], "payload": payload}, "id": "c"}
            ],
        ),
        AIMessage(content="done"),
    ]


def _note_schema() -> SchemaRegistry:
    schema = SchemaRegistry()
    schema.register_type(ArtifactTypeDef(SOURCE, is_root=True))
    schema.register_type(ArtifactTypeDef(NOTE))
    schema.register_operation(OperationSignature("author", inputs=(SOURCE,), output=NOTE))
    return schema


def test_agent_has_native_tools_and_no_custom_run_shell(tmp_path: Path) -> None:
    # The agent gets Deep Agents' native coding tools (file ops + execute) plus the propose tool —
    # and NOT a bespoke run_shell (cleanup: shell execution is the backend's `execute`).
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    fake = ToolCallingFakeModel(messages=iter([AIMessage(content="x")]))
    agent = build_deepagents_agent(
        model=fake,
        operations=_note_schema().operations(),
        outbox=outbox,
        system_prompt="s",
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=False),
    )
    tools = set(agent.get_graph().nodes["tools"].data.tools_by_name)

    assert "run_shell" not in tools  # removed in favour of the backend's native execute
    assert "author" in tools  # the generated propose tool (the typed output seam)
    assert {"read_file", "write_file", "edit_file", "ls", "glob", "grep", "execute"} <= tools


def test_fake_model_drives_the_generated_propose_tool(tmp_path: Path) -> None:
    model = ToolCallingFakeModel(messages=iter(_author_call("hello world")))
    sandbox = AgentSandbox(
        root=tmp_path / "ws",
        driver=DeepAgentsInProcessDriver(model=model),
        schema=_note_schema(),
        proposer_identity="deepagents:fake",
        clock=lambda: "t1",
        id_source=lambda: "note-1",
    )

    async def go() -> Any:
        await sandbox.provision()
        await sandbox.serve_context(ServedContext(system_prompt="SYS", tail="Source id=src-1"))
        return await sandbox.collect_proposal()

    env = asyncio.run(go())

    assert env.artifact.type == NOTE and env.artifact.payload == {"text": "hello world"}
    assert env.operation.op_name == "author" and env.operation.parents == ("src-1",)
    assert env.metadata == "fits" and dict(env.objects) == {}


def _seed_source(store: SqliteStore, art_id: str) -> None:
    store.propose(
        Artifact(art_id, SOURCE, {"raw": 1}, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation(f"op-{art_id}", "load", (), art_id, OperationStatus.SUCCESS, "t0"),
    )


def _configure(tmp_path: Path, model: Any) -> tuple[ControlPlane, SqliteStore]:
    domain = build_fake_domain()
    store = SqliteStore()
    sandbox = AgentSandbox(
        root=tmp_path / "ws",
        driver=DeepAgentsInProcessDriver(model=model),
        schema=domain.schema,
        proposer_identity="deepagents:claude",
        clock=lambda: "t1",
        id_source=lambda: "note-1",
    )
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("deepagents-claude", lambda: sandbox)
    vp.register("fake", lambda: build_fake_verifier())
    cp = ControlPlane(
        store,
        policy=OrchestrationPolicy(stop_on_accept=True),
        sandbox_providers=sp,
        verifier_providers=vp,
    )
    config = TaskConfig(
        task_id="t1",
        instructions="author a single Note with a 'text' field, then submit it",
        domain_instructions="a Note payload is an object with a 'text' field",
        schema=domain.schema,
        gated_types=domain.gated_types,
        retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator,
        object_namer=lambda _a: frozenset(),
        sandbox_key="deepagents-claude",
        verifier_key="fake",
    )
    asyncio.run(cp.configure(config))
    return cp, store


def test_full_cycle_through_deepagents_accepts(tmp_path: Path) -> None:
    cp, store = _configure(tmp_path, ToolCallingFakeModel(messages=iter(_author_call("a note"))))
    _seed_source(store, "src-1")

    result = asyncio.run(cp.run_cycle("t1", goal="author a note from src-1"))

    assert result.commit is not None and result.commit.outcome is CommitOutcome.ACCEPTED
    got = store.get_artifact("note-1")
    assert got is not None and got.status is ArtifactStatus.ACCEPTED


def test_cross_cycle_shape_error_then_accept(tmp_path: Path) -> None:
    # Cycle 1 proposes a Note with no "text" (shape error, records nothing); the correction feeds
    # back; cycle 2 fixes it on a freshly regenerated workspace and commits. Proves cross-cycle
    # operation + feedback threading + ephemerality through the real sandbox.
    messages = _author_payload({"summary": "x"}) + _author_payload({"text": "a kept note"})
    cp, store = _configure(tmp_path, ToolCallingFakeModel(messages=iter(messages)))
    _seed_source(store, "src-1")

    first = asyncio.run(cp.run_cycle("t1", goal="author a note"))
    assert not first.entered_protocol and first.shape_error is not None
    assert store.get_artifact("note-1") is None  # the malformed proposal recorded nothing

    feedback = f"shape-error: {first.shape_error.message}"
    second = asyncio.run(cp.run_cycle("t1", goal="author a note", feedback=feedback))
    assert second.commit is not None and second.commit.outcome is CommitOutcome.ACCEPTED
    got = store.get_artifact("note-1")
    assert got is not None and got.status is ArtifactStatus.ACCEPTED


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
def test_live_claude_authors_an_accepted_note(tmp_path: Path) -> None:
    cp, store = _configure(tmp_path, "anthropic:claude-sonnet-4-6")
    _seed_source(store, "src-1")

    goal = "Author a Note whose text summarizes Source src-1; use src-1 as the parent."
    result = asyncio.run(cp.run_cycle("t1", goal=goal))

    assert result.commit is not None and result.commit.outcome is CommitOutcome.ACCEPTED
    got = store.get_artifact("note-1")
    assert got is not None and got.status is ArtifactStatus.ACCEPTED


# --------------------------------------------------------------------------- 5.3b telemetry


def test_extract_telemetry_sums_usage_and_counts_steps() -> None:
    # Two model steps: one with a tool call + usage, one final answer with usage.
    result = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "author", "args": {}, "id": "c1"}],
                usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                response_metadata={"model_name": "claude-x"},
            ),
            AIMessage(
                content="done",
                usage_metadata={"input_tokens": 130, "output_tokens": 5, "total_tokens": 135},
                response_metadata={"model_name": "claude-x"},
            ),
        ]
    }
    telemetry = _extract_telemetry(result)
    assert telemetry == {
        "input_tokens": 230, "output_tokens": 25, "total_tokens": 255,
        "model_steps": 2, "tool_calls": 1, "model": "claude-x",
    }


def test_extract_telemetry_is_null_safe_for_a_usage_less_model() -> None:
    result = {"messages": [AIMessage(content="hi")]}  # no usage_metadata, no tool calls
    telemetry = _extract_telemetry(result)
    assert telemetry["model_steps"] == 1 and telemetry["total_tokens"] == 0
    assert telemetry["model"] is None  # never assumed


def test_telemetry_flows_through_the_loop_into_the_run_report(tmp_path: Path) -> None:
    # A real in-process Deep Agents run with the fake model writes __telemetry__.json, harvested
    # onto the envelope and surfaced in the RunReport (tokens 0 with the fake model, steps counted).
    cp, store = _configure(tmp_path, ToolCallingFakeModel(messages=iter(_author_call("a note"))))
    _seed_source(store, "src-1")
    asyncio.run(cp.run_cycle("t1", goal="author a note from src-1"))

    report = cp.run_report("t1")
    cycle = report.cycles[0]
    assert cycle.agent_telemetry is not None and cycle.agent_telemetry["model_steps"] >= 1
    assert report.agent_telemetry is not None  # the run total is populated (no longer a null slot)
    assert report.agent_telemetry["model_steps"] >= 1


# --------------------------------------------------------------------------- 5.2 deadline hook


def test_deadline_middleware_injects_a_wrap_up_once_past_the_threshold() -> None:
    ticks = iter([0.0, 50.0, 85.0, 95.0])  # start, then three before_model calls
    mw = DeadlineMiddleware(100.0, now=lambda: next(ticks), warn_fraction=0.8)
    mw.before_agent(None, None)  # start = 0.0

    assert mw.before_model(None, None) is None  # elapsed 50 < 80: no nudge yet
    injected = mw.before_model(None, None)  # elapsed 85 >= 80: inject once
    assert injected is not None
    assert "submit" in injected["messages"][0].content  # the wrap-up nudge
    assert mw.before_model(None, None) is None  # elapsed 95: already warned, silent


def test_deadline_middleware_is_silent_with_no_budget_pressure() -> None:
    mw = DeadlineMiddleware(1000.0, now=lambda: 0.0)
    mw.before_agent(None, None)
    assert mw.before_model(None, None) is None  # elapsed 0: nothing injected


class _Clock:
    """A settable monotonic clock for deterministic middleware time tests."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class _FakeModelRequest:
    """A minimal ModelRequest stand-in: a system prompt + the immutable ``override`` contract."""

    def __init__(self, system_prompt: str, system_message: object | None = None) -> None:
        self._system_prompt = system_prompt
        self.system_message = system_message

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def override(self, **overrides: object) -> _FakeModelRequest:
        # Immutable, like the real ModelRequest.override: a NEW request, original untouched.
        return _FakeModelRequest(self._system_prompt, overrides.get("system_message"))


def test_deadline_middleware_recurring_note_is_ephemeral_50_to_80() -> None:
    # #102: from the halfway mark the recurring tier injects a transient time note into the model
    # request via wrap_model_call — augmenting the system prompt for THAT call only, never the
    # durable request/state (no context bloat).
    clock = _Clock()
    mw = DeadlineMiddleware(100.0, now=clock)
    mw.before_agent(None, None)
    seen: list[Any] = []

    def handler(req: Any) -> Any:
        seen.append(req)
        return req

    # Below 50%: nothing injected — the original request flows through untouched.
    clock.t = 10.0
    original = _FakeModelRequest("base prompt")
    mw.wrap_model_call(original, handler)  # type: ignore[arg-type]
    assert seen[-1] is original and seen[-1].system_message is None

    # 50%–80%: an ephemeral note augments the system prompt for this call on a COPY; the original
    # request object is not mutated, so nothing persists across steps.
    clock.t = 60.0
    original2 = _FakeModelRequest("base prompt")
    mw.wrap_model_call(original2, handler)  # type: ignore[arg-type]
    passed = seen[-1]
    assert passed is not original2
    assert original2.system_message is None  # durable request untouched
    assert "time budget" in passed.system_message.content.lower()
    assert "base prompt" in passed.system_message.content  # original prompt preserved


def test_deadline_middleware_recurring_note_is_throttled() -> None:
    # #102: the recurring note is throttled to a coarse cadence (~once per 10% of the budget), so it
    # paces the agent rather than firing on every model step.
    clock = _Clock()
    mw = DeadlineMiddleware(100.0, now=clock)  # throttle interval = 10s
    mw.before_agent(None, None)
    injected: list[bool] = []

    def handler(req: Any) -> Any:
        injected.append(getattr(req, "system_message", None) is not None)
        return req

    clock.t = 60.0
    mw.wrap_model_call(_FakeModelRequest("p"), handler)  # type: ignore[arg-type]
    clock.t = 61.0  # within 10s of the last note -> throttled
    mw.wrap_model_call(_FakeModelRequest("p"), handler)  # type: ignore[arg-type]
    clock.t = 72.0  # past the throttle window -> fires again
    mw.wrap_model_call(_FakeModelRequest("p"), handler)  # type: ignore[arg-type]
    clock.t = 85.0  # past 80% -> recurring tier stops (the one-time pivot is before_model's job)
    mw.wrap_model_call(_FakeModelRequest("p"), handler)  # type: ignore[arg-type]
    assert injected == [True, False, True, False]


def test_agent_builds_with_a_deadline(tmp_path: Path) -> None:
    # Smoke: building with a deadline (which appends DeadlineMiddleware) succeeds and keeps tools.
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    agent = build_deepagents_agent(
        model=ToolCallingFakeModel(messages=iter([AIMessage(content="x")])),
        operations=_note_schema().operations(),
        outbox=outbox,
        system_prompt="s",
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=False),
        deadline_s=300.0,
    )
    assert "author" in set(agent.get_graph().nodes["tools"].data.tools_by_name)


def test_step_budget_middleware_nudges_then_halts() -> None:
    mw = StepBudgetMiddleware(5, warn_fraction=0.8)  # warn at step 4, halt past step 5
    assert mw.before_model(None, None) is None  # step 1
    assert mw.before_model(None, None) is None  # step 2
    assert mw.before_model(None, None) is None  # step 3
    injected = mw.before_model(None, None)  # step 4: the one-time wrap-up nudge
    assert injected is not None and "submit" in injected["messages"][0].content
    assert mw.before_model(None, None) is None  # step 5: within budget, already warned
    with pytest.raises(StepBudgetExceeded, match="budget exhausted"):
        mw.before_model(None, None)  # step 6: over budget -> hard stop


def test_step_budget_middleware_is_silent_well_under_budget() -> None:
    mw = StepBudgetMiddleware(100)
    for _ in range(3):
        assert mw.before_model(None, None) is None


# --- propose tool: outbox file-presence check + the tested attestation nudge ---

_SUB_SIG = OperationSignature(
    "submit", inputs=("Dataset",), output="Submission",
    object_payload_keys=("entrypoint", "requirements"),
)
_SUB_PAYLOAD = {"entrypoint": "submission.py", "requirements": "requirements.txt"}


def _seed_outbox(tmp_path: Path, *names: str) -> Path:
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    for name in names:
        (outbox / name).write_text("x\n")
    return outbox


def test_propose_rejects_when_a_declared_object_file_is_missing(tmp_path: Path) -> None:
    # The op declares entrypoint/requirements as outbox files; a missing one is an in-cycle reject
    # (records nothing), so the agent can fix it before a gate cycle is spent.
    outbox = _seed_outbox(tmp_path, "requirements.txt")  # entrypoint NOT written
    propose = _make_propose_tool(_SUB_SIG, outbox)
    result = propose(parents=["ds"], payload=_SUB_PAYLOAD)
    assert result.startswith("rejected:")
    assert "entrypoint='submission.py'" in result
    assert not (outbox / RESERVED_PROPOSAL_NAME).exists()  # nothing recorded


def test_propose_records_when_declared_files_present(tmp_path: Path) -> None:
    outbox = _seed_outbox(tmp_path, "submission.py", "requirements.txt")
    propose = _make_propose_tool(_SUB_SIG, outbox)
    result = propose(parents=["ds"], payload=_SUB_PAYLOAD, tested=True)
    assert "proposal recorded" in result and "tested=false" not in result
    assert (outbox / RESERVED_PROPOSAL_NAME).exists()


def test_propose_warns_but_records_when_not_attested_tested(tmp_path: Path) -> None:
    # tested defaults False -> still records (the nudge is soft, not enforced) but warns.
    outbox = _seed_outbox(tmp_path, "submission.py", "requirements.txt")
    propose = _make_propose_tool(_SUB_SIG, outbox)
    result = propose(parents=["ds"], payload=_SUB_PAYLOAD)
    assert "proposal recorded" in result and "tested=false" in result
    assert (outbox / RESERVED_PROPOSAL_NAME).exists()


def test_propose_skips_presence_check_when_op_declares_no_object_keys(tmp_path: Path) -> None:
    # An op with no object_payload_keys (the default) records without any outbox file present.
    outbox = _seed_outbox(tmp_path)
    sig = OperationSignature("author", inputs=("Source",), output="Note")
    propose = _make_propose_tool(sig, outbox)
    result = propose(parents=["src"], payload={"text": "a note"})
    assert "proposal recorded" in result
    assert (outbox / RESERVED_PROPOSAL_NAME).exists()


def test_agent_builds_with_a_step_budget(tmp_path: Path) -> None:
    # Smoke: building with a step budget (appends StepBudgetMiddleware) succeeds and keeps tools.
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    agent = build_deepagents_agent(
        model=ToolCallingFakeModel(messages=iter([AIMessage(content="x")])),
        operations=_note_schema().operations(),
        outbox=outbox,
        system_prompt="s",
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=False),
        step_budget=10,
    )
    assert "author" in set(agent.get_graph().nodes["tools"].data.tools_by_name)


# --------------------------------------------------------------------------- 5.4 sandbox tools


def test_extra_tools_are_bound_when_selected(tmp_path: Path) -> None:
    from verity.sandbox.tools import resolve_tools

    outbox = tmp_path / "outbox"
    outbox.mkdir()
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    fake = ToolCallingFakeModel(messages=iter([AIMessage(content="x")]))

    without = build_deepagents_agent(
        model=fake, operations=_note_schema().operations(), outbox=outbox,
        system_prompt="s", backend=backend,
    )
    assert "read_pdf" not in set(without.get_graph().nodes["tools"].data.tools_by_name)

    with_tool = build_deepagents_agent(
        model=fake, operations=_note_schema().operations(), outbox=outbox,
        system_prompt="s", backend=backend, extra_tools=resolve_tools(("read_pdf",)),
    )
    bound = set(with_tool.get_graph().nodes["tools"].data.tools_by_name)
    assert "read_pdf" in bound and "author" in bound  # the seam tool + the propose tool coexist


# ------------------------------------------------------------------------- the finalize guard


class _FakeAgent:
    """A stand-in langgraph agent: each ``invoke`` appends an AIMessage; an optional ``submit_on``
    call number writes the proposal descriptor to the outbox (simulating the agent finally calling
    its propose tool)."""

    def __init__(self, outbox: Path, *, submit_on: int | None) -> None:
        self.outbox = outbox
        self.submit_on = submit_on
        self.calls = 0

    def invoke(self, state: dict, config: dict) -> dict:  # noqa: ARG002 - mirror the real signature
        self.calls += 1
        if self.submit_on is not None and self.calls == self.submit_on:
            (self.outbox / RESERVED_PROPOSAL_NAME).write_text("{}", encoding="utf-8")
        return {"messages": [*state["messages"], AIMessage(content=f"step {self.calls}")]}


def test_finalize_guard_nudges_until_the_agent_submits(tmp_path: Path) -> None:
    from verity.sandbox.deepagents_driver import run_agent

    outbox = tmp_path
    agent = _FakeAgent(outbox, submit_on=2)  # submits on the first nudge
    run_agent(agent, "go", outbox=outbox)
    assert agent.calls == 2  # initial invoke + one finalize nudge
    assert (outbox / RESERVED_PROPOSAL_NAME).is_file()
    assert not (outbox / "__transcript__.json").exists()  # succeeded → no failure diagnostic


def test_finalize_guard_exhausts_and_persists_transcript(tmp_path: Path) -> None:
    import json

    from verity.sandbox.deepagents_driver import _FINALIZE_RETRIES, run_agent

    outbox = tmp_path
    agent = _FakeAgent(outbox, submit_on=None)  # never submits
    run_agent(agent, "go", outbox=outbox)
    assert agent.calls == 1 + _FINALIZE_RETRIES  # initial + bounded nudges, then give up
    assert not (outbox / RESERVED_PROPOSAL_NAME).is_file()
    transcript = json.loads((outbox / "__transcript__.json").read_text())
    assert transcript and transcript[-1]["type"] == "AIMessage"  # the rendered trace is captured


def test_finalize_guard_survives_a_raising_reinvoke(tmp_path: Path) -> None:
    # A guard re-invoke that raises (a step/recursion limit) is best-effort: no proposal stands,
    # the transcript from the last good result is still persisted, and run_agent does not raise.
    from verity.sandbox.deepagents_driver import run_agent

    class _RaisingAgent(_FakeAgent):
        def invoke(self, state: dict, config: dict) -> dict:
            if self.calls >= 1:
                raise RuntimeError("recursion limit")
            return super().invoke(state, config)

    outbox = tmp_path
    run_agent(_RaisingAgent(outbox, submit_on=None), "go", outbox=outbox)
    assert not (outbox / RESERVED_PROPOSAL_NAME).is_file()
    assert (outbox / "__transcript__.json").exists()
