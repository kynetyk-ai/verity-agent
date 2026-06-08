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
    _extract_telemetry,
    build_deepagents_agent,
)


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
