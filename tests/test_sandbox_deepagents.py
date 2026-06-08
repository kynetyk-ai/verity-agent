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

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from verity.sandbox.deepagents_driver import DeepAgentsInProcessDriver  # noqa: E402


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
