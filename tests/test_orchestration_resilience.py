"""Orchestration resilience + widened feedback (issue #21) — a bad cycle must not kill the run.

A sandbox that fails to produce a proposal (timeout / crash / runaway / no-proposal, all surfaced as
:class:`SandboxError`) is recorded as a failed cycle and the run continues; a persistently-broken
sandbox aborts loudly via the consecutive-failure guard. And the §3.5 feedback channel now also
threads back a **rejection** reason and a **sandbox-error** reason, not only shape-errors/refines.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest
from tools.harness.stub_verifier import StubVerifier

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    Operation,
    OperationStatus,
    ProposalEnvelope,
    ProviderRegistry,
    SandboxPort,
    ServedContext,
    VerdictKind,
    VerifierPort,
)
from verity.control_plane.api import (
    ControlPlane,
    IntakeResult,
    OrchestrationError,
    OrchestrationPolicy,
    _feedback_from,
)
from verity.control_plane.commit import CommitOutcome, CommitResult, ShapeError
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import DefaultRetrievalPolicy
from verity.control_plane.store import Decision, SqliteStore
from verity.domains.fake import NOTE, SOURCE, build_fake_domain, build_fake_verifier
from verity.sandbox.errors import SandboxError


@dataclass
class _ScriptedSandbox:
    """A :class:`SandboxPort` double whose ``collect_proposal`` replays scripted items per cycle.

    Each item is either a :class:`ProposalEnvelope` (a proposal) or an ``Exception`` to raise (a
    sandbox failure). The last item repeats once exhausted, so a single error scripts an
    always-failing sandbox.
    """

    items: list[object]
    cursor: int = 0
    regenerations: int = 0

    async def serve_context(self, context: ServedContext) -> None:
        return None

    async def collect_proposal(self) -> ProposalEnvelope:
        item = self.items[min(self.cursor, len(self.items) - 1)]
        self.cursor += 1
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, ProposalEnvelope)
        return item

    async def regenerate(self) -> None:
        self.regenerations += 1

    async def provision(self) -> None:
        return None

    async def teardown(self) -> None:
        return None

    async def health(self) -> bool:
        return True


def _note(note_id: str) -> ProposalEnvelope:
    artifact = Artifact(note_id, NOTE, {"text": "hi"}, ArtifactStatus.PROPOSED, "agent", "t1")
    op = Operation(f"op-{note_id}", "author", ("src",), note_id, OperationStatus.SUCCESS, "t1")
    return ProposalEnvelope(artifact=artifact, operation=op)


def _build(
    tmp_path: Path, *, items: list[object], policy: OrchestrationPolicy
) -> tuple[ControlPlane, _ScriptedSandbox, SqliteStore]:
    store = SqliteStore()
    store.propose(
        Artifact("src", SOURCE, {"raw": 1}, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation("op-src", "load", (), "src", OperationStatus.SUCCESS, "t0"),
    )
    domain = build_fake_domain()
    sandbox = _ScriptedSandbox(items=items)
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("scripted", lambda: sandbox)
    vp.register("real", lambda: build_fake_verifier())
    cp = ControlPlane(store, policy=policy, sandbox_providers=sp, verifier_providers=vp)
    config = TaskConfig(
        task_id="t1", instructions="make notes", domain_instructions="a Note has text",
        schema=domain.schema, gated_types=domain.gated_types, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator, sandbox_key="scripted", verifier_key="real",
    )
    asyncio.run(cp.configure(config))
    return cp, sandbox, store


# --------------------------------------------------------------------------- run() resilience


def test_a_failed_cycle_is_recorded_and_the_run_continues(tmp_path: Path) -> None:
    # cycle 1's sandbox times out; cycle 2 proposes a good note -> accepted. The run does NOT crash.
    cp, sandbox, store = _build(
        tmp_path,
        items=[SandboxError("sandbox container timed out after 1500.0s"), _note("n1")],
        policy=OrchestrationPolicy(max_cycles=5, stop_on_accept=True),
    )
    results = asyncio.run(cp.run("t1", goal="author a note"))

    assert results[0].sandbox_error is not None and results[0].entered_protocol is False
    assert results[1].commit is not None and results[1].commit.outcome is CommitOutcome.ACCEPTED
    assert store.get_artifact("n1") is not None  # cycle 2 committed despite cycle 1's failure
    assert sandbox.regenerations == 2  # the broken workspace is still regenerated each cycle


def test_too_many_consecutive_failures_abort_the_run(tmp_path: Path) -> None:
    cp, _sandbox, _store = _build(
        tmp_path,
        items=[SandboxError("runaway")],  # repeats -> always fails
        policy=OrchestrationPolicy(max_cycles=20, max_consecutive_sandbox_failures=2),
    )
    with pytest.raises(OrchestrationError, match="consecutive failed cycles"):
        asyncio.run(cp.run("t1", goal="author a note"))


def test_a_success_resets_the_failure_counter(tmp_path: Path) -> None:
    # fail, succeed (reset), fail, fail -> the two trailing failures do not trip a cap of 2 mid-run
    # because the success in between reset the counter; the run completes at max_cycles.
    cp, _sandbox, store = _build(
        tmp_path,
        items=[SandboxError("a"), _note("n1"), SandboxError("b"), SandboxError("c")],
        policy=OrchestrationPolicy(max_cycles=4, max_consecutive_sandbox_failures=3),
    )
    results = asyncio.run(cp.run("t1", goal="author a note"))  # does not raise
    assert len(results) == 4
    assert results[1].commit is not None and results[1].commit.outcome is CommitOutcome.ACCEPTED


# --------------------------------------------------------------------------- widened feedback


def _rejected_commit(rationale: str) -> CommitResult:
    decision = Decision(
        artifact_id="a", gate="selection", verdict=VerdictKind.REJECT,
        rationale=rationale, defects=None, score=0.1, created_at="t1",
    )
    return CommitResult(
        outcome=CommitOutcome.REJECTED, artifact_id="a", status=ArtifactStatus.REJECTED,
        decisions=(decision,),
    )


def test_feedback_threads_a_rejection_reason() -> None:
    result = IntakeResult(entered_protocol=True, commit=_rejected_commit("net 0.32 < bar 0.80"))
    assert _feedback_from(result) == "rejected: net 0.32 < bar 0.80"


def test_feedback_threads_a_sandbox_error() -> None:
    result = IntakeResult(entered_protocol=False, sandbox_error="timed out after 1500.0s")
    feedback = _feedback_from(result)
    assert feedback.startswith("sandbox-error:") and "timed out" in feedback


def test_feedback_still_handles_shape_error_and_refine() -> None:
    shape = IntakeResult(entered_protocol=False, shape_error=ShapeError("missing 'text'"))
    assert _feedback_from(shape) == "shape-error: missing 'text'"
    refine = IntakeResult(
        entered_protocol=True,
        commit=CommitResult(
            outcome=CommitOutcome.REVISED, artifact_id="a", status=ArtifactStatus.REVISED,
            defects=("tone",),
        ),
    )
    assert _feedback_from(refine) == "refine: tone"


def test_sandbox_error_feedback_takes_precedence_and_a_clean_accept_has_none() -> None:
    # a sandbox failure outranks everything; a plain accepted cycle threads no correction.
    both = IntakeResult(entered_protocol=False, sandbox_error="boom",
                        shape_error=ShapeError("ignored"))
    assert _feedback_from(both).startswith("sandbox-error:")
    accepted = IntakeResult(
        entered_protocol=True,
        commit=CommitResult(CommitOutcome.ACCEPTED, "a", ArtifactStatus.ACCEPTED),
    )
    assert _feedback_from(accepted) == ""


def test_stub_sandbox_double_satisfies_the_port() -> None:
    assert isinstance(_ScriptedSandbox(items=[]), SandboxPort)
    assert isinstance(StubVerifier(), VerifierPort)
