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
from hypothesis import given, settings
from hypothesis import strategies as st
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
        shape_validator=domain.shape_validator, object_namer=lambda _a: frozenset(),
        sandbox_key="scripted", verifier_key="real",
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


def test_a_failed_cycle_salvages_its_transcript(tmp_path: Path) -> None:
    # A no-proposal cycle (e.g. a recursion/budget limit-end) carries its transcript on the
    # SandboxError; the control plane stores it + references it in the report, so the failure stays
    # debuggable — the case that matters most.
    transcript = b'[{"index": 0, "type": "AIMessage", "content": "looped to the limit"}]'
    cp, _sandbox, store = _build(
        tmp_path,
        items=[SandboxError("the agent produced no proposal", transcript=transcript)],
        policy=OrchestrationPolicy(max_cycles=1, max_consecutive_sandbox_failures=0),
    )
    results = asyncio.run(cp.run("t1", goal="author a note"))

    assert results[0].entered_protocol is False and results[0].transcript_ref is not None
    ref = cp.run_report("t1").cycles[0].transcript_ref
    assert ref is not None
    assert store.get_object(ref) == transcript  # the salvaged transcript resolves from the store


def test_a_failed_cycle_records_its_telemetry_and_stop_reason(tmp_path: Path) -> None:
    # A no-proposal cycle carries its telemetry (incl. stop_reason) on the SandboxError; the control
    # plane records it on the cycle so the report names *why* it stopped (e.g. a graceful
    # StepBudgetExceeded), not an opaque failure.
    telemetry = b'{"stop_reason": "StepBudgetExceeded: budget exhausted", "model_steps": 80}'
    cp, _sandbox, _store = _build(
        tmp_path,
        items=[SandboxError("the agent produced no proposal", telemetry=telemetry)],
        policy=OrchestrationPolicy(max_cycles=1, max_consecutive_sandbox_failures=0),
    )
    asyncio.run(cp.run("t1", goal="author a note"))

    cycle = cp.run_report("t1").cycles[0]
    assert cycle.agent_telemetry is not None
    assert cycle.agent_telemetry["stop_reason"] == "StepBudgetExceeded: budget exhausted"
    assert cycle.agent_telemetry["model_steps"] == 80


def test_an_unexpected_boundary_failure_is_recorded_not_fatal(tmp_path: Path) -> None:
    # S1: a failure that is NEITHER SandboxError nor GateUnavailable (here a raw RuntimeError, the
    # shape a disk-full store-IO error or other predictable boundary failure takes) must still be a
    # recorded failed cycle the run survives — not an un-recorded run abort. The catch-all records
    # it under `sandbox_error` (prefixed) so the failure cap still bounds a persistent failure.
    cp, sandbox, store = _build(
        tmp_path,
        items=[RuntimeError("disk full writing object"), _note("n1")],
        policy=OrchestrationPolicy(max_cycles=5, stop_on_accept=True),
    )
    results = asyncio.run(cp.run("t1", goal="author a note"))  # does NOT raise

    assert results[0].entered_protocol is False
    assert results[0].sandbox_error is not None and "cycle failed" in results[0].sandbox_error
    assert "disk full" in results[0].sandbox_error
    assert results[1].commit is not None and results[1].commit.outcome is CommitOutcome.ACCEPTED
    assert store.get_artifact("n1") is not None  # the run recovered and committed next cycle
    assert sandbox.regenerations == 2  # the workspace is regenerated even on the unexpected failure


@settings(max_examples=40, deadline=None)
@given(kinds=st.lists(st.sampled_from(["good", "sandbox", "boom"]), min_size=1, max_size=6))
def test_any_failure_sequence_is_fully_recorded_and_never_aborts(kinds: list[str]) -> None:
    # Property (degrade-don't-crash, §3.4): with the failure cap disabled, ANY interleaving of
    # good proposals, SandboxErrors, and unexpected boundary failures runs to completion without
    # raising, and EVERY cycle is recorded with a definite outcome (success or a populated error)
    # — the RunReport never silently drops a cycle.
    items: list[object] = []
    for i, kind in enumerate(kinds):
        if kind == "good":
            items.append(_note(f"n{i}"))
        elif kind == "sandbox":
            items.append(SandboxError(f"sandbox boom {i}"))
        else:
            items.append(RuntimeError(f"boundary boom {i}"))
    cp, _sandbox, _store = _build(
        Path("."),  # _build uses an in-memory SqliteStore; the path is unused
        items=items,
        policy=OrchestrationPolicy(
            max_cycles=len(items), stop_on_accept=False, max_consecutive_sandbox_failures=0
        ),
    )
    results = asyncio.run(cp.run("t1", goal="author a note"))  # must never raise

    assert len(results) == len(items)
    for r in results:
        assert r.entered_protocol or r.sandbox_error or r.gate_error or r.shape_error is not None
    assert cp.run_report("t1").summary.cycles_run == len(items)  # one recorded cycle per item


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
