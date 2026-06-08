"""Gate-boundary degrade-don't-crash (ROADMAP 5.1).

The proposal side already survives a ``SandboxError`` (issue #21); this proves the *gating* side is
symmetric. A :class:`GateUnavailable` — the verifier could not render a verdict (a transient infra
failure that exhausted its retries, or a dispatch that hung past the backstop) — is a **recorded
failed cycle, not a run abort**. A property test asserts the invariant across both failure kinds:
*a failed step is recorded, not fatal.*
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.test_orchestration_resilience import _ScriptedSandbox
from verity.contracts import (
    Artifact,
    ArtifactStatus,
    GateUnavailable,
    Operation,
    OperationStatus,
    ProposalEnvelope,
    ProviderRegistry,
    SandboxPort,
    VerdictBundle,
    VerifierPort,
    VerifierRequest,
)
from verity.control_plane.api import (
    ControlPlane,
    OrchestrationError,
    OrchestrationPolicy,
    _feedback_from,
)
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import DefaultRetrievalPolicy
from verity.control_plane.store import SqliteStore
from verity.domains.fake import NOTE, SOURCE, build_fake_domain, build_fake_verifier
from verity.sandbox.errors import SandboxError

# ----------------------------------------------------------------- proposal + verifier doubles

_GATE_FAIL_MARKER = "_gate_fail"


def _note(note_id: str, *, fail_gate: bool = False) -> ProposalEnvelope:
    payload = {"text": "hi"} | ({_GATE_FAIL_MARKER: True} if fail_gate else {})
    artifact = Artifact(note_id, NOTE, payload, ArtifactStatus.PROPOSED, "agent", "t1")
    op = Operation(f"op-{note_id}", "author", ("src",), note_id, OperationStatus.SUCCESS, "t1")
    return ProposalEnvelope(artifact=artifact, operation=op)


class _FlakyVerifier:
    """Wraps the real fake verifier but raises :class:`GateUnavailable` for marked proposals.

    Models a transient verifier-infra failure (e.g. its model/container call exhausted retries) on a
    per-proposal basis, so accept and gate-unavailable can be interleaved in one run.
    """

    def __init__(self) -> None:
        self._inner = build_fake_verifier()

    @property
    def identity(self) -> str:
        return self._inner.identity

    async def dispatch(self, request: VerifierRequest) -> VerdictBundle:
        payload = request.proposal.payload
        if isinstance(payload, dict) and payload.get(_GATE_FAIL_MARKER):
            raise GateUnavailable("verifier backend unreachable (transient)")
        return await self._inner.dispatch(request)

    async def provision(self) -> None:
        await self._inner.provision()

    async def teardown(self) -> None:
        await self._inner.teardown()

    async def health(self) -> bool:
        return await self._inner.health()


class _HangingVerifier:
    """A verifier whose dispatch never returns — exercises the dispatch-timeout backstop."""

    identity = "hanging-verifier"

    async def dispatch(self, request: VerifierRequest) -> VerdictBundle:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")  # pragma: no cover

    async def provision(self) -> None:
        return None

    async def teardown(self) -> None:
        return None

    async def health(self) -> bool:
        return True


def _build(
    *,
    items: list[object],
    policy: OrchestrationPolicy,
    verifier: VerifierPort,
    dispatch_timeout_s: float | None = None,
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
    vp.register("real", lambda: verifier)
    cp = ControlPlane(
        store, policy=policy, sandbox_providers=sp, verifier_providers=vp,
        dispatch_timeout_s=dispatch_timeout_s,
    )
    config = TaskConfig(
        task_id="t1", instructions="make notes", domain_instructions="a Note has text",
        schema=domain.schema, gated_types=domain.gated_types, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator, sandbox_key="scripted", verifier_key="real",
    )
    asyncio.run(cp.configure(config))
    return cp, sandbox, store


# ----------------------------------------------------------------- the gating-boundary degrade


def test_gate_unavailable_is_recorded_and_the_run_continues() -> None:
    # cycle 1's gate is unavailable; cycle 2 proposes a note the gate accepts -> run does not die.
    cp, sandbox, store = _build(
        items=[_note("n1", fail_gate=True), _note("n2")],
        policy=OrchestrationPolicy(max_cycles=5, stop_on_accept=True),
        verifier=_FlakyVerifier(),
    )
    results = asyncio.run(cp.run("t1", goal="author a note"))

    assert results[0].gate_error is not None and results[0].entered_protocol is False
    assert results[0].sandbox_error is None  # it was the gate, not the sandbox
    assert results[1].commit is not None and results[1].commit.status is ArtifactStatus.ACCEPTED
    assert store.get_artifact("n2") is not None  # cycle 2 committed despite cycle 1's gate failure
    assert sandbox.regenerations == 2  # the workspace is regenerated even on a gate failure


def test_a_hung_verifier_times_out_into_a_gate_error() -> None:
    # The dispatch backstop converts a verifier that never responds into a recoverable gate failure.
    cp, _sandbox, _store = _build(
        items=[_note("n1")],
        policy=OrchestrationPolicy(max_cycles=1),
        verifier=_HangingVerifier(),
        dispatch_timeout_s=0.2,
    )
    results = asyncio.run(cp.run("t1", goal="author a note"))

    assert results[0].gate_error is not None and "did not respond" in results[0].gate_error
    assert results[0].entered_protocol is False


def test_gate_failures_count_toward_the_failure_breaker() -> None:
    # A persistently unavailable gate aborts the run loudly, like a persistently broken sandbox.
    # Distinct ids per cycle (a gate-failed proposal is recorded `proposed`, so reusing an id would
    # be a genuine StoreError rather than a gate failure — the real sandbox mints a fresh id).
    cp, _sandbox, _store = _build(
        items=[_note("n1", fail_gate=True), _note("n2", fail_gate=True)],
        policy=OrchestrationPolicy(max_cycles=20, max_consecutive_sandbox_failures=2),
        verifier=_FlakyVerifier(),
    )
    with pytest.raises(OrchestrationError, match="consecutive failed cycles"):
        asyncio.run(cp.run("t1", goal="author a note"))


def test_gate_failure_surfaces_in_the_run_report() -> None:
    cp, _sandbox, _store = _build(
        items=[_note("n1", fail_gate=True)],
        policy=OrchestrationPolicy(max_cycles=1),
        verifier=_FlakyVerifier(),
    )
    asyncio.run(cp.run("t1", goal="author a note"))
    report = cp.run_report("t1")

    assert report.summary.outcomes.get("gate_failed") == 1
    assert report.cycles[0].gate_error is not None
    assert report.to_dict()["cycles"][0]["gate_error"] is not None  # machine-readable, generic


def test_feedback_threads_a_gate_unavailable_reason() -> None:
    from verity.control_plane.api import IntakeResult

    feedback = _feedback_from(IntakeResult(entered_protocol=False, gate_error="backend down"))
    assert feedback.startswith("gate-unavailable:") and "submit your proposal again" in feedback


# ----------------------------------------------------------------- the invariant, property-tested

_PLANS = st.lists(st.sampled_from(["good", "sandbox", "gate"]), min_size=1, max_size=6)


@settings(max_examples=40, deadline=None)
@given(plan=_PLANS)
def test_any_single_cycle_failure_is_recorded_and_never_fatal(plan: list[str]) -> None:
    """For any interleaving of good / sandbox-failed / gate-failed cycles, the run reaches the end
    and every cycle is recorded with the outcome that actually happened (a failed step is recorded,
    not fatal). The failure breaker is set above the run length so no example aborts early."""
    items: list[object] = []
    for i, kind in enumerate(plan):
        if kind == "good":
            items.append(_note(f"n{i}"))
        elif kind == "sandbox":
            items.append(SandboxError(f"sandbox failed at {i}"))
        else:  # gate
            items.append(_note(f"n{i}", fail_gate=True))

    cp, _sandbox, _store = _build(
        items=items,
        policy=OrchestrationPolicy(
            max_cycles=len(plan), max_consecutive_sandbox_failures=len(plan) + 1,
        ),
        verifier=_FlakyVerifier(),
    )
    results = asyncio.run(cp.run("t1", goal="author a note"))

    assert len(results) == len(plan)  # the run never aborted: every planned cycle ran
    for kind, r in zip(plan, results, strict=True):
        if kind == "good":
            assert r.entered_protocol and r.sandbox_error is None and r.gate_error is None
            assert r.commit is not None
        elif kind == "sandbox":
            assert r.sandbox_error is not None and r.gate_error is None
            assert r.entered_protocol is False
        else:  # gate
            assert r.gate_error is not None and r.sandbox_error is None
            assert r.entered_protocol is False

    # The RunReport projects the same per-cycle truth (generic, machine-readable).
    report = cp.run_report("t1")
    labels = [c["gate_error"] is not None for c in report.to_dict()["cycles"]]
    assert labels == [k == "gate" for k in plan]
