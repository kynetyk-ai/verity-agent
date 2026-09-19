"""The eval / benchmark harness (ROADMAP Phase 6.3).

Pure / offline — no deepagents, no network, no Docker. The cost layer is unit-tested directly; the
harness is driven end-to-end with the same in-test ``SandboxPort`` / ``VerifierPort`` doubles the
control-plane tests use (a scripted proposal + a fixed scored bundle), so two arms yield two rows.
"""

from __future__ import annotations

import asyncio

from tests.helpers import bundle, decision
from verity.contracts import (
    Artifact,
    ArtifactStatus,
    Operation,
    OperationStatus,
    ProposalEnvelope,
    ProviderRegistry,
    SandboxPort,
    ServedContext,
    VerifierPort,
)
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import DefaultRetrievalPolicy
from verity.control_plane.store import SqliteStore
from verity.domains.fake import NOTE, SOURCE, build_fake_domain
from verity.eval import (
    BenchmarkArm,
    ConfiguredTask,
    ModelPrice,
    cost_usd,
    run_benchmark,
)

# ----------------------------------------------------------------------------- cost layer

_TEL = {"input_tokens": 1_000_000, "output_tokens": 2_000_000, "model": "claude-sonnet-4-6"}
_PRICING = {"claude-sonnet-4-6": ModelPrice(3.0, 15.0)}


def test_cost_usd_math() -> None:
    # 1M input @ $3/1M + 2M output @ $15/1M = 3 + 30 = 33.0
    assert cost_usd(_TEL, _PRICING) == 33.0


def test_cost_usd_none_for_absent_telemetry() -> None:
    assert cost_usd(None, _PRICING) is None


def test_cost_usd_none_when_model_unknown_and_no_override() -> None:
    assert cost_usd({"input_tokens": 1, "output_tokens": 1}, _PRICING) is None  # no model key
    assert cost_usd({**_TEL, "model": "mystery"}, _PRICING) is None  # not in the table


def test_cost_usd_none_when_token_counts_missing() -> None:
    assert cost_usd({"model": "claude-sonnet-4-6"}, _PRICING) is None


def test_cost_usd_model_key_override_resolves_the_table() -> None:
    tel = {"input_tokens": 1_000_000, "output_tokens": 0, "model": None}  # null reported model
    assert cost_usd(tel, _PRICING, model_key="claude-sonnet-4-6") == 3.0


def test_cost_usd_explicit_price_wins() -> None:
    # A local arm priced at zero -> 0.0 even with tokens and no table entry.
    assert cost_usd(_TEL, {}, price=ModelPrice(0.0, 0.0)) == 0.0


# ----------------------------------------------------------------------------- the harness

_SCORED = bundle(ArtifactStatus.ACCEPTED, decision("worth-keeping", score=0.9))


class _ScriptedSandbox:
    """A SandboxPort double: serves context, emits one scripted proposal carrying telemetry."""

    def __init__(self, envelope: ProposalEnvelope) -> None:
        self._envelope = envelope

    async def serve_context(self, context: ServedContext) -> None: ...
    async def collect_proposal(self) -> ProposalEnvelope:
        return self._envelope
    async def regenerate(self) -> None: ...
    async def provision(self) -> None: ...
    async def teardown(self) -> None: ...
    async def health(self) -> bool:
        return True


class _FixedVerifier:
    """A VerifierPort double returning a fixed accepted, scored bundle."""

    def __init__(self) -> None:
        self.identity = "fake-verifier"

    async def dispatch(self, request: object) -> object:
        return _SCORED
    async def provision(self) -> None: ...
    async def teardown(self) -> None: ...
    async def health(self) -> bool:
        return True


def _note_envelope(note_id: str, telemetry: dict[str, object]) -> ProposalEnvelope:
    artifact = Artifact(note_id, NOTE, {"text": "hi"}, ArtifactStatus.PROPOSED, "agent", "")
    op = Operation(f"op-{note_id}", "author", ("src",), note_id, OperationStatus.SUCCESS, "")
    return ProposalEnvelope(artifact=artifact, operation=op, agent_telemetry=telemetry)


async def _build_arm(note_id: str, telemetry: dict[str, object]) -> ConfiguredTask:
    domain = build_fake_domain()
    store = SqliteStore()
    store.propose(
        Artifact("src", SOURCE, {"raw": 1}, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation("op-src", "load", (), "src", OperationStatus.SUCCESS, "t0"),
    )
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("stub", lambda: _ScriptedSandbox(_note_envelope(note_id, telemetry)))
    vp.register("stub", lambda: _FixedVerifier())
    cp = ControlPlane(
        store, policy=OrchestrationPolicy(stop_on_accept=True),
        sandbox_providers=sp, verifier_providers=vp,
    )
    config = TaskConfig(
        task_id="t1",
        instructions="write notes",
        domain_instructions="a Note has a text field",
        schema=domain.schema,
        gated_types=domain.gated_types,
        retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator,
        object_namer=lambda _a: frozenset(),
        sandbox_key="stub",
        verifier_key="stub",
    )
    await cp.configure(config)
    return ConfiguredTask(cp, task_id="t1", goal="author a note", model_name="claude-sonnet-4-6")


def test_run_benchmark_one_row_per_arm() -> None:
    tel = {
        "input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000,
        "model_steps": 2, "tool_calls": 1, "model": "claude-sonnet-4-6",
    }
    arms = [
        BenchmarkArm("frontier", lambda: _build_arm("note-a", tel)),
        BenchmarkArm("local", lambda: _build_arm("note-b", tel), price=ModelPrice(0.0, 0.0)),
    ]
    comparison = asyncio.run(run_benchmark(arms, pricing=_PRICING))

    assert [r.label for r in comparison.rows] == ["frontier", "local"]
    for row in comparison.rows:
        assert row.accepted_count >= 1  # the scripted proposal commits ACCEPTED
        assert row.best_score == 0.9  # extracted from the gate decision
        assert row.total_tokens == 1_000_000
    frontier, local = comparison.rows
    assert frontier.cost_usd == 3.0  # 1M input @ $3/1M
    assert local.cost_usd == 0.0  # explicit zero price for the local arm


def test_comparison_to_json_and_render_text() -> None:
    arm = BenchmarkArm(
        "frontier",
        lambda: _build_arm("note-a", {"input_tokens": 10, "output_tokens": 5, "model": "x"}),
    )
    comparison = asyncio.run(run_benchmark([arm], pricing=_PRICING))

    doc = comparison.to_dict()
    assert doc["schema_version"] == 1 and len(doc["rows"]) == 1
    assert "frontier" in comparison.to_json()
    table = comparison.render_text()
    assert "frontier" in table and "accepted" in table
