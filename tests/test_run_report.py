"""The store-derived `RunReport` (ROADMAP 5.3a) — generic, JSON-first, verifier-agnostic.

Drives the real control plane with scripted doubles and asserts the report faithfully projects the
kernel's vocabulary (lifecycle outcomes, decisions verbatim, provenance) — assuming **nothing** of
a verifier. The headline guarantee: a verifier that emits **no numeric score** still yields a
complete report with `score: null` throughout (no score is ever elevated).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

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
    VerdictBundle,
    VerifierPort,
    VerifierRequest,
)
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import DefaultRetrievalPolicy
from verity.control_plane.run_report import REPORT_SCHEMA_VERSION
from verity.control_plane.store import SqliteStore
from verity.domains.fake import NOTE, SOURCE, build_fake_domain, build_fake_verifier
from verity.sandbox.errors import SandboxError


def _timer_seq() -> object:
    seq = iter(i * 0.001 for i in range(100_000))
    return lambda: next(seq)  # deterministic monotonic seconds, so timings are reproducible


@dataclass
class _ScriptedSandbox:
    items: list[object]
    cursor: int = 0

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
        return None

    async def provision(self) -> None:
        return None

    async def teardown(self) -> None:
        return None

    async def health(self) -> bool:
        return True


@dataclass
class _ScriptedVerifier:
    """A `VerifierPort` that replays scripted bundles — for the supersession case."""

    identity: str
    bundles: list[VerdictBundle]
    cursor: int = 0
    requests: list[VerifierRequest] = field(default_factory=list)

    async def dispatch(self, request: VerifierRequest) -> VerdictBundle:
        self.requests.append(request)
        bundle = self.bundles[min(self.cursor, len(self.bundles) - 1)]
        self.cursor += 1
        return bundle

    async def provision(self) -> None:
        return None

    async def teardown(self) -> None:
        return None

    async def health(self) -> bool:
        return True


def _note(note_id: str, payload: dict[str, object], *, parents: tuple[str, ...] = ("src",),
          op_name: str = "author") -> ProposalEnvelope:
    artifact = Artifact(note_id, NOTE, payload, ArtifactStatus.PROPOSED, "agent", "t1")
    op = Operation(f"op-{note_id}", op_name, parents, note_id, OperationStatus.SUCCESS, "t1")
    return ProposalEnvelope(artifact=artifact, operation=op, metadata=f"reason-{note_id}")


def _build(
    tmp_path: Path, *, items: list[object], verifier: VerifierPort,
    policy: OrchestrationPolicy | None = None,
) -> tuple[ControlPlane, SqliteStore]:
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
    vp.register("scripted", lambda: verifier)
    cp = ControlPlane(
        store, policy=policy or OrchestrationPolicy(max_cycles=10),
        sandbox_providers=sp, verifier_providers=vp, timer=_timer_seq(),  # type: ignore[arg-type]
    )
    config = TaskConfig(
        task_id="t1", instructions="make notes", domain_instructions="a Note has text",
        schema=domain.schema, gated_types=domain.gated_types, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator, object_namer=lambda _a: frozenset(),
        sandbox_key="scripted", verifier_key="scripted",
    )
    asyncio.run(cp.configure(config))
    return cp, store


# --------------------------------------------------------------------------- the mixed run


def test_report_projects_every_cycle_faithfully(tmp_path: Path) -> None:
    # accept -> revise (defect) -> reject -> sandbox-fail (skipped), via the real fake verifier.
    items = [
        _note("n1", {"text": "a"}),
        _note("n2", {"text": "b", "defect": "tone"}),
        _note("n3", {"text": "c", "reject": True}),
        SandboxError("sandbox container timed out after 1500.0s"),
    ]
    cp, _store = _build(
        tmp_path, items=items, verifier=build_fake_verifier(),
        policy=OrchestrationPolicy(max_cycles=4),  # exactly the four scripted cycles
    )
    asyncio.run(cp.run("t1", goal="author notes"))

    report = cp.run_report("t1")
    assert report.report_schema_version == REPORT_SCHEMA_VERSION
    assert report.summary.cycles_run == 4
    assert report.summary.outcomes == {
        "accepted": 1, "revised": 1, "rejected": 1, "sandbox_failed": 1
    }
    assert report.summary.trial_count == 1 and report.summary.refine_count == 1

    c0, c1, c2, c3 = report.cycles
    assert c0.commit is not None and c0.commit.outcome == "accepted"
    assert c0.proposal is not None and c0.proposal.artifact_id == "n1" and c0.proposal.type == NOTE
    assert c0.rationale == "reason-n1"  # the segregated proposer note, operator-facing
    assert c1.commit is not None and c1.commit.outcome == "revised"
    assert c1.commit.defects == ["tone"]
    assert c2.commit is not None and c2.commit.outcome == "rejected"
    # the skipped cycle is faithful: a reason, no proposal, no commit
    assert c3.sandbox_error is not None and c3.proposal is None and c3.commit is None
    # timings were measured (structure, not exact values)
    assert c0.timings.cycle_ms is not None and c0.timings.cycle_ms >= 0


def test_decisions_are_projected_with_null_score(tmp_path: Path) -> None:
    # The fake verifier's deterministic checks emit NO score -> the report must show score: null
    # everywhere and still be complete. This is the generalizability guarantee.
    cp, _store = _build(
        tmp_path, items=[_note("n1", {"text": "a"})], verifier=build_fake_verifier(),
        policy=OrchestrationPolicy(stop_on_accept=True),
    )
    asyncio.run(cp.run("t1", goal="author a note"))

    commit = cp.run_report("t1").cycles[0].commit
    assert commit is not None and commit.decisions  # there are recorded decisions
    assert all(d.score is None for d in commit.decisions)  # never assumes a numeric score
    assert all(d.gate and d.verdict and d.rationale for d in commit.decisions)  # still complete


def test_report_round_trips_to_json(tmp_path: Path) -> None:
    cp, _store = _build(
        tmp_path, items=[_note("n1", {"text": "a"})], verifier=build_fake_verifier(),
        policy=OrchestrationPolicy(stop_on_accept=True),
    )
    asyncio.run(cp.run("t1", goal="g"))
    report = cp.run_report("t1")
    parsed = json.loads(report.to_json())
    assert parsed == report.to_dict()  # machine-readable + stable
    assert parsed["report_schema_version"] == REPORT_SCHEMA_VERSION
    assert parsed["agent_telemetry"] is None  # 5.3b slot present but unpopulated


# --------------------------------------------------------------------------- supersession


def test_report_records_supersession(tmp_path: Path) -> None:
    verifier = _ScriptedVerifier(
        identity="scripted-verifier",
        bundles=[VerdictBundle(ArtifactStatus.ACCEPTED),
                 VerdictBundle(ArtifactStatus.ACCEPTED, supersedes="n1")],
    )
    cp, store = _build(
        tmp_path, items=[_note("n1", {"text": "a"}), _note("n2", {"text": "b"})], verifier=verifier,
    )
    asyncio.run(cp.run_cycle("t1", goal="c1"))
    asyncio.run(cp.run_cycle("t1", goal="c2"))

    report = cp.run_report("t1")
    assert report.summary.superseded == [{"id": "n1", "superseded_by": "n2"}]
    assert report.cycles[1].commit is not None and report.cycles[1].commit.supersedes == "n1"


def test_render_text_is_non_empty(tmp_path: Path) -> None:
    cp, _store = _build(
        tmp_path, items=[_note("n1", {"text": "a"})], verifier=build_fake_verifier(),
        policy=OrchestrationPolicy(stop_on_accept=True),
    )
    asyncio.run(cp.run("t1", goal="g"))
    text = cp.run_report("t1").render_text()
    assert "RunReport" in text and "n1" in text


def test_stub_double_satisfies_port() -> None:
    assert isinstance(_ScriptedSandbox(items=[]), SandboxPort)
    assert isinstance(StubVerifier(), VerifierPort)
