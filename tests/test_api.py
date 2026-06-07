"""Control-plane API tests (spec §3.4) — ROADMAP Phase 1.4.

Exercises the async service boundary with minimal in-test doubles for the sandbox and verifier
(the product stubs land in 1.5): configure-by-task, intake (shape-error / harvest / rationale
segregation), verifier dispatch over a rationale-free slice, the cycle loop, and extraction.
"""

from __future__ import annotations

import asyncio

from tests.helpers import accept, reject
from verity.contracts import (
    ProposalEnvelope,
    ProviderRegistry,
    SandboxPort,
    ServedContext,
    VerifierPort,
    VerifierRequest,
)
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.commit import CommitOutcome, GateVerdict
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import DefaultRetrievalPolicy
from verity.control_plane.store import (
    Artifact,
    ArtifactStatus,
    Operation,
    OperationStatus,
    SqliteStore,
)
from verity.domains.fake import NOTE, SOURCE, build_fake_domain


class FakeVerifier:
    """An in-test VerifierPort: scripted per-gate verdicts; records every request it is handed."""

    def __init__(self, verdicts: dict[str, GateVerdict] | None = None) -> None:
        self.verdicts = verdicts or {}
        self.requests: list[VerifierRequest] = []
        self.provisioned = False

    async def dispatch(self, request: VerifierRequest) -> GateVerdict:
        self.requests.append(request)
        return self.verdicts.get(request.gate, accept())

    async def provision(self) -> None:
        self.provisioned = True

    async def teardown(self) -> None:
        self.provisioned = False

    async def health(self) -> bool:
        return True


class FakeSandbox:
    """An in-test SandboxPort: serves context, emits scripted proposals, counts regenerations."""

    def __init__(self, proposals: list[ProposalEnvelope] | None = None) -> None:
        self.queue = list(proposals or [])
        self.served: list[ServedContext] = []
        self.regenerated = 0

    async def serve_context(self, context: ServedContext) -> None:
        self.served.append(context)

    async def collect_proposal(self) -> ProposalEnvelope:
        return self.queue.pop(0)

    async def regenerate(self) -> None:
        self.regenerated += 1

    async def provision(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def health(self) -> bool:
        return True


def _config(domain) -> TaskConfig:
    return TaskConfig(
        task_id="t1",
        instructions="write notes",
        domain_instructions="a Note has a text field",
        schema=domain.schema,
        gates=domain.gates,
        retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator,
        sandbox_key="stub",
        verifier_key="stub",
    )


def _note(note_id: str, *, text: str = "hi", metadata: str = "", objects=None) -> ProposalEnvelope:
    artifact = Artifact(
        id=note_id,
        type=NOTE,
        payload={"text": text},
        status=ArtifactStatus.PROPOSED,
        created_by="agent",
        created_at="",
    )
    op = Operation(
        op_id=f"op-{note_id}",
        op_name="author",
        parents=("src",),
        output_id=note_id,
        status=OperationStatus.SUCCESS,
        created_at="",
    )
    return ProposalEnvelope(
        artifact=artifact, operation=op, metadata=metadata, objects=objects or {}
    )


def _setup(
    *, verdicts=None, proposals=None
) -> tuple[ControlPlane, FakeSandbox, FakeVerifier, SqliteStore]:
    store = SqliteStore()
    # seed a root input the notes descend from
    store.propose(
        Artifact("src", SOURCE, {"raw": 1}, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation("op-src", "load", (), "src", OperationStatus.SUCCESS, "t0"),
    )
    sandbox = FakeSandbox(proposals)
    verifier = FakeVerifier(verdicts)
    sandbox_providers: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    verifier_providers: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sandbox_providers.register("stub", lambda: sandbox)
    verifier_providers.register("stub", lambda: verifier)
    cp = ControlPlane(
        store,
        sandbox_providers=sandbox_providers,
        verifier_providers=verifier_providers,
    )
    asyncio.run(cp.configure(_config(build_fake_domain())))
    return cp, sandbox, verifier, store


def test_configure_stamps_schema_and_provisions_services() -> None:
    cp, sandbox, verifier, store = _setup()
    assert verifier.provisioned is True
    current = store.current_schema_version()
    assert current is not None and NOTE in current.types


def test_happy_path_accepts_and_commits() -> None:
    cp, sandbox, verifier, store = _setup()
    result = asyncio.run(cp.submit_proposal("t1", _note("n1")))
    assert result.entered_protocol is True
    assert result.commit is not None and result.commit.outcome is CommitOutcome.ACCEPTED
    got = store.get_artifact("n1")
    assert got is not None and got.status is ArtifactStatus.ACCEPTED
    # both gate decisions came from the verifier, not the in-process domain runner
    assert len(verifier.requests) == 2


def test_reject_verdict_is_recorded() -> None:
    cp, _sandbox, _verifier, store = _setup(verdicts={"well-formed": reject("nope")})
    result = asyncio.run(cp.submit_proposal("t1", _note("n1")))
    assert result.commit is not None and result.commit.outcome is CommitOutcome.REJECTED
    assert [a.id for a in cp.rejected_log(type=NOTE)] == ["n1"]


def test_malformed_proposal_records_nothing() -> None:
    cp, _sandbox, verifier, store = _setup()
    bad = _note("n1")
    bad = ProposalEnvelope(
        artifact=Artifact("n1", NOTE, {"no_text": 1}, ArtifactStatus.PROPOSED, "agent", ""),
        operation=bad.operation,
        objects={"leftover.py": b"print()"},
    )
    result = asyncio.run(cp.submit_proposal("t1", bad))
    assert result.entered_protocol is False
    assert result.shape_error is not None
    # nothing recorded: no proposed row, no decision, no object harvested (§7.0)
    assert store.get_artifact("n1") is None
    assert result.harvested == {}
    assert verifier.requests == []


def test_object_is_harvested_and_content_addressed() -> None:
    cp, _sandbox, _verifier, store = _setup()
    code = b"def feature(df): return df"
    result = asyncio.run(cp.submit_proposal("t1", _note("n1", objects={"submission.py": code})))
    assert "submission.py" in result.harvested
    ref = result.harvested["submission.py"]
    assert store.get_object(ref.content_hash) == code  # round-trips by reference


def test_verifier_slice_excludes_rationale_and_sees_incumbents() -> None:
    cp, _sandbox, verifier, store = _setup()
    # first submission accepts and becomes the incumbent
    asyncio.run(cp.submit_proposal("t1", _note("n1", metadata="i tried hard")))
    # the rationale is stored on the provenance/context channel...
    assert cp.rationale_for("n1") == "i tried hard"
    # ...but never reached the verifier: no request carried it (the slice is Artifacts only)
    for request in verifier.requests:
        assert not hasattr(request, "rationale")
        for sliced in request.store_slice:
            assert isinstance(sliced, Artifact)

    # a second submission's slice now includes the accepted incumbent n1
    verifier.requests.clear()
    asyncio.run(cp.submit_proposal("t1", _note("n2")))
    saw_incumbent = any(
        any(a.id == "n1" for a in req.store_slice) for req in verifier.requests
    )
    assert saw_incumbent


def test_run_loop_drives_cycles_and_regenerates() -> None:
    proposals = [_note("n1"), _note("n2")]
    policy = OrchestrationPolicy(max_cycles=5, stop_on_accept=True)
    store = SqliteStore()
    store.propose(
        Artifact("src", SOURCE, {"raw": 1}, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation("op-src", "load", (), "src", OperationStatus.SUCCESS, "t0"),
    )
    sandbox = FakeSandbox(proposals)
    verifier = FakeVerifier()
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("stub", lambda: sandbox)
    vp.register("stub", lambda: verifier)
    cp = ControlPlane(store, policy=policy, sandbox_providers=sp, verifier_providers=vp)
    asyncio.run(cp.configure(_config(build_fake_domain())))

    results = asyncio.run(cp.run("t1", goal="make a good note"))
    assert len(results) == 1  # stop_on_accept ended after the first accept
    assert sandbox.regenerated == 1  # workspace regenerated each cycle
    assert [a.id for a in cp.accepted_artifacts(type=NOTE)] == ["n1"]


def test_extraction_answers_why_from_provenance() -> None:
    cp, _sandbox, _verifier, store = _setup()
    asyncio.run(cp.submit_proposal("t1", _note("n1")))
    prov = cp.provenance("n1")
    assert prov.artifact.id == "n1"
    assert "src" in {a.id for a in prov.ancestors}
    assert any(d.verdict.value == "accept" for d in prov.decisions)
