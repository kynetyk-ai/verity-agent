"""Control-plane API tests (spec §3.4; ADR 0001) — ROADMAP Phase 1.4/refactor.

Exercises the async service boundary with in-test doubles: configure-by-task, intake (shape-error /
harvest / rationale segregation / object sidecar), the single opaque-verifier handoff over a
rationale-free declared slice, the cycle loop, and extraction.
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
    VerdictBundle,
    VerdictKind,
    VerifierPort,
    VerifierRequest,
)
from verity.control_plane.api import ControlPlane, OrchestrationPolicy
from verity.control_plane.commit import CommitOutcome
from verity.control_plane.config import TaskConfig, orientation_digest
from verity.control_plane.registries import DefaultRetrievalPolicy
from verity.control_plane.store import SqliteStore
from verity.control_plane.workspace import WORKSPACE_CONTRACT
from verity.domains.fake import NOTE, SOURCE, build_fake_domain, declared_objects

_ACCEPT = bundle(ArtifactStatus.ACCEPTED, decision("well-formed"), decision("worth-keeping"))


class FakeVerifier:
    """An in-test VerifierPort: returns a fixed bundle; records every request it is handed."""

    def __init__(self, result: VerdictBundle | None = None) -> None:
        self.identity = "fake-verifier"
        self.result = result if result is not None else _ACCEPT
        self.requests: list[VerifierRequest] = []
        self.provisioned = False

    async def dispatch(self, request: VerifierRequest) -> VerdictBundle:
        self.requests.append(request)
        return self.result

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


def _config(domain, *, retrieval=None) -> TaskConfig:
    return TaskConfig(
        task_id="t1",
        instructions="write notes",
        domain_instructions="a Note has a text field",
        schema=domain.schema,
        gated_types=domain.gated_types,
        retrieval=retrieval if retrieval is not None else DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator,
        object_namer=declared_objects,
        sandbox_key="stub",
        verifier_key="stub",
    )


def _note(note_id: str, *, text: str = "hi", metadata: str = "", objects=None) -> ProposalEnvelope:
    # Declare the attached object names in the payload so the harvest keeps them (a well-formed
    # proposal declares its objects); the fake domain's namer reads this 'objects' list.
    payload: dict = {"text": text}
    if objects:
        payload["objects"] = sorted(objects)
    artifact = Artifact(note_id, NOTE, payload, ArtifactStatus.PROPOSED, "agent", "")
    op = Operation(f"op-{note_id}", "author", ("src",), note_id, OperationStatus.SUCCESS, "")
    return ProposalEnvelope(
        artifact=artifact, operation=op, metadata=metadata, objects=objects or {}
    )


def _setup(
    *, result=None, proposals=None, policy=None
) -> tuple[ControlPlane, FakeSandbox, FakeVerifier, SqliteStore]:
    store = SqliteStore()
    store.propose(
        Artifact("src", SOURCE, {"raw": 1}, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation("op-src", "load", (), "src", OperationStatus.SUCCESS, "t0"),
    )
    sandbox = FakeSandbox(proposals)
    verifier = FakeVerifier(result)
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("stub", lambda: sandbox)
    vp.register("stub", lambda: verifier)
    cp = ControlPlane(store, policy=policy, sandbox_providers=sp, verifier_providers=vp)
    asyncio.run(cp.configure(_config(build_fake_domain())))
    return cp, sandbox, verifier, store


def test_configure_stamps_schema_and_provisions_services() -> None:
    _cp, _sandbox, verifier, store = _setup()
    assert verifier.provisioned is True
    current = store.current_schema_version()
    assert current is not None and NOTE in current.types


def test_configure_stamps_the_workspace_contract_version() -> None:
    # #12: the workspace-contract / orientation version is recorded like the schema version (§3.4).
    _cp, _sandbox, _verifier, store = _setup()
    cv = store.current_contract_version()
    assert cv is not None
    assert cv.version == WORKSPACE_CONTRACT.version
    assert cv.orientation_digest == orientation_digest(WORKSPACE_CONTRACT)


def test_happy_path_accepts_and_commits() -> None:
    cp, _sandbox, verifier, store = _setup()
    result = asyncio.run(cp.submit_proposal("t1", _note("n1")))
    assert result.entered_protocol is True
    assert result.commit is not None and result.commit.outcome is CommitOutcome.ACCEPTED
    got = store.get_artifact("n1")
    assert got is not None and got.status is ArtifactStatus.ACCEPTED
    assert len(verifier.requests) == 1  # one opaque handoff, not per-gate dispatch (ADR 0001)


def test_reject_bundle_is_recorded() -> None:
    rejected = bundle(ArtifactStatus.REJECTED, decision("well-formed", VerdictKind.REJECT, "nope"))
    cp, _sandbox, _verifier, _store = _setup(result=rejected)
    result = asyncio.run(cp.submit_proposal("t1", _note("n1")))
    assert result.commit is not None and result.commit.outcome is CommitOutcome.REJECTED
    assert [a.id for a in cp.rejected_log(type=NOTE)] == ["n1"]


def test_malformed_proposal_records_nothing() -> None:
    cp, _sandbox, verifier, store = _setup()
    bad = ProposalEnvelope(
        artifact=Artifact("n1", NOTE, {"no_text": 1}, ArtifactStatus.PROPOSED, "agent", ""),
        operation=Operation("op-n1", "author", ("src",), "n1", OperationStatus.SUCCESS, ""),
        objects={"leftover.py": b"print()"},
    )
    result = asyncio.run(cp.submit_proposal("t1", bad))
    assert result.entered_protocol is False
    assert result.shape_error is not None
    # nothing recorded: no proposed row, no decision, no object harvested, no dispatch (§7.0)
    assert store.get_artifact("n1") is None
    assert result.harvested == {}
    assert verifier.requests == []


def test_a_non_object_payload_is_refused_at_the_kernel(  # #13: kernel JSON-object guarantee
) -> None:
    cp, _sandbox, verifier, store = _setup()
    # a list payload would slip past the domain validator (it only checks NOTE dicts); the kernel
    # guard refuses any non-object payload before the domain validator runs and records nothing.
    bad_payload = ["not", "an", "object"]
    bad = ProposalEnvelope(
        artifact=Artifact("n1", NOTE, bad_payload, ArtifactStatus.PROPOSED, "agent", ""),
        operation=Operation("op-n1", "author", ("src",), "n1", OperationStatus.SUCCESS, ""),
    )
    result = asyncio.run(cp.submit_proposal("t1", bad))
    assert result.entered_protocol is False
    assert result.shape_error is not None and "JSON object" in result.shape_error.message
    assert store.get_artifact("n1") is None and verifier.requests == []


def test_an_operation_can_require_payload_keys_at_intake() -> None:  # #13 per-op payload schema
    from verity.control_plane.registries import (
        ArtifactTypeDef,
        GatedTypeRegistry,
        OperationSignature,
        SchemaRegistry,
        StoreInput,
    )

    schema = SchemaRegistry()
    schema.register_type(ArtifactTypeDef(SOURCE, is_root=True))
    schema.register_type(ArtifactTypeDef(NOTE))
    schema.register_operation(
        OperationSignature("author", (SOURCE,), NOTE, required_payload_keys=("text", "priority"))
    )
    gated = GatedTypeRegistry()
    gated.gate(NOTE, declared_inputs=frozenset({StoreInput.INCUMBENTS}))

    store = SqliteStore()
    store.propose(
        Artifact("src", SOURCE, {"raw": 1}, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation("op-src", "load", (), "src", OperationStatus.SUCCESS, "t0"),
    )
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("stub", lambda: FakeSandbox(None))
    vp.register("stub", lambda: FakeVerifier(bundle(ArtifactStatus.TENTATIVE, decision("g"))))
    cp = ControlPlane(store, sandbox_providers=sp, verifier_providers=vp)
    asyncio.run(cp.configure(TaskConfig(
        task_id="t1", instructions="x", domain_instructions="x", schema=schema, gated_types=gated,
        retrieval=DefaultRetrievalPolicy(), shape_validator=lambda _a: None,
        object_namer=lambda _a: frozenset(), sandbox_key="stub", verifier_key="stub",
    )))

    # payload missing the declared 'priority' key -> refused at intake, records nothing
    missing = ProposalEnvelope(
        artifact=Artifact("n1", NOTE, {"text": "hi"}, ArtifactStatus.PROPOSED, "agent", ""),
        operation=Operation("op-n1", "author", ("src",), "n1", OperationStatus.SUCCESS, ""),
    )
    refused = asyncio.run(cp.submit_proposal("t1", missing))
    assert refused.entered_protocol is False
    assert refused.shape_error is not None and "priority" in refused.shape_error.message
    assert store.get_artifact("n1") is None

    # the same payload WITH the declared keys clears the per-op check and proceeds
    complete = ProposalEnvelope(
        artifact=Artifact(
            "n2", NOTE, {"text": "hi", "priority": 1}, ArtifactStatus.PROPOSED, "agent", ""
        ),
        operation=Operation("op-n2", "author", ("src",), "n2", OperationStatus.SUCCESS, ""),
    )
    accepted = asyncio.run(cp.submit_proposal("t1", complete))
    assert accepted.entered_protocol is True and accepted.shape_error is None


def test_object_is_harvested_and_content_addressed() -> None:
    cp, _sandbox, _verifier, store = _setup()
    code = b"def feature(df): return df"
    result = asyncio.run(cp.submit_proposal("t1", _note("n1", objects={"submission.py": code})))
    assert "submission.py" in result.harvested
    ref = result.harvested["submission.py"]
    assert store.get_object(ref.content_hash) == code  # round-trips by reference


def test_harvested_objects_are_linked_as_an_artifact_sidecar() -> None:
    # the object↔artifact link is recorded on the artifact's sidecar (§4.1, ADR 0001), durable and
    # auditable — the domain payload is stored verbatim, the control plane never touches it.
    cp, _sandbox, _verifier, store = _setup()
    code = b"def feature(df):\n    return df['a']\n"
    asyncio.run(cp.submit_proposal("t1", _note("n1", objects={"submission.py": code})))

    art = store.get_artifact("n1")
    assert art is not None
    objects = dict(art.objects)
    assert store.get_object(objects["submission.py"].content_hash) == code
    assert art.payload == {"text": "hi", "objects": ["submission.py"]}  # payload untouched


def test_undeclared_outbox_objects_are_dropped_and_warned() -> None:
    # Context discipline (§3.4): the harvest keeps ONLY the objects the proposal declares. An extra
    # file the agent left in the outbox is dropped — never content-addressed, never on the sidecar,
    # never provisioned next cycle — and the drop is logged. (Declared via the payload 'objects'.)
    import structlog

    cp, _sandbox, _verifier, store = _setup()
    env = ProposalEnvelope(
        artifact=Artifact(
            "n1", NOTE, {"text": "hi", "objects": ["keep.py"]}, ArtifactStatus.PROPOSED, "agent", ""
        ),
        operation=Operation("op-n1", "author", ("src",), "n1", OperationStatus.SUCCESS, ""),
        objects={"keep.py": b"keep", "drop.py": b"junk"},  # drop.py is undeclared
    )
    with structlog.testing.capture_logs() as logs:
        result = asyncio.run(cp.submit_proposal("t1", env))

    assert "keep.py" in result.harvested and "drop.py" not in result.harvested
    art = store.get_artifact("n1")
    assert art is not None and dict(art.objects).keys() == {"keep.py"}  # sidecar excludes it too
    dropped = [e for e in logs if e["event"] == "undeclared_outbox_objects_dropped"]
    assert dropped and dropped[0]["dropped"] == ["drop.py"]


def test_objectless_proposal_has_an_empty_sidecar_and_untouched_payload() -> None:
    cp, _sandbox, _verifier, store = _setup()
    asyncio.run(cp.submit_proposal("t1", _note("n1")))
    art = store.get_artifact("n1")
    assert art is not None and art.payload == {"text": "hi"} and art.objects == ()


def test_verifier_request_excludes_rationale_and_sees_the_declared_slice() -> None:
    cp, _sandbox, verifier, _store = _setup()
    asyncio.run(cp.submit_proposal("t1", _note("n1", metadata="i tried hard")))
    # the rationale is stored on the provenance/context channel...
    assert cp.rationale_for("n1") == "i tried hard"
    # ...but never reaches the verifier: the request carries no rationale, the slice is Artifacts
    for request in verifier.requests:
        assert not hasattr(request, "rationale")
        assert all(isinstance(a, Artifact) for a in request.store_slice)

    # a second submission's declared slice (Note → incumbents) now includes the accepted n1
    verifier.requests.clear()
    asyncio.run(cp.submit_proposal("t1", _note("n2")))
    assert any(a.id == "n1" for a in verifier.requests[0].store_slice)


def test_task_retrieval_policy_feeds_context_assembly() -> None:
    # a domain-supplied retrieval policy (§8.4) must drive the served tail, not the kernel default
    consulted: list[str] = []

    class OnlySources:
        def select(self, goal: str, store, *, limit: int) -> list[Artifact]:
            consulted.append(goal)
            return [a for a in store.query_artifacts() if a.type == SOURCE]

    store = SqliteStore()
    store.propose(
        Artifact("src", SOURCE, {"raw": 1}, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation("op-src", "load", (), "src", OperationStatus.SUCCESS, "t0"),
    )
    store.propose(  # a Note the default policy would surface but this one excludes from the tail
        Artifact("n1", NOTE, {"text": "hi"}, ArtifactStatus.PROPOSED, "agent", "t1"),
        Operation("op-n1", "author", ("src",), "n1", OperationStatus.SUCCESS, "t1"),
    )
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("stub", lambda: FakeSandbox())
    vp.register("stub", lambda: FakeVerifier())
    cp = ControlPlane(store, sandbox_providers=sp, verifier_providers=vp)
    asyncio.run(cp.configure(_config(build_fake_domain(), retrieval=OnlySources())))

    served = asyncio.run(cp.serve_context("t1", goal="find roots"))
    assert consulted == ["find roots"]  # the task's policy was the one consulted
    assert "Source/src" in served.tail  # its selection drives the tail
    assert "Note/n1" not in served.tail  # the Note it excluded is absent from the tail


def test_run_loop_drives_cycles_and_regenerates() -> None:
    cp, sandbox, _verifier, store = _setup(
        proposals=[_note("n1"), _note("n2")],
        policy=OrchestrationPolicy(max_cycles=5, stop_on_accept=True),
    )
    results = asyncio.run(cp.run("t1", goal="make a good note"))
    assert len(results) == 1  # stop_on_accept ended after the first accept
    assert sandbox.regenerated == 1  # workspace regenerated each cycle
    assert [a.id for a in cp.accepted_artifacts(type=NOTE)] == ["n1"]


def test_extraction_answers_why_from_provenance() -> None:
    cp, _sandbox, _verifier, _store = _setup()
    asyncio.run(cp.submit_proposal("t1", _note("n1")))
    prov = cp.provenance("n1")
    assert prov.artifact.id == "n1"
    assert "src" in {a.id for a in prov.ancestors}
    assert any(d.verdict.value == "accept" for d in prov.decisions)
