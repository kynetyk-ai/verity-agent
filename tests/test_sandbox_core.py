"""The framework-neutral sandbox core (ROADMAP Phase 3) — offline, no Deep Agents.

Drives :class:`~verity.sandbox.core.AgentSandbox` with a scripted ``FakeDriver`` (it just leaves a
descriptor + attachments in the outbox), proving the trusted harvest/mint half and a full
``run_cycle`` against the real control plane + real ``SdkVerifier`` on the fake and code domains.
The Deep Agents driver binding is tested separately (``test_sandbox_deepagents.py``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from tools.harness.sample_code import CLEAN

from verity.contracts import (
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
from verity.control_plane.commit import CommitOutcome
from verity.control_plane.config import TaskConfig
from verity.control_plane.registries import (
    ArtifactTypeDef,
    DefaultRetrievalPolicy,
    OperationSignature,
    SchemaRegistry,
)
from verity.control_plane.store import Artifact, SqliteStore
from verity.control_plane.workspace import DefaultLayout, ProvisionedWorkspace
from verity.domains.code import (
    DATASET,
    ENTRYPOINT,
    SUBMISSION,
    build_code_domain,
)
from verity.domains.code import (
    declared_objects as code_declared_objects,
)
from verity.domains.fake import (
    NOTE,
    SOURCE,
    build_fake_domain,
    build_fake_verifier,
)
from verity.domains.fake import (
    declared_objects as fake_declared_objects,
)
from verity.sandbox import AgentSandbox, ProposalDescriptor, SandboxError
from verity.sandbox.descriptor import RESERVED_PROPOSAL_NAME, RESERVED_TELEMETRY_NAME
from verity.verifier import FakeCodeRunner, RunResult


@dataclass
class FakeDriver:
    """A scripted ``SandboxDriver``: writes a descriptor (+ attachments + telemetry) to outbox."""

    descriptor: ProposalDescriptor | None
    objects: dict[str, bytes] = field(default_factory=dict)
    telemetry: bytes | None = None  # raw bytes for __telemetry__.json (may be deliberately corrupt)
    seen_prompt: str = ""

    async def run(
        self,
        *,
        system_prompt: str,
        user_message: str,
        operations: object,
        workspace: ProvisionedWorkspace,
    ) -> None:
        self.seen_prompt = system_prompt
        for name, data in self.objects.items():
            (workspace.outbox() / name).write_bytes(data)
        if self.telemetry is not None:
            (workspace.outbox() / RESERVED_TELEMETRY_NAME).write_bytes(self.telemetry)
        if self.descriptor is not None:
            (workspace.outbox() / RESERVED_PROPOSAL_NAME).write_bytes(self.descriptor.to_json())


def _note_schema() -> SchemaRegistry:
    schema = SchemaRegistry()
    schema.register_type(ArtifactTypeDef(SOURCE, is_root=True))
    schema.register_type(ArtifactTypeDef(NOTE))
    schema.register_operation(OperationSignature("author", inputs=(SOURCE,), output=NOTE))
    return schema


def _sandbox(root: Path, driver: object, schema: SchemaRegistry) -> AgentSandbox:
    return AgentSandbox(
        root=root,
        driver=driver,  # type: ignore[arg-type]
        schema=schema,
        proposer_identity="deepagents:test",
        clock=lambda: "t1",
        id_source=lambda: "art-1",
    )


# ----------------------------------------------------------------- core mint / harvest


def test_mints_typed_envelope_and_splits_descriptor(tmp_path: Path) -> None:
    driver = FakeDriver(
        descriptor=ProposalDescriptor("author", ("src-1",), {"text": "hi"}, metadata="because"),
        objects={"evidence.txt": b"supporting"},
    )
    sandbox = _sandbox(tmp_path / "ws", driver, _note_schema())

    async def go() -> ProposalEnvelope:
        await sandbox.provision()
        await sandbox.serve_context(ServedContext(system_prompt="SYS", tail="Source id=src-1"))
        return await sandbox.collect_proposal()

    env = asyncio.run(go())

    assert env.artifact.id == "art-1" and env.artifact.type == NOTE
    assert env.artifact.payload == {"text": "hi"}
    assert env.artifact.status is ArtifactStatus.PROPOSED
    assert env.artifact.created_by == "deepagents:test" and env.artifact.is_root is False
    assert env.operation.op_name == "author" and env.operation.parents == ("src-1",)
    assert env.operation.output_id == "art-1"
    assert env.metadata == "because"
    # the reserved descriptor is split out; only the genuine attachment is carried as an object
    assert dict(env.objects) == {"evidence.txt": b"supporting"}
    assert driver.seen_prompt == "SYS"


def test_no_proposal_raises(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path / "ws", FakeDriver(descriptor=None), _note_schema())

    async def go() -> None:
        await sandbox.provision()
        await sandbox.serve_context(ServedContext(system_prompt="SYS"))
        await sandbox.collect_proposal()

    with pytest.raises(SandboxError, match="no proposal"):
        asyncio.run(go())


@dataclass
class _TimeoutDriver:
    """Writes an incremental transcript/telemetry to the outbox, then raises as if the worker was
    SIGKILLed by a hard wall-clock timeout (F4) — the writes persist on the bind mount."""

    transcript: bytes
    telemetry: bytes

    async def run(self, *, system_prompt: str, user_message: str, operations: object,
                  workspace: ProvisionedWorkspace) -> None:
        from verity.sandbox.descriptor import RESERVED_TRANSCRIPT_NAME
        (workspace.outbox() / RESERVED_TRANSCRIPT_NAME).write_bytes(self.transcript)
        (workspace.outbox() / RESERVED_TELEMETRY_NAME).write_bytes(self.telemetry)
        raise SandboxError("sandbox container timed out after 1800s")


def test_hard_timeout_salvages_transcript_and_telemetry_from_the_outbox(tmp_path: Path) -> None:
    # F4: a hard timeout SIGKILLs the worker mid-run (no clean exit, no harvest), but the
    # incrementally-written transcript/telemetry survive on the bind-mounted outbox, and
    # collect_proposal salvages them onto the SandboxError so even a timeout stays debuggable.
    tx = b'[{"index": 0, "type": "AIMessage", "content": "training..."}]'
    tel = b'{"model_steps": 7, "model": "x"}'
    driver = _TimeoutDriver(transcript=tx, telemetry=tel)
    sandbox = _sandbox(tmp_path / "ws", driver, _note_schema())

    async def go() -> None:
        await sandbox.provision()
        await sandbox.serve_context(ServedContext(system_prompt="SYS"))
        await sandbox.collect_proposal()

    with pytest.raises(SandboxError) as ei:
        asyncio.run(go())
    assert "timed out" in str(ei.value)
    assert ei.value.transcript == tx
    assert ei.value.telemetry == tel


def test_unknown_operation_raises(tmp_path: Path) -> None:
    driver = FakeDriver(descriptor=ProposalDescriptor("nope", ("src-1",), {"text": "hi"}))
    sandbox = _sandbox(tmp_path / "ws", driver, _note_schema())

    async def go() -> None:
        await sandbox.provision()
        await sandbox.serve_context(ServedContext(system_prompt="SYS"))
        await sandbox.collect_proposal()

    with pytest.raises(SandboxError, match="unknown operation"):
        asyncio.run(go())


def test_collect_before_serve_raises(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path / "ws", FakeDriver(descriptor=None), _note_schema())

    async def go() -> None:
        await sandbox.provision()
        await sandbox.collect_proposal()

    with pytest.raises(SandboxError, match="before serve_context"):
        asyncio.run(go())


def test_regenerate_empties_the_outbox(tmp_path: Path) -> None:
    driver = FakeDriver(descriptor=ProposalDescriptor("author", ("src-1",), {"text": "hi"}))
    sandbox = _sandbox(tmp_path / "ws", driver, _note_schema())

    async def go() -> list[str]:
        await sandbox.provision()
        await sandbox.serve_context(ServedContext(system_prompt="SYS"))
        await sandbox.collect_proposal()
        await sandbox.regenerate()
        return sorted(p.name for p in (tmp_path / "ws" / "outbox").iterdir())

    assert asyncio.run(go()) == []  # the writable outbox is discarded on regeneration


# ------------------------------------------------------------- telemetry + harvest resilience (5.1)


def _collect(sandbox: AgentSandbox) -> ProposalEnvelope:
    async def go() -> ProposalEnvelope:
        await sandbox.provision()
        await sandbox.serve_context(ServedContext(system_prompt="SYS", tail="Source id=src-1"))
        return await sandbox.collect_proposal()

    return asyncio.run(go())


def test_valid_telemetry_is_carried_on_the_envelope(tmp_path: Path) -> None:
    driver = FakeDriver(
        descriptor=ProposalDescriptor("author", ("src-1",), {"text": "hi"}),
        telemetry=b'{"total_tokens": 1234, "model_steps": 3, "model": "claude-x"}',
    )
    env = _collect(_sandbox(tmp_path / "ws", driver, _note_schema()))
    assert env.agent_telemetry == {"total_tokens": 1234, "model_steps": 3, "model": "claude-x"}


def test_corrupt_telemetry_degrades_to_none_without_sinking_the_proposal(tmp_path: Path) -> None:
    # Telemetry is advisory (5.3b): a malformed __telemetry__.json must NOT fail an otherwise-good
    # cycle. The proposal is still minted; only the telemetry is dropped (ROADMAP 5.1).
    driver = FakeDriver(
        descriptor=ProposalDescriptor("author", ("src-1",), {"text": "hi"}),
        telemetry=b"{ this is not valid json ",
    )
    env = _collect(_sandbox(tmp_path / "ws", driver, _note_schema()))
    assert env.artifact.id == "art-1" and env.artifact.payload == {"text": "hi"}
    assert env.agent_telemetry is None


def test_non_object_telemetry_also_degrades_to_none(tmp_path: Path) -> None:
    driver = FakeDriver(
        descriptor=ProposalDescriptor("author", ("src-1",), {"text": "hi"}),
        telemetry=b"[1, 2, 3]",  # valid JSON, but not a telemetry object
    )
    assert _collect(_sandbox(tmp_path / "ws", driver, _note_schema())).agent_telemetry is None


class _HarvestRaisesLayout(DefaultLayout):
    """A layout that provisions normally but fails on harvest, simulating an outbox I/O error."""

    def harvest(self, workspace: ProvisionedWorkspace) -> dict[str, bytes]:
        raise OSError("disk read error in the outbox")


def test_harvest_io_error_becomes_a_sandbox_error(tmp_path: Path) -> None:
    # A raw OSError on the harvest path would bypass the control plane's degrade-don't-crash catch;
    # the core types it as a SandboxError so a transient outbox failure is recoverable (5.1).
    sandbox = AgentSandbox(
        root=tmp_path / "ws",
        driver=FakeDriver(descriptor=ProposalDescriptor("author", ("src-1",), {"text": "hi"})),
        schema=_note_schema(),
        proposer_identity="deepagents:test",
        layout=_HarvestRaisesLayout(),
        clock=lambda: "t1",
        id_source=lambda: "art-1",
    )
    with pytest.raises(SandboxError, match="could not harvest the outbox"):
        _collect(sandbox)


# ----------------------------------------------------------------- full run_cycle (real verifier)


def _seed_root(store: SqliteStore, *, art_id: str, art_type: str, payload: object) -> None:
    store.propose(
        Artifact(art_id, art_type, payload, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation(f"op-{art_id}", "load", (), art_id, OperationStatus.SUCCESS, "t0"),
    )


def _drive(
    tmp_path: Path,
    *,
    domain_schema: SchemaRegistry,
    gated_types: object,
    shape_validator: object,
    object_namer: object,
    verifier: VerifierPort,
    driver: FakeDriver,
    id_source: object,
) -> tuple[ControlPlane, SqliteStore]:
    store = SqliteStore()
    sandbox = AgentSandbox(
        root=tmp_path / "ws",
        driver=driver,  # type: ignore[arg-type]
        schema=domain_schema,
        proposer_identity="deepagents:test",
        clock=lambda: "t1",
        id_source=id_source,  # type: ignore[arg-type]
    )
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("deepagents", lambda: sandbox)
    vp.register("real", lambda: verifier)
    cp = ControlPlane(
        store,
        policy=OrchestrationPolicy(stop_on_accept=True),
        sandbox_providers=sp,
        verifier_providers=vp,
    )
    config = TaskConfig(
        task_id="t1",
        instructions="produce a proposal",
        domain_instructions="follow the domain shape",
        schema=domain_schema,
        gated_types=gated_types,  # type: ignore[arg-type]
        retrieval=DefaultRetrievalPolicy(),
        shape_validator=shape_validator,  # type: ignore[arg-type]
        object_namer=object_namer,  # type: ignore[arg-type]
        sandbox_key="deepagents",
        verifier_key="real",
    )
    asyncio.run(cp.configure(config))
    return cp, store


def test_fake_domain_full_cycle_accepts(tmp_path: Path) -> None:
    domain = build_fake_domain()
    driver = FakeDriver(ProposalDescriptor("author", ("src-1",), {"text": "a kept note"}))
    cp, store = _drive(
        tmp_path,
        domain_schema=domain.schema,
        gated_types=domain.gated_types,
        shape_validator=domain.shape_validator,
        object_namer=fake_declared_objects,
        verifier=build_fake_verifier(),
        driver=driver,
        id_source=lambda: "note-1",
    )
    _seed_root(store, art_id="src-1", art_type=SOURCE, payload={"raw": 1})

    result = asyncio.run(cp.run_cycle("t1", goal="author a note"))

    assert result.commit is not None and result.commit.outcome is CommitOutcome.ACCEPTED
    got = store.get_artifact("note-1")
    assert got is not None and got.status is ArtifactStatus.ACCEPTED and got.type == NOTE


def test_code_domain_full_cycle_accepts(tmp_path: Path) -> None:
    domain = build_code_domain(FakeCodeRunner(script=lambda _r: RunResult(0, "", "")))
    driver = FakeDriver(
        ProposalDescriptor("submit", ("ds",), {"entrypoint": ENTRYPOINT}),
        objects={ENTRYPOINT: CLEAN},
    )
    cp, store = _drive(
        tmp_path,
        domain_schema=domain.schema,
        gated_types=domain.gated_types,
        shape_validator=domain.shape_validator,
        object_namer=code_declared_objects,
        verifier=domain.verifier,
        driver=driver,
        id_source=lambda: "sub-1",
    )
    _seed_root(store, art_id="ds", art_type=DATASET, payload={"n": 10})

    result = asyncio.run(cp.run_cycle("t1", goal="submit a feature"))

    assert result.commit is not None and result.commit.outcome is CommitOutcome.ACCEPTED
    got = store.get_artifact("sub-1")
    assert got is not None and got.status is ArtifactStatus.ACCEPTED and got.type == SUBMISSION
    # the genuine code object travelled the loop as an attachment, split from the descriptor
    assert any(ENTRYPOINT in req.objects for req in domain.verifier.requests)


def test_provision_writes_the_output_schema_to_spec(tmp_path: Path) -> None:
    """The kernel orientation promises the output schema under /work/spec/; provision must put it
    there, rendered accurately from the SchemaRegistry (the §3.4 spec role, no longer empty)."""
    import json

    from verity.sandbox.core import SPEC_SCHEMA_NAME

    sandbox = _sandbox(tmp_path / "ws", object(), _note_schema())  # provision() ignores the driver
    asyncio.run(sandbox.provision())

    matches = list((tmp_path).rglob(SPEC_SCHEMA_NAME))
    assert matches, f"no {SPEC_SCHEMA_NAME} provisioned"
    spec_path = matches[0]
    assert spec_path.parent.name == "spec"  # it lives in the spec/ role
    doc = json.loads(spec_path.read_text(encoding="utf-8"))
    # accurate to the registry: SOURCE(root)+NOTE types, author(Source->Note)
    types = {t["name"]: t["is_root"] for t in doc["artifact_types"]}
    assert types == {"Source": True, "Note": False}
    ops = {o["name"]: o for o in doc["operations"]}
    assert ops["author"]["inputs"] == ["Source"]
    assert ops["author"]["output"] == "Note"
