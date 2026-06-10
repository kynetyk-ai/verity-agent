"""The backend-backed sandbox driver (ROADMAP 7.4.c). Offline — `FakeBackend`, no Docker, no model.

The acceptance proof for 7.4.c: the real `BackendSandboxDriver` produces the same sandbox
`WorkerSpec` the 7.4.a wishful test hand-built, AND the full `AgentSandbox` cycle works through it —
the worker's
outbox bytes are bridged into the host workspace, harvested, and minted unchanged. So the contract
held: the driver consumes the `WorkerSpec` the extraction predicted.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from verity.contracts import ServedContext
from verity.domains.fake import build_fake_domain
from verity.provisioning import CompletedWorker, FakeBackend
from verity.sandbox.backend_driver import BackendSandboxDriver
from verity.sandbox.core import AgentSandbox
from verity.sandbox.descriptor import ProposalDescriptor
from verity.sandbox.errors import SandboxError

_SCHEMA = build_fake_domain().schema  # op "author": Source -> Note


def _proposal() -> bytes:
    return ProposalDescriptor(
        op_name="author", parents=("src",), payload={"text": "hi"}, metadata="why"
    ).to_json()


def _outbox(data: bytes) -> CompletedWorker:
    return CompletedWorker(
        exit_code=0, stdout="", stderr="", outputs={"/work/outbox/__proposal__.json": data}
    )


def test_driver_builds_the_sandbox_spec_and_bridges_the_outbox(tmp_path: Path) -> None:
    backend = FakeBackend(script=lambda _s: _outbox(_proposal()))
    sandbox = AgentSandbox(
        root=tmp_path / "ws",
        driver=BackendSandboxDriver(backend=backend, model="anthropic:x", config="fe"),
        schema=_SCHEMA, proposer_identity="deepagents-worker:fake",
    )
    asyncio.run(sandbox.provision())
    asyncio.run(sandbox.serve_context(ServedContext(system_prompt="SYS", tail="author a note")))
    envelope = asyncio.run(sandbox.collect_proposal())

    # the worker's outbox bytes were bridged to the host workspace and minted unchanged
    assert envelope.artifact.type == "Note"
    assert envelope.operation.op_name == "author"
    assert envelope.artifact.created_by == "deepagents-worker:fake"

    # the spec the driver built IS the wishful sandbox posture, now produced for real
    spec = backend.launched[0]
    assert spec.command == ("python", "-m", "verity.sandbox.container_entry")
    assert spec.labels.role == "sandbox" and spec.labels.config == "fe"
    assert "/sandbox/input.json" in spec.readonly_inputs  # the cycle input, physically read-only
    assert spec.writable_dirs == ("/work",)
    assert spec.output_globs == ("/work/outbox/*",)
    assert spec.network is True  # the agent must reach the model API
    assert spec.env["VERITY_SANDBOX_MODEL"] == "anthropic:x"


def test_readonly_roles_seed_physically_read_only_inputs(tmp_path: Path) -> None:
    # static_contents populates the gold `data` role; it must ride as a ro worker input (§3.5) and
    # never a writable one — the worker can read train.csv but cannot mutate it.
    backend = FakeBackend(script=lambda _s: _outbox(_proposal()))
    sandbox = AgentSandbox(
        root=tmp_path / "ws", driver=BackendSandboxDriver(backend=backend),
        schema=_SCHEMA, proposer_identity="p",
        static_contents={"data": {"train.csv": b"a,b\n1,2\n"}},
    )
    asyncio.run(sandbox.provision())
    asyncio.run(sandbox.serve_context(ServedContext(system_prompt="SYS", tail="go")))
    asyncio.run(sandbox.collect_proposal())

    spec = backend.launched[0]
    assert spec.readonly_inputs["/work/data/train.csv"] == b"a,b\n1,2\n"
    assert "/work/data/train.csv" not in spec.writable_dirs


def test_nonzero_worker_exit_degrades_to_a_sandbox_error(tmp_path: Path) -> None:
    backend = FakeBackend(
        script=lambda _s: CompletedWorker(exit_code=1, stdout="", stderr="boom in the worker")
    )
    sandbox = AgentSandbox(
        root=tmp_path / "ws", driver=BackendSandboxDriver(backend=backend),
        schema=_SCHEMA, proposer_identity="p",
    )
    asyncio.run(sandbox.provision())
    asyncio.run(sandbox.serve_context(ServedContext(system_prompt="SYS", tail="go")))
    with pytest.raises(SandboxError, match="exited 1"):
        asyncio.run(sandbox.collect_proposal())


def test_worker_timeout_degrades_to_a_sandbox_error(tmp_path: Path) -> None:
    backend = FakeBackend(
        script=lambda _s: CompletedWorker(exit_code=-1, stdout="", stderr="", timed_out=True)
    )
    sandbox = AgentSandbox(
        root=tmp_path / "ws", driver=BackendSandboxDriver(backend=backend, timeout_s=5.0),
        schema=_SCHEMA, proposer_identity="p",
    )
    asyncio.run(sandbox.provision())
    asyncio.run(sandbox.serve_context(ServedContext(system_prompt="SYS", tail="go")))
    with pytest.raises(SandboxError, match="timed out"):
        asyncio.run(sandbox.collect_proposal())
