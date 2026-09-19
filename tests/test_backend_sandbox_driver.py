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


def test_provisioned_objects_reach_the_worker(tmp_path: Path) -> None:
    # The control plane materializes the prior accepted submission into the writable `scratch` role
    # (ObjectProvisioningPolicy → `provided/`). The agent runs INSIDE the worker, so these must be
    # carried in — else it never sees its incumbent and restarts from zero every cycle. (Regression:
    # the driver used to carry only the read-only roles, silently dropping `scratch/provided/`.)
    backend = FakeBackend(script=lambda _s: _outbox(_proposal()))
    sandbox = AgentSandbox(
        root=tmp_path / "ws", driver=BackendSandboxDriver(backend=backend),
        schema=_SCHEMA, proposer_identity="p",
    )
    asyncio.run(sandbox.provision())
    asyncio.run(sandbox.serve_context(ServedContext(
        system_prompt="SYS", tail="build on the incumbent",
        workspace_objects={"scratch": {
            "provided/submission.py": b"# prior incumbent\n",
            "provided/INDEX.md": b"rank 01: accepted\n",
        }},
    )))
    asyncio.run(sandbox.collect_proposal())

    spec = backend.launched[0]
    assert spec.readonly_inputs["/work/scratch/provided/submission.py"] == b"# prior incumbent\n"
    assert "/work/scratch/provided/INDEX.md" in spec.readonly_inputs


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


def test_transcript_and_telemetry_reconstructed_from_worker_stderr(tmp_path: Path) -> None:
    # The worker writes NO transcript/telemetry FILE (the agent shares its uid and could tamper) —
    # only the proposal descriptor. The host reconstructs both from the worker's captured stderr log
    # stream and bridges them into the outbox, so the minted envelope carries them as usual.
    import json

    from verity.sandbox.log_transcript import TELEMETRY_EVENT, TURN_EVENT

    def _line(event: str, **f: object) -> str:
        return json.dumps({"event": event, "level": "info", "timestamp": "t", **f})

    stderr = "\n".join(
        [
            _line(TURN_EVENT, index=0, type="HumanMessage", content="author a note"),
            _line(TURN_EVENT, index=1, type="AIMessage", content="done"),
            _line(TELEMETRY_EVENT, input_tokens=5, model_steps=1, stop_reason="clean"),
        ]
    )
    backend = FakeBackend(
        script=lambda _s: CompletedWorker(
            exit_code=0, stdout="", stderr=stderr,
            outputs={"/work/outbox/__proposal__.json": _proposal()},  # NO diagnostics files
        )
    )
    sandbox = AgentSandbox(
        root=tmp_path / "ws", driver=BackendSandboxDriver(backend=backend),
        schema=_SCHEMA, proposer_identity="p",
    )
    asyncio.run(sandbox.provision())
    asyncio.run(sandbox.serve_context(ServedContext(system_prompt="SYS", tail="go")))
    envelope = asyncio.run(sandbox.collect_proposal())

    assert envelope.agent_telemetry is not None
    assert envelope.agent_telemetry["model_steps"] == 1
    assert envelope.transcript is not None
    transcript = json.loads(envelope.transcript)
    assert [e["content"] for e in transcript] == ["author a note", "done"]


# --------------------------------------------- failed-cycle telemetry is always reported (#125)


def _turn_line(index: int, **fields: object) -> str:
    import json

    return json.dumps({
        "event": "agent_turn", "level": "info", "timestamp": "t",
        "index": index, **fields,
    })


_KILLED_STDERR = "\n".join([
    _turn_line(0, type="HumanMessage", content="go"),
    _turn_line(1, type="AIMessage", content="", tool_calls=[{"name": "execute", "args": {}}],
               usage={"input_tokens": 500, "output_tokens": 80}, model="m1"),
    _turn_line(2, type="ToolMessage", content="running...", tool_name="execute"),
    # no agent_telemetry line: the worker was killed before its end-of-run event
])


def _collect_failure(backend: FakeBackend, root: Path, *, timeout_s: float = 5.0) -> SandboxError:
    sandbox = AgentSandbox(
        root=root, driver=BackendSandboxDriver(backend=backend, timeout_s=timeout_s),
        schema=_SCHEMA, proposer_identity="p",
    )
    asyncio.run(sandbox.provision())
    asyncio.run(sandbox.serve_context(ServedContext(system_prompt="SYS", tail="go")))
    with pytest.raises(SandboxError) as excinfo:
        asyncio.run(sandbox.collect_proposal())
    return excinfo.value


def test_timeout_synthesizes_telemetry_from_streamed_turns(tmp_path: Path) -> None:
    # The #125 case: a SIGKILLed worker never emits its end-of-run telemetry event. The host
    # reconstructs the aggregate from the streamed turns (real sums) and stamps WHY it stopped.
    import json

    backend = FakeBackend(script=lambda _s: CompletedWorker(
        exit_code=-1, stdout="", stderr=_KILLED_STDERR, timed_out=True,
    ))
    error = _collect_failure(backend, tmp_path / "ws", timeout_s=5.0)

    assert error.transcript is not None  # the streamed turns survived
    assert error.telemetry is not None
    telemetry = json.loads(error.telemetry)
    assert telemetry["stop_reason"] == "sandbox_timeout" and telemetry["timeout_s"] == 5.0
    assert telemetry["source"] == "reconstructed_from_turns"
    assert telemetry["total_tokens"] == 580 and telemetry["model_steps"] == 1


def test_timeout_before_any_turn_still_says_why(tmp_path: Path) -> None:
    import json

    backend = FakeBackend(script=lambda _s: CompletedWorker(
        exit_code=-1, stdout="", stderr="", timed_out=True,
    ))
    error = _collect_failure(backend, tmp_path / "ws")
    telemetry = json.loads(error.telemetry or b"{}")
    assert telemetry["stop_reason"] == "sandbox_timeout" and telemetry["source"] == "host"


def test_abrupt_crash_synthesizes_worker_crash_telemetry(tmp_path: Path) -> None:
    # An OOM-kill / entrypoint death: non-zero exit, no telemetry line in stderr.
    import json

    backend = FakeBackend(script=lambda _s: CompletedWorker(
        exit_code=137, stdout="", stderr=_KILLED_STDERR,
    ))
    error = _collect_failure(backend, tmp_path / "ws")
    telemetry = json.loads(error.telemetry or b"{}")
    assert telemetry["stop_reason"] == "worker_crash" and telemetry["exit_code"] == 137
    assert telemetry["total_tokens"] == 580  # recovered from the streamed turns


def test_worker_emitted_crash_telemetry_is_never_overwritten(tmp_path: Path) -> None:
    # A crash that reached run_agent's crash path logged REAL telemetry before dying — the host
    # must keep it (no host stop_reason overlay).
    import json

    worker_line = json.dumps({
        "event": "agent_telemetry", "level": "info", "timestamp": "t",
        "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
        "model_steps": 1, "tool_calls": 0, "model": "m1",
        "stop_reason": "crash: ValueError: boom",
    })
    backend = FakeBackend(script=lambda _s: CompletedWorker(
        exit_code=1, stdout="", stderr=_KILLED_STDERR + "\n" + worker_line,
    ))
    error = _collect_failure(backend, tmp_path / "ws")
    telemetry = json.loads(error.telemetry or b"{}")
    assert telemetry["stop_reason"] == "crash: ValueError: boom"
    assert "source" not in telemetry  # worker-emitted, not synthesized


def test_launch_failure_synthesizes_telemetry(tmp_path: Path) -> None:
    import json

    from verity.provisioning.backend import ProvisioningError

    def _no_launch(_s):  # type: ignore[no-untyped-def]
        raise ProvisioningError("no docker daemon")

    error = _collect_failure(FakeBackend(script=_no_launch), tmp_path / "ws")
    telemetry = json.loads(error.telemetry or b"{}")
    assert telemetry["stop_reason"] == "launch_failure" and telemetry["source"] == "host"
    assert "no docker daemon" in telemetry["error"]


def test_timeout_cycle_run_report_carries_synthesized_telemetry(tmp_path: Path) -> None:
    # The full #125 chain: a timed-out worker → host synthesis → SandboxError carry → the failed
    # cycle's RunReport entry says WHY it stopped, with the token sums recovered from the turns.
    from verity.contracts import (
        Artifact,
        ArtifactStatus,
        Operation,
        OperationStatus,
        ProviderRegistry,
        SandboxPort,
        VerifierPort,
    )
    from verity.control_plane.api import ControlPlane, OrchestrationPolicy
    from verity.control_plane.config import TaskConfig
    from verity.control_plane.registries import DefaultRetrievalPolicy
    from verity.control_plane.store import SqliteStore
    from verity.domains.fake import SOURCE, build_fake_domain, build_fake_verifier

    store = SqliteStore()
    store.propose(
        Artifact("src", SOURCE, {"raw": 1}, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation("op-src", "load", (), "src", OperationStatus.SUCCESS, "t0"),
    )
    domain = build_fake_domain()
    backend = FakeBackend(script=lambda _s: CompletedWorker(
        exit_code=-1, stdout="", stderr=_KILLED_STDERR, timed_out=True,
    ))
    sandbox = AgentSandbox(
        root=tmp_path / "ws",
        driver=BackendSandboxDriver(backend=backend, timeout_s=5.0),
        schema=domain.schema, proposer_identity="p",
    )
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("real", lambda: sandbox)
    vp.register("real", lambda: build_fake_verifier())
    cp = ControlPlane(
        store, policy=OrchestrationPolicy(max_cycles=1), sandbox_providers=sp,
        verifier_providers=vp,
    )
    asyncio.run(cp.configure(TaskConfig(
        task_id="t1", instructions="i", domain_instructions="d",
        schema=domain.schema, gated_types=domain.gated_types, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator, object_namer=lambda _a: frozenset(),
        sandbox_key="real", verifier_key="real",
    )))
    asyncio.run(cp.run("t1", goal="go"))

    cycle = cp.run_report("t1").cycles[0]
    assert cycle.sandbox_error is not None and "timed out" in cycle.sandbox_error
    assert cycle.agent_telemetry is not None
    assert cycle.agent_telemetry["stop_reason"] == "sandbox_timeout"
    assert cycle.agent_telemetry["total_tokens"] == 580  # recovered from the streamed turns
    assert cycle.transcript_ref is not None  # the turns are content-addressed as usual
