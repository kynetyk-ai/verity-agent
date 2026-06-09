"""The `WorkerBackend` contract (ROADMAP 7.4.a). Offline — no Docker, no consumers yet.

The contract is validated from both sides in one place: the **consumer side** is "wishful" tests
that build a `WorkerSpec` exactly as the not-yet-written backend-backed driver / code-runner will
(the ergonomics are pinned now — an awkward spec means the interface is wrong); the
**implementor side** is `FakeBackend`. The two specs mirror the two real argv builders
(`container_driver.docker_command`, `code_runner._docker_cmd`), so passing them proves the one
`WorkerSpec` expresses today's actual launch requirements.
"""

from __future__ import annotations

import asyncio

from verity.provisioning import (
    CompletedWorker,
    FakeBackend,
    Labels,
    ResourceLimits,
    Tmpfs,
    WorkerSpec,
    WorkerStatus,
)

# ----------------------------------------------------------------- wishful: the two real postures


def test_sandbox_posture_runs_and_harvests_the_outbox() -> None:
    # The spec a BackendSandboxDriver will build: network ON, /work writable, the gold roles + the
    # cycle input seeded READ-ONLY (so §3.5 gold-data isolation stays physical), outbox harvested.
    spec = WorkerSpec(
        image="verity-sandbox:latest",
        command=("python", "-m", "verity.sandbox.container_entry"),
        labels=Labels(role="sandbox", run="r1", cycle="0", config="fe"),
        readonly_inputs={
            "sandbox/input.json": b'{"system_prompt": "..."}',
            "work/data/train.csv": b"a,b\n1,2\n",
            "work/spec/schema.json": b"{}",
        },
        writable_dirs=("/work",),
        output_globs=("outbox/*",),
        network=True,
        limits=ResourceLimits(memory="4g", cpus="2", pids=512),
        scratch=Tmpfs(size="256m"),
        timeout_s=1800.0,
    )
    backend = FakeBackend(
        script=lambda _s: CompletedWorker(
            exit_code=0, stdout="ok", stderr="",
            outputs={"outbox/__proposal__.json": b'{"op_name": "submit"}'},
        )
    )

    result = asyncio.run(backend.run_to_completion(spec))

    assert result.exit_code == 0
    assert result.outputs["outbox/__proposal__.json"] == b'{"op_name": "submit"}'
    assert backend.launched[0] is spec  # the worker got exactly this spec
    # the gold roles + cycle input are read-only, NOT in the writable area (the isolation invariant)
    assert "work/data/train.csv" in spec.readonly_inputs
    assert spec.writable_dirs == ("/work",)
    assert spec.network is True


def test_code_runner_posture_no_network_collects_result() -> None:
    # The spec a BackendCodeRunner will build from a RunRequest: code + datasets read-only, /out
    # writable, the deps tmpfs marked exec, one result file harvested.
    spec = WorkerSpec(
        image="python:3.12-slim",
        command=("sh", "-c", "pip install -r requirements.txt && python submission.py"),
        labels=Labels(role="code-runner", run="r1", cycle="0", config="fe"),
        readonly_inputs={
            "work/submission.py": b"print('train')",
            "work/requirements.txt": b"scikit-learn==1.5.0\n",
            "data/train.csv": b"a,b\n1,2\n",
        },
        writable_dirs=("/out",),
        output_globs=("out/result.json",),
        network=True,  # deps install needs outbound (the gate sets this); strict default is False
        limits=ResourceLimits(memory="2g", cpus="1", pids=128),
        scratch=Tmpfs(size="1g", allow_exec=True),
        timeout_s=600.0,
    )
    backend = FakeBackend(
        script=lambda _s: CompletedWorker(
            exit_code=0, stdout="", stderr="", outputs={"out/result.json": b'{"score": 0.91}'}
        )
    )

    result = asyncio.run(backend.run_to_completion(spec))

    assert result.outputs["out/result.json"] == b'{"score": 0.91}'
    assert spec.scratch.allow_exec is True  # the deps path
    assert "work/submission.py" in spec.readonly_inputs  # the script can't rewrite itself
    assert spec.writable_dirs == ("/out",)


def test_default_backend_is_a_clean_no_output_run() -> None:
    spec = WorkerSpec(image="x", command=("true",), labels=Labels(role="sandbox"))
    result = asyncio.run(FakeBackend().run_to_completion(spec))
    assert result.exit_code == 0 and result.outputs == {}


# ----------------------------------------------------------------- value types


def test_labels_as_dict_drops_empties_and_matches_is_partial() -> None:
    labels = Labels(role="sandbox", tenant="acme", run="r9", cycle="3")
    assert labels.as_dict() == {
        "harness": "verity", "tenant": "acme", "run": "r9", "cycle": "3", "role": "sandbox"
    }
    assert labels.matches({"run": "r9", "role": "sandbox"})  # partial query
    assert not labels.matches({"run": "r9", "cycle": "4"})  # cycle differs


def test_resource_and_tmpfs_defaults() -> None:
    assert ResourceLimits() == ResourceLimits(memory="512m", cpus="1", pids=128)
    assert Tmpfs().allow_exec is False


# ----------------------------------------------------------------- fleet management (powers GC)


def test_list_and_reap_select_by_label() -> None:
    backend = FakeBackend()

    def _spec(run: str, cycle: str) -> WorkerSpec:
        return WorkerSpec("x", ("true",), Labels(role="sandbox", run=run, cycle=cycle))

    async def go() -> tuple[int, list[str], int]:
        await backend.launch(_spec("r1", "0"))
        await backend.launch(_spec("r1", "1"))
        await backend.launch(_spec("r2", "0"))
        r1 = await backend.list({"run": "r1"})
        roles = [h.labels.role for h in r1]
        reaped = await backend.reap({"run": "r1"})
        return len(r1), roles, reaped

    count, roles, reaped = asyncio.run(go())
    assert count == 2 and roles == ["sandbox", "sandbox"]
    assert reaped == 2
    # r2 survives the run-scoped reap
    assert len(asyncio.run(backend.list({}))) == 1


def test_status_is_gone_after_destroy() -> None:
    backend = FakeBackend()

    async def go() -> tuple[WorkerStatus, WorkerStatus]:
        handle = await backend.launch(WorkerSpec("x", ("true",), Labels(role="verifier")))
        running = await backend.status(handle)
        await backend.destroy(handle)
        return running, await backend.status(handle)

    running, gone = asyncio.run(go())
    assert running is WorkerStatus.RUNNING and gone is WorkerStatus.GONE
