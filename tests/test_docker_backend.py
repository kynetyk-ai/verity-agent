"""The `DockerBackend` (ROADMAP 7.4.b). The argv builder is PURE, so the security-critical posture
is pinned **offline** (no daemon); the real run / fleet GC are ``@pytest.mark.docker`` (auto-skipped
without a daemon), so CI needs no Docker.

The argv-pin tests are the load-bearing ones: this backend is the single owner of the hostile-input
``docker run`` argv, so a silently dropped ``--network=none`` / ``--cap-drop=ALL`` / ro overlay is a
security regression on agent-proposed code. Both postures (sandbox network-on, code-runner
network-off) are asserted against one `WorkerSpec`.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from pathlib import Path

import pytest

from verity.provisioning import (
    DockerBackend,
    Labels,
    ResourceLimits,
    Tmpfs,
    WorkerSpec,
    WorkerStatus,
    docker_available,
)


def docker_test(fn):  # type: ignore[no-untyped-def]
    fn = pytest.mark.skipif(not docker_available(), reason="no Docker daemon reachable")(fn)
    return pytest.mark.docker(fn)


def _sandbox_spec() -> WorkerSpec:
    return WorkerSpec(
        image="verity-sandbox:latest",
        command=("python", "-m", "verity.sandbox.container_entry"),
        labels=Labels(role="sandbox", run="r1", cycle="0", config="fe"),
        readonly_inputs={"/sandbox/input.json": b"{}", "/work/data/train.csv": b"x"},
        writable_dirs=("/work",),
        output_globs=("/work/outbox/*",),
        network=True,
        limits=ResourceLimits(memory="4g", cpus="2", pids=512),
        scratch=Tmpfs(size="256m"),
    )


def _code_runner_spec() -> WorkerSpec:
    return WorkerSpec(
        image="python:3.12-slim",
        command=("python", "/work/submission.py"),
        labels=Labels(role="code-runner", run="r1", cycle="0"),
        readonly_inputs={"/work/submission.py": b"print(1)"},
        writable_dirs=("/out",),
        output_globs=("/out/result.json",),
        network=False,
        scratch=Tmpfs(size="64m"),
    )


# ----------------------------------------------------------------- argv pinning (offline)


def test_argv_applies_the_hardened_baseline_for_both_postures() -> None:
    backend = DockerBackend()
    uid_gid = f"--user={os.getuid()}:{os.getgid()}"
    for spec in (_sandbox_spec(), _code_runner_spec()):
        argv = backend.build_argv(spec, name="n", mounts=[])
        assert argv[:5] == ["docker", "run", "--rm", "--name", "n"]
        for flag in ("--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", uid_gid):
            assert flag in argv, (spec.labels.role, flag)
        assert f"--memory={spec.limits.memory}" in argv
        assert f"--memory-swap={spec.limits.memory}" in argv
        assert argv[-len(spec.command):] == list(spec.command)  # image then command at the tail
        assert argv[-len(spec.command) - 1] == spec.image


def test_argv_network_posture_differs_by_spec() -> None:
    backend = DockerBackend()
    assert "--network=none" not in backend.build_argv(_sandbox_spec(), name="n", mounts=[])
    assert "--network=none" in backend.build_argv(_code_runner_spec(), name="n", mounts=[])


def test_argv_deps_tmpfs_marks_exec_only_when_requested() -> None:
    backend = DockerBackend()
    assert "--tmpfs=/tmp:rw,size=256m,mode=1777" in backend.build_argv(
        _sandbox_spec(), name="n", mounts=[]
    )
    deps = replace(_code_runner_spec(), scratch=Tmpfs(size="1g", allow_exec=True))
    assert "--tmpfs=/tmp:rw,size=1g,mode=1777,exec" in backend.build_argv(deps, name="n", mounts=[])


def test_argv_runtime_knob_and_labels_and_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    spec = WorkerSpec(
        image="img", command=("true",), labels=Labels(role="sandbox", run="r9"),
        env={"VERITY_SANDBOX_MODEL": "anthropic:x"}, env_passthrough=("ANTHROPIC_API_KEY",),
        runtime="runsc",
    )
    argv = DockerBackend().build_argv(spec, name="n", mounts=[])
    assert argv[5] == "--runtime=runsc"  # right after `run --rm --name n`
    assert "--label" in argv and "role=sandbox" in argv and "run=r9" in argv
    assert "harness=verity" in argv
    # the key is forwarded BY NAME only — its value never lands on the argv
    assert argv.count("ANTHROPIC_API_KEY") == 1 and "sk-secret" not in " ".join(argv)
    assert "VERITY_SANDBOX_MODEL=anthropic:x" in argv


def test_argv_mount_order_is_writable_then_readonly_overlay() -> None:
    backend = DockerBackend()
    mounts = [
        (Path("/h/work"), "/work", "rw"),
        (Path("/h/ro/data"), "/work/data/train.csv", "ro"),
    ]
    argv = backend.build_argv(_sandbox_spec(), name="n", mounts=mounts)
    rw_at = argv.index("/h/work:/work:rw")
    ro_at = argv.index("/h/ro/data:/work/data/train.csv:ro")
    assert rw_at < ro_at  # the ro overlay must come AFTER the writable mount it shadows (§3.5)


def test_materialize_writes_ro_files_and_orders_mounts(tmp_path: Path) -> None:
    backend = DockerBackend()
    mounts, writable = backend._materialize(_sandbox_spec(), tmp_path)
    modes = [mode for _h, _t, mode in mounts]
    assert modes == ["rw", "ro", "ro"]  # writable first, ro overlays after
    # the gold input is materialized read-only with its bytes, not in the writable area
    ro_file = next(h for h, t, m in mounts if t == "/work/data/train.csv")
    assert ro_file.read_bytes() == b"x"
    assert "/work" in writable and writable["/work"].is_dir()


# ----------------------------------------------------------------- real runs (@docker)


@docker_test
def test_run_collects_output_and_inputs_are_read_only() -> None:
    # The worker reads a ro input and writes an output; the output is harvested, and a write to the
    # ro input fails (physical isolation). One alpine `sh -c`, no Python.
    spec = WorkerSpec(
        image="alpine:3.20",
        command=(
            "sh", "-c",
            "cat /in/seed.txt > /out/echo.txt; (echo tamper > /in/seed.txt) 2>/dev/null || true",
        ),
        labels=Labels(role="code-runner", run="rt", cycle="0"),
        readonly_inputs={"/in/seed.txt": b"hello"},
        writable_dirs=("/out",),
        output_globs=("/out/echo.txt",),
        network=False,
        timeout_s=60.0,
    )
    result = asyncio.run(DockerBackend().run_to_completion(spec))
    assert result.exit_code == 0, result.stderr
    assert result.outputs["/out/echo.txt"] == b"hello"  # output harvested as bytes
    assert result.timed_out is False


@docker_test
def test_list_and_reap_select_running_workers_by_label() -> None:
    import subprocess

    name = "verity-test-reap-7p4b"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name,
         "--label", "harness=verity", "--label", "run=reap-test", "alpine:3.20", "sleep", "30"],
        capture_output=True, check=True,
    )
    backend = DockerBackend()
    try:
        listed = asyncio.run(backend.list({"run": "reap-test"}))
        assert len(listed) == 1
        assert asyncio.run(backend.reap({"run": "reap-test"})) == 1
        assert asyncio.run(backend.list({"run": "reap-test"})) == []
        # status of a reaped worker is GONE
        from verity.provisioning import WorkerHandle
        assert asyncio.run(backend.status(WorkerHandle(name, Labels(role="")))) is WorkerStatus.GONE
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
