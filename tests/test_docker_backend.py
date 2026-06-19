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


def test_argv_emits_writable_then_readonly_mounts_in_order() -> None:
    backend = DockerBackend()
    mounts = [
        (Path("/h/out"), "/out", "rw"),
        (Path("/h/ro/seed"), "/in/seed.txt", "ro"),
    ]
    argv = backend.build_argv(_code_runner_spec(), name="n", mounts=mounts)
    assert argv.index("/h/out:/out:rw") < argv.index("/h/ro/seed:/in/seed.txt:ro")


def test_staging_root_resolves_from_arg_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERITY_WORKER_STAGING", "/from/env")
    assert DockerBackend()._staging_root == Path("/from/env")  # env when no arg
    assert DockerBackend(staging_root="/from/arg")._staging_root == Path("/from/arg")  # arg wins
    monkeypatch.delenv("VERITY_WORKER_STAGING")
    assert DockerBackend()._staging_root is None  # default: system temp (host-side behaviour)


def test_staging_lands_under_the_configured_root(tmp_path: Path) -> None:
    # The sibling-container fix: when set, every worker's staging dir is created under this shared
    # root (bind-mounted host<->CP-container at an identical path), so worker -v sources resolve on
    # the host daemon.
    staging = DockerBackend(staging_root=tmp_path / "shared")._make_staging()
    assert staging.parent == tmp_path / "shared" and staging.is_dir()
    # default (unset) lands in the system temp dir, not under our root
    assert DockerBackend()._make_staging().parent != tmp_path / "shared"


def test_materialize_nests_inputs_into_the_writable_dir_and_ro_mounts_the_rest(
    tmp_path: Path,
) -> None:
    backend = DockerBackend()
    mounts, writable = backend._materialize(_sandbox_spec(), tmp_path)
    # /work is the one writable mount; the gold input nested under it is NOT a separate mount (a
    # nested file bind-mount fails on Docker Desktop) — it is written into the writable tree.
    assert [mode for _h, _t, mode in mounts] == ["rw", "ro"]
    assert {t for _h, t, _m in mounts} == {"/work", "/sandbox/input.json"}
    gold = writable["/work"] / "data" / "train.csv"
    assert gold.read_bytes() == b"x"  # materialized into the writable mount, agent can read it
    # the non-nested input (the cycle input) stays a plain :ro mount
    ro = next(h for h, t, _m in mounts if t == "/sandbox/input.json")
    assert ro.read_bytes() == b"{}"


# ------------------------------------------- trusted-service launch posture (§9.1, offline)


def _verifier_service_spec() -> WorkerSpec:
    return WorkerSpec(
        image="verity-verifier:latest",
        command=(),
        labels=Labels(role="verifier", config="fe-kaggle"),
        service_name="verity-verifier-abc1234567",
        network_name="verity-net",
        mount_docker_socket=True,
        host_mounts=(("/srv/staging", "/srv/staging", "rw"),),
        env={"VERITY_VERIFIER": "fe-kaggle"},
        env_passthrough=("KAGGLE_KEY",),
    )


def test_service_argv_is_a_trusted_detached_posture() -> None:
    name = "verity-verifier-abc1234567"
    argv = DockerBackend().build_service_argv(_verifier_service_spec(), name=name)
    assert argv[:5] == ["docker", "run", "-d", "--rm", "--name"] and name in argv
    # trusted infrastructure — NONE of the hostile-worker flags
    hostile = ("--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--network=none")  # noqa: E501
    for flag in hostile:
        assert flag not in argv, flag
    assert not any(a.startswith("--user=") for a in argv)
    # joins the user-defined network, mounts the socket + the shared staging at an identical path
    assert argv[argv.index("--network") + 1] == "verity-net"
    assert "/var/run/docker.sock:/var/run/docker.sock" in argv
    assert "/srv/staging:/srv/staging:rw" in argv
    # labels + env still applied; resource limits still bound
    assert "role=verifier" in argv and "config=fe-kaggle" in argv
    assert "VERITY_VERIFIER=fe-kaggle" in argv
    assert any(a.startswith("--memory=") for a in argv)


def test_service_argv_forwards_creds_by_name_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_KEY", "secret-key")
    argv = DockerBackend().build_service_argv(_verifier_service_spec(), name="n")
    # the credential is forwarded BY NAME; its value never lands on the argv
    assert argv.count("KAGGLE_KEY") == 1 and "secret-key" not in " ".join(argv)


def test_hostile_build_argv_never_mounts_the_docker_socket() -> None:
    # The structural §9.1 guarantee: the socket mount lives ONLY on build_service_argv (the launch
    # path). Even a (misconfigured) hostile spec carrying mount_docker_socket/network_name gets no
    # socket and stays network-isolated on the run_to_completion path — build_argv has no branch for
    # them, so an untrusted worker can never acquire the socket.
    spec = replace(_code_runner_spec(), mount_docker_socket=True, network_name="verity-net")
    argv = DockerBackend().build_argv(spec, name="n", mounts=[])
    assert "/var/run/docker.sock:/var/run/docker.sock" not in argv
    assert "--network=none" in argv  # still isolated; network_name is ignored on the hostile path


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
def test_sandbox_shaped_worker_runs_with_inputs_nested_under_the_writable_dir() -> None:
    # The case the live smoke caught: a read-only input nested under a writable /work. A nested file
    # bind-mount fails on Docker Desktop (virtiofs), so it is materialized into the writable mount.
    # The worker must see the input and write its output. One alpine `sh -c`, no Python.
    spec = WorkerSpec(
        image="alpine:3.20",
        command=("sh", "-c", "mkdir -p /work/outbox && cat /work/data/train.csv > /work/outbox/p"),
        labels=Labels(role="sandbox", run="rn", cycle="1"),
        readonly_inputs={"/sandbox/input.json": b"{}", "/work/data/train.csv": b"gold"},
        writable_dirs=("/work",),
        output_globs=("/work/outbox/*",),
        network=False,
        timeout_s=60.0,
    )
    result = asyncio.run(DockerBackend().run_to_completion(spec))
    assert result.exit_code == 0, result.stderr  # no more exit 125 mount failure
    assert result.outputs["/work/outbox/p"] == b"gold"  # the nested input was readable


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
