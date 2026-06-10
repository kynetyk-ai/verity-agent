"""The backend-backed code-runner (ROADMAP 7.4.d). Offline via `FakeBackend`; real run is `@docker`.

This is the resolution of the dropped S5 question: untrusted submitted code runs in a launched,
isolated **worker**, not an in-process subprocess. The offline tests pin the `RunRequest` ->
`WorkerSpec` mapping (both postures), the output mapping, and infra-failure degradation; the
`@docker` smoke runs real code in a real container; and `auto_code_runner` composes over it.
"""

from __future__ import annotations

import asyncio

import pytest

from verity.provisioning import DockerBackend, FakeBackend
from verity.provisioning.backend import CompletedWorker, ProvisioningError, WorkerSpec
from verity.verifier import (
    BackendCodeRunner,
    RunRequest,
    auto_code_runner,
    docker_available,
)


def docker_test(fn):  # type: ignore[no-untyped-def]
    fn = pytest.mark.skipif(not docker_available(), reason="no Docker daemon reachable")(fn)
    return pytest.mark.docker(fn)


# ----------------------------------------------------------------- RunRequest -> WorkerSpec mapping


def test_no_deps_request_maps_to_a_plain_python_worker() -> None:
    backend = FakeBackend(
        script=lambda _s: CompletedWorker(
            exit_code=0, stdout="", stderr="", outputs={"/out/result.json": b'{"score": 0.9}'}
        )
    )
    runner = BackendCodeRunner(backend=backend)
    result = asyncio.run(
        runner.run(RunRequest(code=b"print(1)", inputs={"nums.txt": b"1 2 3"}))
    )

    assert result.exit_code == 0
    assert result.output == b'{"score": 0.9}'  # the declared output file, harvested
    spec = backend.launched[0]
    assert spec.command == ("python", "/work/submission.py")
    assert spec.labels.role == "code-runner"
    assert spec.readonly_inputs["/work/submission.py"] == b"print(1)"  # the code is ro
    assert spec.readonly_inputs["/data/nums.txt"] == b"1 2 3"  # the dataset is ro
    assert spec.writable_dirs == ("/out",)
    assert spec.output_globs == ("/out/result.json",)
    assert spec.network is False  # the strict hostile-input default
    assert spec.scratch.allow_exec is False


def test_deps_request_pip_installs_in_an_exec_tmpfs_with_network() -> None:
    backend = FakeBackend()
    runner = BackendCodeRunner(backend=backend)
    asyncio.run(
        runner.run(
            RunRequest(code=b"import sklearn", requirements=b"scikit-learn==1.5\n", network=True)
        )
    )
    spec = backend.launched[0]
    assert spec.command[0] == "sh" and "pip install" in spec.command[2]
    assert spec.readonly_inputs["/work/requirements.txt"] == b"scikit-learn==1.5\n"
    assert spec.scratch.allow_exec is True  # native wheels need an exec tmpfs
    assert spec.network is True


def test_a_failed_launch_degrades_to_gate_unavailable() -> None:
    class _Failing:
        async def run_to_completion(self, spec: WorkerSpec) -> CompletedWorker:
            raise ProvisioningError("no docker daemon")

    runner = BackendCodeRunner(backend=_Failing())  # type: ignore[arg-type]
    from verity.contracts import GateUnavailable

    with pytest.raises(GateUnavailable, match="could not run the submitted code"):
        asyncio.run(runner.run(RunRequest(code=b"print(1)")))


def test_auto_code_runner_composes_over_the_backend_runner() -> None:
    # the gate primitive treats it like any CodeRunner: a clean run accepts.
    from verity.contracts import Artifact, ArtifactStatus, VerdictKind, VerifierRequest

    primitive = auto_code_runner(BackendCodeRunner(backend=FakeBackend()))
    proposal = Artifact("a1", "Submission", {"v": 1}, ArtifactStatus.PROPOSED, "agent", "t1")
    request = VerifierRequest(proposal=proposal, objects={"submission.py": b"x = 1"})
    verdict = asyncio.run(primitive(request))
    assert verdict is not None and verdict.kind is VerdictKind.ACCEPT


# ----------------------------------------------------------------- real run (@docker)


@docker_test
def test_real_submission_reads_data_and_writes_output() -> None:
    code = (
        "import os, json, pathlib\n"
        "d = pathlib.Path(os.environ['VERITY_DATA']); o = pathlib.Path(os.environ['VERITY_OUT'])\n"
        "total = sum(int(x) for x in (d / 'nums.txt').read_text().split())\n"
        "(o / 'result.json').write_text(json.dumps({'total': total}))\n"
    )
    runner = BackendCodeRunner(backend=DockerBackend())
    result = asyncio.run(
        runner.run(
            RunRequest(
                code=code.encode(), inputs={"nums.txt": b"1 2 3 4"},
                env={"VERITY_DATA": "/data", "VERITY_OUT": "/out"}, timeout_s=60,
            )
        )
    )
    assert result.exit_code == 0, result.stderr
    import json

    assert result.output is not None and json.loads(result.output) == {"total": 10}
