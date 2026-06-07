"""Code-runner seam + auto-code-runner primitive (spec §3.6, §11, §12) — ROADMAP Phase 2.2.

Unit tests run offline against :class:`FakeCodeRunner`. The ``@pytest.mark.docker`` integration
tests exercise the real :class:`ContainerCodeRunner` and **auto-skip** when no Docker daemon is
reachable, so CI needs no Docker-in-CI. They prove the hardening claims concretely: a result file
round-trips, a crash is captured, and the no-network / read-only mounts actually hold.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from verity.contracts import VerifierRequest
from verity.control_plane.commit import GateVerdict
from verity.control_plane.store import Artifact, ArtifactStatus, VerdictKind
from verity.verifier import (
    ContainerCodeRunner,
    FakeCodeRunner,
    RunRequest,
    RunResult,
    auto_code_runner,
    docker_available,
)


def docker_test(fn):
    """Mark a test ``docker`` (so ``-m docker`` / ``-m "not docker"`` select it) and skip it when
    no Docker daemon is reachable (so CI needs no Docker-in-CI)."""
    fn = pytest.mark.skipif(not docker_available(), reason="no Docker daemon reachable")(fn)
    return pytest.mark.docker(fn)


def _request(*, objects=None, gate: str = "runs") -> VerifierRequest:
    proposal = Artifact("a1", "Submission", {"v": 1}, ArtifactStatus.PROPOSED, "agent", "t1")
    return VerifierRequest(proposal=proposal, gate=gate, objects=objects or {})


def _run(primitive, request):
    return asyncio.run(primitive(request))


# ----------------------------------------------------------------- the fake runner


def test_fake_runner_records_calls_and_returns_scripted_result() -> None:
    runner = FakeCodeRunner(script=lambda _r: RunResult(exit_code=0, stdout="hi", stderr=""))
    result = asyncio.run(runner.run(RunRequest(code=b"print('hi')")))
    assert result.stdout == "hi"
    assert runner.calls[0].code == b"print('hi')"


# ----------------------------------------------------------------- the auto-code-runner primitive


def test_missing_attachment_is_a_reject_not_a_crash() -> None:
    primitive = auto_code_runner(FakeCodeRunner())
    verdict = _run(primitive, _request(objects={}))  # agent submitted no runnable code
    assert verdict is not None and verdict.kind is VerdictKind.REJECT


def test_clean_run_accepts_by_default() -> None:
    runner = FakeCodeRunner(script=lambda _r: RunResult(exit_code=0, stdout="", stderr=""))
    verdict = _run(auto_code_runner(runner), _request(objects={"submission.py": b"x=1"}))
    assert verdict is not None and verdict.kind is VerdictKind.ACCEPT


def test_nonzero_exit_rejects_with_stderr_tail() -> None:
    runner = FakeCodeRunner(
        script=lambda _r: RunResult(exit_code=1, stdout="", stderr="Trace\nValueError: bad")
    )
    verdict = _run(auto_code_runner(runner), _request(objects={"submission.py": b"raise"}))
    assert verdict is not None and verdict.kind is VerdictKind.REJECT
    assert "ValueError: bad" in verdict.rationale


def test_timeout_rejects() -> None:
    runner = FakeCodeRunner(script=lambda _r: RunResult(-1, "", "", timed_out=True))
    verdict = _run(auto_code_runner(runner), _request(objects={"submission.py": b"while 1: pass"}))
    assert verdict is not None and verdict.kind is VerdictKind.REJECT
    assert "timed out" in verdict.rationale


def test_inputs_fn_supplies_datasets_to_the_run() -> None:
    runner = FakeCodeRunner()
    primitive = auto_code_runner(runner, inputs=lambda _req: {"train.csv": b"a,b\n1,2\n"})
    _run(primitive, _request(objects={"submission.py": b"x=1"}))
    assert runner.calls[0].inputs == {"train.csv": b"a,b\n1,2\n"}


def test_custom_interpret_scores_from_the_output_file() -> None:
    # a domain scores by reading result.output; here: accept iff the reported score clears 0.8
    def interpret(result: RunResult) -> GateVerdict:
        score = json.loads(result.output or b"{}").get("score", 0.0)
        kind = VerdictKind.ACCEPT if score >= 0.8 else VerdictKind.REJECT
        return GateVerdict(kind, f"score {score}", score=score)

    runner = FakeCodeRunner(
        script=lambda _r: RunResult(0, "", "", output=b'{"score": 0.91}')
    )
    verdict = _run(
        auto_code_runner(runner, interpret=interpret),
        _request(objects={"submission.py": b"x=1"}),
    )
    assert verdict is not None and verdict.kind is VerdictKind.ACCEPT
    assert verdict.score == pytest.approx(0.91)


def test_docker_available_returns_a_bool() -> None:
    assert isinstance(docker_available(), bool)


# ----------------------------------------------------------------- container integration (Docker)


@docker_test
def test_container_runs_real_code_reads_data_and_round_trips_output() -> None:
    code = b"""
import json, pathlib
nums = pathlib.Path('/data/nums.txt').read_text().split()
total = sum(int(n) for n in nums)
pathlib.Path('/out/result.json').write_text(json.dumps({'total': total}))
print('computed', total)
"""
    runner = ContainerCodeRunner()
    result = asyncio.run(
        runner.run(RunRequest(code=code, inputs={"nums.txt": b"1 2 3 4"}, timeout_s=60))
    )
    assert result.exit_code == 0, result.stderr
    assert result.timed_out is False
    assert "computed 10" in result.stdout
    assert result.output is not None and json.loads(result.output) == {"total": 10}


@docker_test
def test_container_captures_a_crash() -> None:
    code = b"raise ValueError('boom')\n"
    result = asyncio.run(ContainerCodeRunner().run(RunRequest(code=code, timeout_s=60)))
    assert result.exit_code != 0
    assert result.timed_out is False
    assert "ValueError: boom" in result.stderr
    assert result.output is None  # no result file written


@docker_test
def test_container_has_no_network() -> None:
    # --network=none means even a DNS/TCP attempt fails fast; the script exits non-zero
    code = b"""
import socket
socket.create_connection(('1.1.1.1', 53), timeout=5)
print('REACHED NETWORK')
"""
    result = asyncio.run(ContainerCodeRunner().run(RunRequest(code=code, timeout_s=60)))
    assert result.exit_code != 0
    assert "REACHED NETWORK" not in result.stdout


@docker_test
def test_container_work_mount_is_read_only() -> None:
    # code is mounted read-only at /work; writing there must fail (only /out is writable)
    code = b"""
import pathlib
pathlib.Path('/work/evil.py').write_text('pwned')
print('WROTE TO WORK')
"""
    result = asyncio.run(ContainerCodeRunner().run(RunRequest(code=code, timeout_s=60)))
    assert result.exit_code != 0
    assert "WROTE TO WORK" not in result.stdout
