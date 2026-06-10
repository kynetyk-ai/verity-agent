"""The 7.4.g done-line: the feature-engineering task, built **purely from declarative config**.

This is the capstone of ROADMAP 7.4 (ADR 0003): no hand-built drivers, code-runners, or
registries — just ``build_fe_control_plane(backend=..., split=...)`` and a backend selection. The
read → propose → gate → commit loop runs entirely over the `WorkerBackend` seam: each cycle's agent
runs as a ``role=sandbox`` worker, and the verifier executes the untrusted submission as a
``role=code-runner`` worker. The same builder runs offline on `FakeBackend` (this module's
deterministic capstone) and on real Docker + a real model (the ``@live`` demonstration).

Four properties are asserted on both paths (the offline one captures every `WorkerSpec`, the live
one wraps the real backend to do the same):

1. an accepted `Submission` lands;
2. **no cross-cycle bleed (§3.5)** — every cycle's sandbox worker is handed the *pristine* gold
   data, so a hostile prior cycle cannot corrupt what the next one sees;
3. the untrusted submitted code runs in a ``{role:code-runner}``-labelled worker (not in-process);
4. the gate's **reserved labels** (the answer key) never enter *any* worker, by construction.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from verity.composition import ProvisioningConfig, build_fe_control_plane
from verity.contracts import ArtifactStatus
from verity.control_plane.store import SqliteStore
from verity.domains.feature_engineering import (
    ENTRYPOINT,
    PREDICTIONS_OUTPUT,
    REQUIREMENTS,
    SUBMISSION,
)
from verity.provisioning import DockerBackend, FakeBackend, docker_available
from verity.provisioning.backend import CompletedWorker, WorkerBackend, WorkerHandle, WorkerSpec
from verity.sandbox import ProposalDescriptor
from verity.sandbox.container_io import CONTAINER_OUTBOX
from verity.sandbox.descriptor import RESERVED_PROPOSAL_NAME

# The reserved rows the gate scores against — held privately by the verifier, never sent to workers.
RESERVED = {"1": "STAR", "2": "STAR", "3": "GALAXY", "4": "GALAXY", "5": "QSO", "6": "QSO"}


@dataclass(frozen=True)
class _Split:
    """A stratified split's shape: the agent's labelled data, the unlabelled reserved rows, and the
    reserved answer key the verifier keeps to itself."""

    agent_train_csv: bytes = b"id,class\n0,STAR\n7,GALAXY\n8,QSO\n"
    reserved_test_csv: bytes = b"id\n1\n2\n3\n4\n5\n6\n"
    reserved_labels: dict[str, str] = field(default_factory=lambda: dict(RESERVED))


def _preds_csv(preds: dict[str, str]) -> bytes:
    return ("id,class\n" + "\n".join(f"{k},{v}" for k, v in preds.items()) + "\n").encode()


def _entrypoint(marker: bytes, names: list[str]) -> bytes:
    """A submission script carrying a marker (the fake runner keys predictions off it) and the
    declared feature names (so the static features-defined gate is satisfied)."""
    return marker + b"\n# defines: " + " ".join(names).encode()


# A submission: marker -> the predictions the runner returns for that code.
_OK = (b"ok", {**RESERVED, "5": "STAR", "6": "STAR"})  # 0.667 balanced accuracy -> sets incumbent
_GOOD = (b"good", dict(RESERVED))  # 1.0 -> beats the incumbent, supersedes


@dataclass
class _FeWorkers:
    """Plays *both* worker roles for the FE task over a `WorkerBackend`, fully config-driven.

    A ``sandbox`` worker emits the next scripted ``(op, parents, marker, names)`` proposal into the
    outbox; a ``code-runner`` worker "runs" the submitted script (marker -> predictions) and writes
    ``predictions.csv``. One callable, dispatched on the worker's ``role`` label — so a single
    `FakeBackend` drives the whole loop with no Docker and no model.
    """

    steps: list[tuple[str, tuple[str, ...], bytes, list[str]]]
    by_marker: dict[bytes, dict[str, str]]
    cursor: int = 0

    def __call__(self, spec: WorkerSpec) -> CompletedWorker:
        role = spec.labels.role
        if role == "sandbox":
            return self._propose()
        if role == "code-runner":
            return self._run_code(spec)
        return CompletedWorker(exit_code=1, stdout="", stderr=f"unexpected worker role {role!r}")

    def _propose(self) -> CompletedWorker:
        if self.cursor >= len(self.steps):  # defensive: a surplus cycle proposes nothing
            return CompletedWorker(exit_code=0, stdout="", stderr="")
        op_name, parents, marker, names = self.steps[self.cursor]
        self.cursor += 1
        features = [{"name": n, "definition": "d", "rationale": "r"} for n in names]
        payload = {"entrypoint": ENTRYPOINT, "requirements": REQUIREMENTS, "features": features}
        descriptor = ProposalDescriptor(op_name, parents, payload, metadata="i reasoned thus")
        return CompletedWorker(
            exit_code=0, stdout="", stderr="",
            outputs={
                f"{CONTAINER_OUTBOX}/{ENTRYPOINT}": _entrypoint(marker, names),
                f"{CONTAINER_OUTBOX}/{REQUIREMENTS}": b"pandas==2.2.2",
                f"{CONTAINER_OUTBOX}/{RESERVED_PROPOSAL_NAME}": descriptor.to_json(),
            },
        )

    def _run_code(self, spec: WorkerSpec) -> CompletedWorker:
        code = spec.readonly_inputs.get(f"/work/{ENTRYPOINT}", b"")
        for marker, preds in self.by_marker.items():
            if marker in code:
                return CompletedWorker(
                    exit_code=0, stdout="", stderr="",
                    outputs={f"/out/{PREDICTIONS_OUTPUT}": _preds_csv(preds)},
                )
        return CompletedWorker(exit_code=0, stdout="", stderr="")  # no scorable output


@dataclass
class _Recording:
    """Wraps a real `WorkerBackend`, recording every `WorkerSpec` so the live path can run the same
    structural assertions as the offline one. Delegates the whole port to the inner backend."""

    inner: WorkerBackend
    launched: list[WorkerSpec] = field(default_factory=list)

    async def run_to_completion(self, spec: WorkerSpec) -> CompletedWorker:
        self.launched.append(spec)
        return await self.inner.run_to_completion(spec)

    async def launch(self, spec: WorkerSpec) -> WorkerHandle:
        self.launched.append(spec)
        return await self.inner.launch(spec)

    async def wait(self, handle: WorkerHandle, *, timeout_s: float) -> CompletedWorker:
        return await self.inner.wait(handle, timeout_s=timeout_s)

    async def status(self, handle: WorkerHandle) -> object:
        return await self.inner.status(handle)

    async def logs(self, handle: WorkerHandle) -> bytes:
        return await self.inner.logs(handle)

    async def destroy(self, handle: WorkerHandle) -> None:
        await self.inner.destroy(handle)

    async def list(self, selector: object) -> list[WorkerHandle]:
        return await self.inner.list(selector)  # type: ignore[arg-type]

    async def reap(self, selector: object) -> int:
        return await self.inner.reap(selector)  # type: ignore[arg-type]


def _assert_isolation(launched: list[WorkerSpec], split: _Split) -> None:
    """The three structural §3.5 / topology-(b) properties, over the captured worker specs."""
    sandboxes = [s for s in launched if s.labels.role == "sandbox"]
    code_runners = [s for s in launched if s.labels.role == "code-runner"]

    # (3) the untrusted submitted code ran in a code-runner-labelled worker — not in-process.
    assert code_runners, "no code-runner worker was launched"
    assert all(f"/work/{ENTRYPOINT}" in s.readonly_inputs for s in code_runners)

    # (2) every cycle's sandbox worker got the pristine gold — no prior cycle could corrupt it.
    assert len(sandboxes) >= 2, "expected a fresh sandbox worker per cycle"
    assert all(
        s.readonly_inputs.get("/work/data/train.csv") == split.agent_train_csv for s in sandboxes
    )

    # (4) the reserved answer key never enters any worker (inputs or env).
    answer_key = _preds_csv(split.reserved_labels)
    for spec in launched:
        for blob in spec.readonly_inputs.values():
            assert answer_key not in blob, f"answer key leaked into a {spec.labels.role} input"
        for value in spec.env.values():
            assert answer_key.decode() not in value
        # the code-runner is handed only the unlabelled reserved rows, never a 'class' column
        if spec.labels.role == "code-runner":
            assert spec.readonly_inputs.get("/data/test.csv") == split.reserved_test_csv
            assert b"class" not in split.reserved_test_csv


def test_fe_accepts_via_config_offline() -> None:
    """The deterministic capstone: FE built purely from config, the loop run over `FakeBackend`."""
    split = _Split()
    workers = _FeWorkers(
        steps=[
            ("submit", ("ds",), _OK[0], ["feat0"]),   # cycle 1: runs clean, 0.667 -> accepted
            ("submit", ("ds",), _GOOD[0], ["feat1"]),  # cycle 2: 1.0 -> supersedes the incumbent
        ],
        by_marker=dict([_OK, _GOOD]),
    )
    backend = FakeBackend(script=workers)
    store = SqliteStore()
    cp, config = build_fe_control_plane(backend=backend, split=split, max_cycles=2, store=store)
    asyncio.run(cp.configure(config))
    asyncio.run(cp.run("fe", goal="improve balanced accuracy"))

    # (1) an accepted submission landed.
    accepted = store.query_artifacts(type=SUBMISSION, status=ArtifactStatus.ACCEPTED)
    assert accepted, "no accepted Submission produced from the config-driven run"
    # (2)-(4) the isolation properties hold over every worker the run launched.
    _assert_isolation(backend.launched, split)


# ----------------------------------------------------- the live, fully-containerized demonstration


@pytest.mark.live
def test_fe_accepts_via_config_live() -> None:
    """The done-line, for real: FE driven entirely by control-plane config over Docker workers and a
    real model. Same builder, a `DockerBackend` swapped in for the `FakeBackend`. Manual / billed —
    auto-skips without a key, Docker, the dataset, or the ``sandbox`` extra."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("no ANTHROPIC_API_KEY")
    if not docker_available():
        pytest.skip("no Docker for the worker backend")
    dataset = Path("feature-engineering-test/train.csv")
    if not dataset.exists():
        pytest.skip("the stellar dataset is not present")
    from tools.harness.dataset import stratified_split

    raw = _subsample(dataset.read_bytes(), per_class=400)  # a small slice for fast per-cycle work
    real = stratified_split(raw, target="class", id_column="id", reserved_fraction=0.5)

    backend = _Recording(inner=DockerBackend())
    store = SqliteStore()
    cp, config = build_fe_control_plane(
        backend=backend, split=real, provisioning=ProvisioningConfig(), max_cycles=4, store=store
    )
    asyncio.run(cp.configure(config))
    asyncio.run(cp.run("fe", goal="Improve balanced accuracy via feature engineering."))

    accepted = store.query_artifacts(type=SUBMISSION, status=ArtifactStatus.ACCEPTED)
    assert accepted, "no accepted Submission across the live, containerized run"
    _assert_isolation(backend.launched, real)  # type: ignore[arg-type]


def _subsample(data: bytes, *, per_class: int, target: str = "class") -> bytes:
    """Keep up to ``per_class`` rows per target class — a fast, stratified slice for a live run."""
    import csv
    import io

    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    header = list(reader.fieldnames or [])
    seen: dict[str, int] = {}
    kept: list[dict[str, str]] = []
    for row in reader:
        label = row[target]
        if seen.get(label, 0) < per_class:
            seen[label] = seen.get(label, 0) + 1
            kept.append(row)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header)
    writer.writeheader()
    writer.writerows(kept)
    return buf.getvalue().encode("utf-8")
