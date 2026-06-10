"""Shared offline-FE test substrate (ROADMAP 8.1): a config-driven `FakeBackend` worker script.

Not a test module (no ``test_`` prefix). It provides a `FeWorkers` callable that plays *both* FE
worker roles over a `WorkerBackend` — a ``sandbox`` worker emits a scripted proposal descriptor, a
``code-runner`` worker "runs" the submitted script and returns predictions keyed off a marker — so a
single `FakeBackend` drives the whole read→propose→gate→commit loop with no Docker and no model.
This mirrors the capstone in ``tests/test_fe_via_config_acceptance.py`` but is reusable across the
8.1 tests (byte-provenance, per-task isolation, the control service).
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.domains.feature_engineering import ENTRYPOINT, PREDICTIONS_OUTPUT, REQUIREMENTS
from verity.provisioning.backend import CompletedWorker, WorkerSpec
from verity.sandbox import ProposalDescriptor
from verity.sandbox.container_io import CONTAINER_OUTBOX
from verity.sandbox.descriptor import RESERVED_PROPOSAL_NAME


def preds_csv(preds: dict[str, str]) -> bytes:
    return ("id,class\n" + "\n".join(f"{k},{v}" for k, v in preds.items()) + "\n").encode()


def entrypoint_bytes(marker: bytes, names: list[str]) -> bytes:
    """A submission script carrying a marker (the fake runner keys predictions off it) and the
    declared feature names (so the static features-defined gate is satisfied)."""
    return marker + b"\n# defines: " + " ".join(names).encode()


@dataclass
class FeWorkers:
    """Dispatches on the worker's ``role`` label: ``sandbox`` proposes, ``code-runner`` scores.

    ``steps`` is a list of ``(op_name, parents, marker, feature_names)`` proposals, emitted one per
    sandbox cycle; ``by_marker`` maps a submission's marker to the predictions the runner returns
    for it (so a script that predicts the reserved labels exactly scores 1.0 → accepted).
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
        if self.cursor >= len(self.steps):  # a surplus cycle proposes nothing
            return CompletedWorker(exit_code=0, stdout="", stderr="")
        op_name, parents, marker, names = self.steps[self.cursor]
        self.cursor += 1
        features = [{"name": n, "definition": "d", "rationale": "r"} for n in names]
        payload = {"entrypoint": ENTRYPOINT, "requirements": REQUIREMENTS, "features": features}
        descriptor = ProposalDescriptor(op_name, parents, payload, metadata="i reasoned thus")
        return CompletedWorker(
            exit_code=0, stdout="", stderr="",
            outputs={
                f"{CONTAINER_OUTBOX}/{ENTRYPOINT}": entrypoint_bytes(marker, names),
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
                    outputs={f"/out/{PREDICTIONS_OUTPUT}": preds_csv(preds)},
                )
        return CompletedWorker(exit_code=0, stdout="", stderr="")  # no scorable output
