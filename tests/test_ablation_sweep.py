"""The ablation-ladder sweep orchestrator (experiments/ablation) — offline, end to end.

Drives a small matrix (2 conditions × 1 model × 2 seeds = 4 cells) through a real `ControlService`
wired to the offline hold-out catalog + a stateless `FakeBackend` — no Docker, model, or network.
Pins that the sweep is **config-only**: every cell runs the same task, and the per-cell verdict
policy reaches the real scorer (the decision gate differs — ``score-and-accept`` under ``always`` vs
``selection`` under ``improve_over_best_prior``). Plus the spec's condition→config mapping and the
equal-compute guard.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest
from experiments.ablation.sweep import (
    SweepSpec,
    build_task_request,
    load_role_files,
    run_sweep,
)

from tests._fe_offline import offline_holdout_catalog, preds_csv, role_files_from_raw
from verity.domains.feature_engineering import ENTRYPOINT, PREDICTIONS_OUTPUT, REQUIREMENTS
from verity.provisioning import FakeBackend
from verity.provisioning.backend import CompletedWorker, WorkerSpec
from verity.sandbox import ProposalDescriptor
from verity.sandbox.container_io import CONTAINER_OUTBOX
from verity.sandbox.descriptor import RESERVED_PROPOSAL_NAME
from verity.service.control_service import ControlService

_RAW = b"id,a,class\n0,0,X\n1,2,X\n2,4,Y\n3,6,Y\n4,8,Z\n5,10,Z\n"
_TEST = b"id\n100\n101\n"


@dataclass
class _StatelessFeWorker:
    """A cursor-free FE worker (the sweep reuses one backend across cells): the sandbox always
    proposes the same submission; the code-runner always returns ``reserved`` (a perfect score)."""

    reserved: dict[str, str]

    def __call__(self, spec: WorkerSpec) -> CompletedWorker:
        if spec.labels.role == "sandbox":
            payload = {"entrypoint": ENTRYPOINT, "requirements": REQUIREMENTS}
            descriptor = ProposalDescriptor(
                "submit", ("ds",), payload, metadata="ESTIMATED_BALANCED_ACCURACY: 0.99"
            )
            return CompletedWorker(
                exit_code=0, stdout="", stderr="",
                outputs={
                    f"{CONTAINER_OUTBOX}/{ENTRYPOINT}": b"marker",
                    f"{CONTAINER_OUTBOX}/{REQUIREMENTS}": b"pandas==2.2.2",
                    f"{CONTAINER_OUTBOX}/{RESERVED_PROPOSAL_NAME}": descriptor.to_json(),
                },
            )
        if spec.labels.role == "code-runner":
            return CompletedWorker(
                exit_code=0, stdout="", stderr="",
                outputs={f"/out/{PREDICTIONS_OUTPUT}": preds_csv(self.reserved)},
            )
        return CompletedWorker(
            exit_code=1, stdout="", stderr=f"unexpected role {spec.labels.role!r}"
        )


def _spec_dict(**overrides) -> dict:  # type: ignore[no-untyped-def]
    base = {
        "task_type": "fe-holdout",
        "goal": "optimize balanced accuracy",
        "models": [{"label": "fake", "model": "anthropic:claude-sonnet-4-6"}],
        "seeds": 2,
        "budgets": {"max_cycles": 1},
        "conditions": [
            {"name": "exp2", "accept_policy": "always", "provisioning": "all"},
            {"name": "exp3", "accept_policy": "improve_over_best_prior", "provisioning": "all"},
        ],
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------- spec parsing + mapping


def test_build_task_request_maps_the_condition_axes() -> None:
    spec = SweepSpec.from_dict(_spec_dict())
    exp2, exp3 = spec.conditions
    arm = spec.models[0]

    req2 = build_task_request(spec, exp2, arm, seed=7)
    assert req2.type_name == "fe-holdout"
    assert req2.verifier.knobs["accept_policy"] == "always"
    assert req2.policy.provisioning == "all"
    assert req2.policy.max_cycles == 1  # the shared budget (no per-condition override)
    assert req2.sandbox.extra["seed"] == 7  # seed rides into the model seam for replicates

    req3 = build_task_request(spec, exp3, arm, seed=0)
    assert req3.verifier.knobs["accept_policy"] == "improve_over_best_prior"


def test_single_shot_condition_overrides_max_cycles() -> None:
    spec = SweepSpec.from_dict(
        _spec_dict(conditions=[
            {"name": "exp1", "accept_policy": "always", "provisioning": "none", "max_cycles": 1},
        ])
    )
    req = build_task_request(spec, spec.conditions[0], spec.models[0], seed=0)
    assert req.policy.max_cycles == 1 and req.policy.provisioning == "none"


def test_equal_compute_guard_rejects_a_drifting_loop_budget() -> None:
    with pytest.raises(ValueError, match="equal-compute"):
        SweepSpec.from_dict(
            _spec_dict(
                budgets={"max_cycles": 10},
                conditions=[
                    {"name": "exp3", "accept_policy": "improve_over_best_prior",
                     "provisioning": "all", "max_cycles": 5},  # a loop rung with its own budget
                ],
            )
        )


def test_cells_enumerates_condition_model_seed() -> None:
    spec = SweepSpec.from_dict(_spec_dict())
    assert len(spec.cells()) == 2 * 1 * 2  # conditions × models × seeds


def test_load_role_files_reads_the_prep_layout(tmp_path) -> None:
    (tmp_path / "agent").mkdir()
    (tmp_path / "verifier").mkdir()
    (tmp_path / "agent" / "train.csv").write_bytes(b"id,class\n0,X\n")
    (tmp_path / "agent" / "test.csv").write_bytes(b"id\n1\n")
    (tmp_path / "verifier" / "holdout_labels.csv").write_bytes(b"id,class\n1,X\n")
    roles = load_role_files(tmp_path)
    assert set(roles) == {"agent", "verifier"}
    assert set(roles["agent"]) == {"train.csv", "test.csv"}
    assert roles["verifier"]["holdout_labels.csv"] == b"id,class\n1,X\n"


# ------------------------------------------------------------------- end-to-end offline sweep


def test_run_sweep_runs_every_cell_and_writes_reports(tmp_path) -> None:
    from tools.harness.dataset import stratified_split, subsample

    rf = role_files_from_raw(_RAW, _TEST, per_class=2, reserved_fraction=0.5)
    sub = subsample(_RAW, per_class=2, target="class")
    reserved = dict(
        stratified_split(sub, target="class", id_column="id", reserved_fraction=0.5).reserved_labels
    )
    service = ControlService(
        backend=FakeBackend(script=_StatelessFeWorker(reserved=reserved)),
        root=None, catalog=offline_holdout_catalog(),
    )
    spec = SweepSpec.from_dict(_spec_dict())
    results = asyncio.run(
        run_sweep(spec, service, role_files=rf, results_dir=tmp_path / "out")
    )

    assert len(results) == 4  # 2 conditions × 1 model × 2 seeds
    assert all(r.status == "complete" and r.report is not None for r in results)
    # every cell wrote its RunReport JSON, plus a manifest, for the analysis layer
    written = {p.name for p in (tmp_path / "out").glob("*.json")}
    assert written == {r.key() + ".json" for r in results} | {"manifest.json"}
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert {(m["condition"], m["model"], m["seed"]) for m in manifest} == {
        (r.condition, r.model, r.seed) for r in results
    }
    assert all(m["report"] in written for m in manifest)

    # config-only: the per-cell accept_policy reached the real scorer — the decision gate differs.
    gate_by_condition = {
        r.condition: r.report.cycles[0].commit.decisions[-1].gate
        for r in results
        if r.report is not None and r.report.cycles[0].commit is not None
    }
    assert gate_by_condition == {"exp2": "score-and-accept", "exp3": "selection"}
