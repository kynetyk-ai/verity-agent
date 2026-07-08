"""The ablation-ladder sweep orchestrator (Exp 1-4b, ``docs/experimental-design.md`` §1/§6.3).

Drives the ladder as **pure configuration** over the one `fe-holdout` task type: each cell is a
``(condition, model, seed)`` triple that differs only in the verifier ``accept_policy`` knob, the
``policy.provisioning`` preset, and the loop budget — never in code or wording. A declarative JSON
spec lists the conditions, the model arms, the seed count, and the shared compute budgets; this
module turns each cell into a :class:`TaskRequest`, runs it through a :class:`ControlService`
(create → run → results), and collects the store-derived :class:`RunReport` per cell.

The orchestrator is transport-agnostic: it drives a ``ControlService`` directly. Offline tests
inject the `offline_catalog` + a `FakeBackend`; a real ablation run injects the discovered
`default_catalog` + a `DockerBackend`, which launches real sibling worker containers (the daemon is
the same service behind HTTP — code-writing domains need the container sandbox, never in-process).
Runs are serial by design (``ControlService.run`` blocks), so deltas reflect mechanism, not load.

Shared data (the stellar dataset's role-keyed files) is prepared once outside Verity
(``tools/prepare_fe_data.py``, ADR 0005) and passed to every cell; only configuration varies.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.composition.task_request import (
    DataRequest,
    PolicyRequest,
    SandboxRequest,
    TaskRequest,
    VerifierRequest,
)
from verity.control_plane.run_report import RunReport
from verity.logging import get_logger
from verity.service.control_service import ControlService

log = get_logger("experiments.ablation.sweep")

__all__ = [
    "ModelArm",
    "Condition",
    "Budgets",
    "SweepSpec",
    "CellResult",
    "load_spec",
    "load_role_files",
    "build_task_request",
    "run_sweep",
    "main",
]

#: A role-keyed file set: ``{role: {filename: bytes}}`` (the shared dataset, prepped once).
RoleFiles = Mapping[str, Mapping[str, bytes]]


@dataclass(frozen=True, slots=True)
class ModelArm:
    """One model under test: a display ``label`` + the OpenAI-compatible seam knobs."""

    label: str
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ModelArm:
        return cls(
            label=str(d["label"]), model=d.get("model"), base_url=d.get("base_url"),
            api_key_env=d.get("api_key_env"),
        )


@dataclass(frozen=True, slots=True)
class Condition:
    """One rung of the ladder: the two ablation axes + an optional per-condition cycle override.

    ``max_cycles`` overrides the shared budget for this condition only (Exp 1 is the single-shot
    baseline at 1 cycle; the loop rungs all inherit the shared, equal-compute budget).
    """

    name: str
    accept_policy: str
    provisioning: str
    max_cycles: int | None = None
    stop_on_accept: bool = False

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Condition:
        return cls(
            name=str(d["name"]), accept_policy=str(d["accept_policy"]),
            provisioning=str(d["provisioning"]), max_cycles=d.get("max_cycles"),
            stop_on_accept=bool(d.get("stop_on_accept", False)),
        )


@dataclass(frozen=True, slots=True)
class Budgets:
    """The compute budget shared across the loop conditions (the §1.3 equal-compute control)."""

    max_cycles: int = 10
    sandbox_timeout_s: float = 1500.0
    # 2400 s is the measured calibration cap (results/ablation-ladder-calibration/REPORT.md §4):
    # ≈2× the heaviest clean gate run, bounding a deadlocked submission to 40 min.
    code_timeout_s: float = 2400.0
    recursion_limit: int = 200
    # The selection gate's noise-floor margin (audit C / V2): a submission must beat the incumbent
    # by MORE than this to be accepted, so scoring noise of magnitude ~ε can neither manufacture a
    # spurious "improvement" nor ratchet the bar past genuine progress. Shared across all conditions
    # (it's part of "the gate", not a per-condition treatment); `always` conditions ignore it. Set
    # to ~2–3× the measured ε (see docs/experimental-design.md §7) before a real sweep; 0.0 = off.
    selection_margin: float = 0.0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Budgets:
        return cls(
            max_cycles=int(d.get("max_cycles", 10)),
            sandbox_timeout_s=float(d.get("sandbox_timeout_s", 1500.0)),
            code_timeout_s=float(d.get("code_timeout_s", 2400.0)),
            recursion_limit=int(d.get("recursion_limit", 200)),
            selection_margin=float(d.get("selection_margin", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class SweepSpec:
    """The whole ablation matrix: conditions × models × seeds, with shared budgets."""

    task_type: str
    goal: str
    models: tuple[ModelArm, ...]
    conditions: tuple[Condition, ...]
    seeds: int
    budgets: Budgets

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SweepSpec:
        spec = cls(
            task_type=str(d.get("task_type", "fe-holdout")),
            goal=str(d.get("goal", "")),
            models=tuple(ModelArm.from_dict(m) for m in d["models"]),
            conditions=tuple(Condition.from_dict(c) for c in d["conditions"]),
            seeds=int(d.get("seeds", 1)),
            budgets=Budgets.from_dict(d.get("budgets", {})),
        )
        _validate_equal_compute(spec)
        return spec

    def cells(self) -> list[tuple[Condition, ModelArm, int]]:
        """Every ``(condition, model, seed)`` cell, in a stable order."""
        return [
            (cond, arm, seed)
            for cond in self.conditions
            for arm in self.models
            for seed in range(self.seeds)
        ]


def _validate_equal_compute(spec: SweepSpec) -> None:
    """Enforce the §1.3 equal-compute control: every **loop** condition (one that does not pin its
    own ``max_cycles``) runs the shared budget. A condition that overrides ``max_cycles`` is the
    deliberate single-shot baseline (Exp 1); the rest must be comparable, so we forbid silent
    per-condition cycle drift among the loop rungs."""
    loop_overrides = {
        c.name: c.max_cycles
        for c in spec.conditions
        if c.max_cycles is not None and c.max_cycles > 1
    }
    if loop_overrides:
        raise ValueError(
            "equal-compute control (experimental-design §1.3): loop conditions must share the "
            f"budget's max_cycles ({spec.budgets.max_cycles}); these pin a different multi-cycle "
            f"budget: {loop_overrides}. Move the shared value to budgets.max_cycles, or set 1 for "
            "the single-shot baseline."
        )


def load_spec(path: str | Path) -> SweepSpec:
    """Load + validate a sweep spec from a JSON file."""
    return SweepSpec.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_role_files(data_dir: str | Path) -> dict[str, dict[str, bytes]]:
    """Load the shared, pre-prepared dataset as ``{role: {filename: bytes}}`` (ADR 0005).

    Reads ``<data_dir>/agent/*.csv`` and ``<data_dir>/verifier/*.csv`` — the layout
    ``tools/prepare_fe_data.py`` writes. Prep is the user's responsibility; this only routes files.
    """
    root = Path(data_dir)
    roles: dict[str, dict[str, bytes]] = {}
    for role in ("agent", "verifier"):
        role_dir = root / role
        if role_dir.is_dir():
            roles[role] = {p.name: p.read_bytes() for p in sorted(role_dir.glob("*.csv"))}
    return roles


def build_task_request(
    spec: SweepSpec, condition: Condition, arm: ModelArm, seed: int
) -> TaskRequest:
    """Turn one cell into a declarative `TaskRequest` — a pure config diff over the shared task.

    The seed rides in ``sandbox.extra`` (forwarded to the OpenAI-compatible model path), so a
    seedable local model (Qwen) gives reproducible replicates; hosted models stay non-deterministic.
    """
    budgets = spec.budgets
    sandbox = SandboxRequest(
        model=arm.model, base_url=arm.base_url, api_key_env=arm.api_key_env,
        extra={"seed": seed},
        sandbox_timeout_s=budgets.sandbox_timeout_s, code_timeout_s=budgets.code_timeout_s,
        recursion_limit=budgets.recursion_limit,
    )
    return TaskRequest(
        type_name=spec.task_type,
        goal=spec.goal,
        sandbox=sandbox,
        verifier=VerifierRequest(
            knobs={
                "accept_policy": condition.accept_policy,
                # The noise-floor margin is shared across conditions (C/V2); `always` ignores it.
                "margin": budgets.selection_margin,
            }
        ),
        policy=PolicyRequest(
            max_cycles=condition.max_cycles or budgets.max_cycles,
            stop_on_accept=condition.stop_on_accept,
            provisioning=condition.provisioning,
        ),
        data=DataRequest(),
    )


@dataclass(slots=True)
class CellResult:
    """One cell's outcome: its coordinates, the run id, the run status, and the store-derived report
    (``None`` only if the run produced no record — a degraded run is still recorded)."""

    condition: str
    model: str
    seed: int
    run_id: str
    status: str
    report: RunReport | None

    def key(self) -> str:
        return f"{self.condition}-{self.model}-seed{self.seed}"


async def run_sweep(
    spec: SweepSpec,
    service: ControlService,
    *,
    role_files: RoleFiles,
    results_dir: str | Path | None = None,
) -> list[CellResult]:
    """Run every cell serially through ``service``; collect a :class:`CellResult` per cell.

    ``role_files`` is the shared, pre-prepared dataset (``{role: {filename: bytes}}``) ingested into
    each cell's task. When ``results_dir`` is set, each cell's `RunReport` JSON is written there as
    ``<condition>-<model>-seedN.json``, plus a ``manifest.json`` listing every cell + its coords —
    the robust file→cell mapping the analysis layer reads (no filename parsing).
    """
    out_dir = Path(results_dir) if results_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    files = {role: dict(named) for role, named in role_files.items()}

    results: list[CellResult] = []
    cells = spec.cells()
    for i, (condition, arm, seed) in enumerate(cells, start=1):
        request = build_task_request(spec, condition, arm, seed)
        log.info(
            "sweep_cell_start", index=i, total=len(cells),
            condition=condition.name, model=arm.label, seed=seed,
        )
        task_id = service.create_task(request, files=files)
        run_id = await service.run(task_id, goal=spec.goal or None)
        record = service.results(run_id)
        result = CellResult(
            condition=condition.name, model=arm.label, seed=seed, run_id=run_id,
            status=record.status if record is not None else "missing",
            report=record.report if record is not None else None,
        )
        results.append(result)
        if out_dir is not None and result.report is not None:
            (out_dir / f"{result.key()}.json").write_text(
                result.report.to_json(), encoding="utf-8"
            )
            await _harvest_cell(service, task_id, result, out_dir)
        log.info("sweep_cell_done", index=i, total=len(cells), key=result.key(),
                 status=result.status)
    if out_dir is not None:
        _write_manifest(out_dir, results)
    return results


async def _harvest_cell(
    service: ControlService, task_id: str, result: CellResult, out_dir: Path
) -> None:
    """Pull each cell's TRANSCRIPT(s) + accepted submission objects out of the durable store and
    write them in human-readable form next to the report — transcripts captured automatically, no
    manual extraction (the batch-1/2 loss lesson; needs the persisted ``--store-root``).

    Per cycle: ``transcript_ref`` is a content hash into the store (present even on rejected /
    no-proposal / timed-out cycles, F4); fetch the bytes, save the raw JSON, and render Markdown via
    ``tools/render_transcript.py``. Accepted artifacts' objects (``submission.py`` etc.) go under
    ``submissions/``. Best-effort and fully isolated: any failure here logs and is swallowed, so
    harvesting can NEVER fail a cell or abort a sweep. (Rejected-cell submission code is not a
    durable accepted artifact, but it is visible in that cell's transcript.)
    """
    report = result.report
    if report is None:
        return
    multi = len(report.cycles) > 1
    for cyc in report.cycles:
        ref = cyc.transcript_ref
        if not ref:
            continue
        try:
            raw = service.get_object(task_id, ref)
        except Exception as exc:  # noqa: BLE001 - best-effort observability, never fatal
            log.warning("harvest_transcript_miss", key=result.key(), ref=ref, error=str(exc))
            continue
        stem = f"{result.key()}-cycle{cyc.index}" if multi else result.key()
        tdir = out_dir / "transcripts"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / f"{stem}.json").write_bytes(raw)
        try:
            from tools.render_transcript import render  # type: ignore[import-not-found]

            entries = json.loads(raw)
            (tdir / f"{stem}.md").write_text(render(entries, limit=None), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - rendering is a convenience; raw JSON is the source
            log.warning("harvest_render_failed", key=result.key(), error=str(exc))

    try:
        accepted_ids = [a["id"] for a in report.summary.accepted]
        by_id = {a.id: a for a in await service.accepted_artifacts(task_id)}
        for art_id in accepted_ids:
            art = by_id.get(art_id)
            if art is None:
                continue
            for name, oref in art.objects:
                dest = out_dir / "submissions" / result.key() / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(service.get_object(task_id, oref.content_hash))
    except Exception as exc:  # noqa: BLE001 - best-effort; transcripts already carry the narrative
        log.warning("harvest_submissions_failed", key=result.key(), error=str(exc))


def _write_manifest(out_dir: Path, results: list[CellResult]) -> None:
    """Write ``manifest.json`` — the cell coordinates + report filename the analysis layer reads.

    Only cells that produced a report file are listed (a missing-report cell has nothing to read).
    The ``report`` filename matches what ``run_sweep`` wrote, so the reader never parses keys.
    """
    manifest = [
        {
            "condition": r.condition, "model": r.model, "seed": r.seed,
            "run_id": r.run_id, "status": r.status, "report": f"{r.key()}.json",
        }
        for r in results
        if r.report is not None
    ]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run a real sweep: ``python -m experiments.ablation.sweep SPEC --data DIR --out DIR``.

    Wires a real `ControlService` (the discovered catalog over a `DockerBackend`, launching sibling
    worker containers — the same engine the daemon serves), so this needs Docker and the worker
    images. The heavy imports are local so importing this module for the tested core stays light.

    PREFER ``experiments/ablation/run_sweep_container.sh`` over invoking this directly: this *is*
    the control plane, so it must run inside a container on ``verity-net`` to reach the §9.1
    verifier sibling (a host process can't resolve its Docker-network name → every cell aborts).
    See the package README's "Running it" box.
    """
    import argparse

    from verity.logging import configure_logging
    from verity.service.daemon import build_service

    parser = argparse.ArgumentParser(description="Run the ablation ladder (Exp 1-4b) sweep.")
    parser.add_argument("spec", help="path to the JSON sweep spec (see spec.example.json)")
    parser.add_argument("--data", required=True, help="prepared data dir (agent/ + verifier/ CSVs)")
    parser.add_argument("--out", required=True, help="dir to write per-cell RunReport JSON into")
    parser.add_argument(
        "--store-root", default=None,
        help="durable store root. OMITTING THIS USES AN IN-MEMORY STORE that dies with the "
        "process: the RunReport JSONs survive (scores/rationale/telemetry → analyze.py + all "
        "figures), but the agent TRANSCRIPTS and submission objects (submission.py, "
        "requirements.txt) are LOST (transcript_ref is only a hash into the dead store). Pass a "
        "path to keep them.",
    )
    args = parser.parse_args(argv)

    configure_logging(json_output=True)
    if not args.store_root:
        # Loud, not fatal — an intentional throwaway run is legitimate, but a forgotten
        # --store-root on a multi-hour sweep silently discards every transcript + submission. Make
        # that impossible to miss (the lesson from the batch-1/2 transcript loss).
        log.warning(
            "sweep_ephemeral_store_no_store_root",
            impact="transcripts + submission objects WILL NOT be saved; only RunReports persist",
            fix="pass --store-root <dir>, or use run_sweep_container.sh which sets it",
        )
    spec = load_spec(args.spec)
    role_files = load_role_files(args.data)
    service = build_service(root=Path(args.store_root) if args.store_root else None)
    results = asyncio.run(run_sweep(spec, service, role_files=role_files, results_dir=args.out))
    completed = sum(1 for r in results if r.status == "complete")
    log.info("sweep_done", cells=len(results), completed=completed, out=args.out)
    return 0 if completed == len(results) else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper over the tested run_sweep core
    raise SystemExit(main())
