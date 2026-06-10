"""Live cross-model benchmark (ROADMAP Phase 6.3) — the thesis instrument.

Runs the **same task** through a frontier Anthropic arm and a host-local OpenAI-compatible arm, then
prints the quality / cost / latency comparison from the eval harness. **Manual + live** (real
models, Docker, the image) — not a test, not run in CI.

Two domains:

- ``--domain code`` (default): the trivial "write submission.py that prints ok" task with a
  ``FakeCodeRunner`` gate (boolean parses/runs) — fast, validates the pipeline, no score.
- ``--domain feature-engineering``: the real §12 feature-engineering task — the agent engineers
  features over the stellar dataset, the verifier **trains + scores** each script in a container
  (``balanced_accuracy``), over multiple refine cycles. This is the *quality* benchmark: the
  ``best_score`` column is a real number, so frontier vs local actually differentiate.

Usage (from the repo root, sandbox extra + the image built)::

    uv run --extra sandbox python -m tools.benchmark_models \
        --domain feature-engineering --max-cycles 4 --per-class 300 \
        --local-model qwen3.6:27b-coding-mxfp8 \
        --local-base-url http://host.docker.internal:11434/v1

Pass ``--no-anthropic`` to skip the frontier arm, etc.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import os
import tempfile
from pathlib import Path

from verity.composition import FE_GOAL, ProvisioningConfig, build_fe_control_plane
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
from verity.domains.code import DATASET, build_code_domain
from verity.eval import BenchmarkArm, ConfiguredTask, ModelPrice, run_benchmark
from verity.provisioning import DockerBackend
from verity.sandbox.model_spec import ModelSpec
from verity.sandbox.providers import local_spec
from verity.sandbox.registration import build_container_sandbox
from verity.verifier import FakeCodeRunner, RunResult

_IMAGE = "verity-sandbox:latest"

# ---------------------------------------------------------------- code domain (fast, boolean gate)

_CODE_GOAL = "submit a feature as code, using dataset ds"


async def _build_code_task(label: str, model_spec: ModelSpec) -> ConfiguredTask:
    domain = build_code_domain(FakeCodeRunner(script=lambda _r: RunResult(0, "", "")))
    store = SqliteStore()
    store.propose(
        Artifact("ds", DATASET, {"n": 10}, ArtifactStatus.PROPOSED, "loader", "t0", is_root=True),
        Operation("op-ds", "load", (), "ds", OperationStatus.SUCCESS, "t0"),
    )
    root = Path(tempfile.mkdtemp(prefix=f"verity-bench-{label}-"))
    sandbox = build_container_sandbox(
        schema=domain.schema, root=root, model="unused", model_spec=model_spec,
        image=_IMAGE, id_source=lambda: f"sub-{label}",
    )
    sp: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
    vp: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    sp.register("container", lambda: sandbox)
    vp.register("real", lambda: domain.verifier)
    cp = ControlPlane(
        store, policy=OrchestrationPolicy(stop_on_accept=True),
        sandbox_providers=sp, verifier_providers=vp,
    )
    config = TaskConfig(
        task_id="t1",
        instructions=(
            "Write submission.py to outbox/ that prints 'ok' (run it to check), then submit it "
            "as a Submission with entrypoint submission.py, using dataset id 'ds' as the parent."
        ),
        domain_instructions="a Submission names an entrypoint script written to outbox/",
        schema=domain.schema, gated_types=domain.gated_types, retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator, sandbox_key="container", verifier_key="real",
    )
    await cp.configure(config)
    return ConfiguredTask(cp, task_id="t1", goal=_CODE_GOAL, model_name=model_spec.model)


# ------------------------------------------------- feature-engineering domain (real trained score)
#
# The goal + instructions now live with the declarative wiring in ``verity.composition.fe``; this
# module only subsamples the dataset and hands a split to ``build_fe_control_plane``.


def _subsample(data: bytes, *, per_class: int, target: str = "class") -> bytes:
    """Keep up to ``per_class`` rows per target class — a fast, stratified slice for a live run."""
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


async def _build_fe_task(
    label: str, model_spec: ModelSpec, split: object, *, max_cycles: int
) -> ConfiguredTask:
    # Declarative wiring (ROADMAP 7.4.f): one builder seeds the dataset, registers the FE
    # sandbox/verifier providers (backend-backed workers — the sandbox writes the script, the
    # verifier runs it in an isolated worker), and assembles the TaskConfig. No hand-built wiring.
    cp, config = build_fe_control_plane(
        backend=DockerBackend(),
        split=split,
        provisioning=ProvisioningConfig(model_spec=model_spec, sandbox_image=_IMAGE),
        max_cycles=max_cycles,
    )
    await cp.configure(config)
    return ConfiguredTask(cp, task_id="fe", goal=FE_GOAL, model_name=model_spec.model)


# --------------------------------------------------------------------------------- arms + runner


async def _build_task(
    label: str, model_spec: ModelSpec, args: argparse.Namespace, split: object
) -> ConfiguredTask:
    if args.domain == "feature-engineering":
        return await _build_fe_task(label, model_spec, split, max_cycles=args.max_cycles)
    return await _build_code_task(label, model_spec)


def _load_split(args: argparse.Namespace) -> object:
    if args.domain != "feature-engineering":
        return None
    from tools.harness.dataset import stratified_split

    raw = _subsample(Path(args.dataset).read_bytes(), per_class=args.per_class)
    return stratified_split(raw, target="class", id_column="id",
                            reserved_fraction=args.reserved_fraction)


def _arms(args: argparse.Namespace, split: object) -> list[BenchmarkArm]:
    arms: list[BenchmarkArm] = []
    if not args.no_anthropic:
        a_spec = ModelSpec.from_provider_string(args.anthropic_model)
        # Bind the spec per-arm (default arg) so the lambda doesn't capture a later-rebound spec.
        arms.append(
            BenchmarkArm("anthropic", lambda s=a_spec: _build_task("anthropic", s, args, split))
        )
    if args.local_model:
        l_spec = local_spec(args.local_model, base_url=args.local_base_url)
        arms.append(
            BenchmarkArm(
                "local", lambda s=l_spec: _build_task("local", s, args, split),
                price=ModelPrice(0.0, 0.0),
            )
        )
    return arms


async def _amain(args: argparse.Namespace) -> None:
    split = _load_split(args)
    arms = _arms(args, split)
    if not arms:
        raise SystemExit("no arms selected — pass --local-model and/or drop --no-anthropic")
    comparison = await run_benchmark(arms)
    print(comparison.render_text())
    print()
    print(comparison.to_json())


def main() -> None:
    parser = argparse.ArgumentParser(description="Live cross-model benchmark (Phase 6.3).")
    parser.add_argument("--domain", choices=("code", "feature-engineering"), default="code")
    parser.add_argument("--max-cycles", type=int, default=4, help="FE: refine cycles per arm")
    parser.add_argument("--per-class", type=int, default=300, help="FE: rows per class subsample")
    parser.add_argument("--reserved-fraction", type=float, default=0.5, help="FE held-out fraction")
    parser.add_argument("--dataset", default="feature-engineering-test/train.csv")
    parser.add_argument("--anthropic-model", default="anthropic:claude-sonnet-4-6")
    parser.add_argument("--no-anthropic", action="store_true", help="skip the Anthropic arm")
    parser.add_argument("--local-model", default=os.environ.get("VERITY_LOCAL_MODEL"))
    parser.add_argument(
        "--local-base-url",
        default=os.environ.get("VERITY_LOCAL_MODEL_BASE_URL", "http://host.docker.internal:8000/v1"),
    )
    args = parser.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
