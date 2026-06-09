"""Live cross-model benchmark (ROADMAP Phase 6.3) — the thesis instrument.

Runs the **same** code-submission task through a frontier Anthropic arm and a host-local
OpenAI-compatible arm, then prints the quality / cost / latency comparison from the eval harness.
This is a **manual, live** entrypoint (real models, Docker, the sandbox image) — not a test, and it
does not run in CI.

Usage (from the repo root, with the sandbox extra and the image built)::

    # frontier baseline needs ANTHROPIC_API_KEY; the local arm needs a running OpenAI-compatible
    # server on the host (see docs/local-models.md) and its served model name:
    uv run --extra sandbox python -m tools.benchmark_models \
        --local-model qwen2.5-coder --local-base-url http://host.docker.internal:8000/v1

Each arm is optional: pass ``--no-anthropic`` to benchmark only the local model, etc.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
from pathlib import Path

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
from verity.sandbox.model_spec import ModelSpec
from verity.sandbox.providers import local_spec
from verity.sandbox.registration import build_container_sandbox
from verity.verifier import FakeCodeRunner, RunResult

_GOAL = "submit a feature as code, using dataset ds"
_IMAGE = "verity-sandbox:latest"


async def _build_task(label: str, model_spec: ModelSpec) -> ConfiguredTask:
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
    return ConfiguredTask(cp, task_id="t1", goal=_GOAL, model_name=model_spec.model)


def _arms(args: argparse.Namespace) -> list[BenchmarkArm]:
    arms: list[BenchmarkArm] = []
    if not args.no_anthropic:
        spec = ModelSpec.from_provider_string(args.anthropic_model)
        arms.append(BenchmarkArm("anthropic", lambda: _build_task("anthropic", spec)))
    if args.local_model:
        spec = local_spec(args.local_model, base_url=args.local_base_url)
        arms.append(
            BenchmarkArm("local", lambda: _build_task("local", spec), price=ModelPrice(0.0, 0.0))
        )
    return arms


async def _amain(args: argparse.Namespace) -> None:
    arms = _arms(args)
    if not arms:
        raise SystemExit("no arms selected — pass --local-model and/or drop --no-anthropic")
    comparison = await run_benchmark(arms)
    print(comparison.render_text())
    print()
    print(comparison.to_json())


def main() -> None:
    parser = argparse.ArgumentParser(description="Live cross-model benchmark (Phase 6.3).")
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
