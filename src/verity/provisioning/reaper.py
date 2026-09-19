"""The label-reaper (ROADMAP 7.4.h / ADR 0003 §e): reclaim orphaned workers by label.

Every worker the harness launches is stamped with ``{harness,tenant,run,cycle,role,config}`` labels
(7.4.h). That makes garbage collection a *query*: an out-of-band reaper selects workers by label and
destroys them, with no knowledge of the run that made them. This is the death-switch for the cases
the per-cycle ``launch -> wait -> destroy`` lifetime cannot cover — a control plane that crashed
mid-cycle, an aborted run, a developer's stray fleet.

It lives in ``provisioning`` (neutral infra) and talks only to a `WorkerBackend`, so it reaps on
whatever substrate the workers ran on (Docker now, k8s later). The control plane never calls it — it
stays provisioning-agnostic; the reaper is a deployment-side tool, exposed as the ``verity-reaper``
console entrypoint. The ``harness=verity`` floor is always in the selector, so it can never touch a
container this harness did not create.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping

from verity.logging import get_logger
from verity.provisioning.backend import ProvisioningError, WorkerBackend
from verity.provisioning.docker import DockerBackend

__all__ = ["build_selector", "reap", "main"]

log = get_logger("verity.provisioning.reaper")


def build_selector(
    *,
    tenant: str | None = None,
    run: str | None = None,
    role: str | None = None,
    harness: str = "verity",
) -> dict[str, str]:
    """A label selector for the reaper. ``harness`` is always present (the floor), so a reap is
    scoped to this harness's workers even when no other facet is given (the broad GC sweep)."""
    selector = {"harness": harness}
    if tenant:
        selector["tenant"] = tenant
    if run:
        selector["run"] = run
    if role:
        selector["role"] = role
    return selector


async def reap(backend: WorkerBackend, selector: Mapping[str, str]) -> int:
    """Destroy every worker matching ``selector``; return how many were reaped."""
    count = await backend.reap(selector)
    log.info("workers_reaped", selector=dict(selector), count=count)
    return count


def main(argv: list[str] | None = None) -> int:
    """The ``verity-reaper`` entrypoint: reap orphaned workers on the local Docker backend."""
    parser = argparse.ArgumentParser(
        prog="verity-reaper",
        description="Reap orphaned verity workers by label (tenant / run / role).",
    )
    parser.add_argument("--tenant", help="reap only this tenant's workers")
    parser.add_argument("--run", help="reap only this run's workers")
    parser.add_argument("--role", help="reap only this role (sandbox / code-runner / verifier)")
    parser.add_argument("--harness", default="verity", help="the harness label floor")
    args = parser.parse_args(argv)

    selector = build_selector(
        tenant=args.tenant, run=args.run, role=args.role, harness=args.harness
    )
    backend = DockerBackend()
    try:
        count = asyncio.run(reap(backend, selector))
    except ProvisioningError as exc:
        print(f"verity-reaper: could not reap ({exc})")
        return 1
    print(f"verity-reaper: reaped {count} worker(s) matching {selector}")
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI shim
    raise SystemExit(main())
