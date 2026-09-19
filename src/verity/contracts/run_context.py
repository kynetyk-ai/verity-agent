"""Run identity the control plane stamps onto the work a cycle provisions (ROADMAP 7.4.h, ADR 0003).

A :class:`RunContext` is the neutral ``(tenant_id, run_id, cycle)`` triple that names *which run, on
whose behalf, on which cycle* a unit of work belongs to. The control plane owns it — a run's id is
minted per ``run()``, the cycle counter advances per cycle — and pushes it to any service that
advertises :class:`SupportsRunContext`. A backend-backed sandbox driver / code-runner reads it when
it builds a worker spec, so every ephemeral worker is **labelled** for audit and garbage collection
(reap-by-run, ADR e); a plain in-process stub simply does not implement the capability and is
untouched.

It lives in ``contracts`` because it is shared vocabulary, owned by neither side: the control plane
sets it, the sandbox / verifier read it, and it carries no provisioning detail (mapping it onto
container *labels* is the backend's private business). The object is **mutable by design** — the
control plane binds the cell once and then advances ``cycle`` (and sets ``run_id`` per run) in
place, so a bound service always sees the current values without a per-cycle handshake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["RunContext", "SupportsRunContext"]


@dataclass(slots=True)
class RunContext:
    """The ``(tenant_id, run_id, cycle)`` identity of the work a cycle provisions.

    Mutable on purpose: the control plane binds one cell per task, then advances ``cycle`` and sets
    ``run_id`` in place across the run, so every bound service reads the current values lazily.
    """

    tenant_id: str = "default"
    run_id: str = ""
    cycle: int = 0


@runtime_checkable
class SupportsRunContext(Protocol):
    """A service that stamps run identity onto whatever it provisions (e.g. worker labels).

    The control plane binds the shared :class:`RunContext` once (it is duck-typed via
    ``isinstance`` — a service that does not implement this is simply skipped). A composite service
    forwards the binding to the resource that actually provisions (a verifier to its code-runner, a
    sandbox to its driver).
    """

    def bind_run_context(self, ctx: RunContext) -> None: ...
