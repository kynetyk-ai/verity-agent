"""The run-record store (ROADMAP Phase 7 — multi-tenancy & run-control, seams-first).

A :class:`RunRecord` is the *operational* metadata around a run — its status, its ``RunReport``, and
pointers to the accepted artifacts — keyed by ``(tenant_id, run_id)``. It references the
typed-provenance store, never copies it: the artifact ids point back at the system of record.
The :class:`RunRecordStore` is a pure seam with an in-memory default; a SQLite/Postgres-backed store
(mirroring the artifact store) is a later adapter swap that the multi-tenancy engine forces.

The control plane does not write these in the loop yet (the loop stays generic); a caller builds one
from ``cp.run_report(task_id)`` at run end. Wiring it into a standing job server is the engine step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from verity.control_plane.run_report import RunReport

__all__ = ["RunRecord", "RunRecordStore", "InMemoryRunRecordStore"]


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Operational metadata for one **run**, identified by ``(tenant_id, run_id)``.

    ``task_id`` records **which task (configuration)** this run executed — a run is one *execution*
    of a task, so task and run are distinct identities (see ``docs/glossary.md``). ``run_id`` is the
    run's own id; ``status`` is a free string (``"complete"`` / ``"aborted"`` / …) the consumer
    reads; ``accepted_artifact_ids`` point into the provenance store (the audit record), not a copy.
    """

    tenant_id: str
    task_id: str
    run_id: str
    status: str
    report: RunReport
    accepted_artifact_ids: tuple[str, ...] = ()

    @classmethod
    def from_report(cls, report: RunReport, *, run_id: str, status: str) -> RunRecord:
        """Build from a ``RunReport``. ``run_id`` is **required**: a run's identity is distinct from
        the task it ran (task != run), so the caller mints it, never reusing the ``task_id``."""
        return cls(
            tenant_id=report.tenant_id,
            task_id=report.task_id,
            run_id=run_id,
            status=status,
            report=report,
            accepted_artifact_ids=tuple(a["id"] for a in report.summary.accepted),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "status": self.status,
            "accepted_artifact_ids": list(self.accepted_artifact_ids),
            "report": self.report.to_dict(),
        }


@runtime_checkable
class RunRecordStore(Protocol):
    """Put / get / list run records, scoped by tenant (the run-results read side).

    ``list`` optionally filters by ``task_id`` — "which runs belong to this task" — since a task may
    be run more than once (task : run is 1 : many).
    """

    def put(self, record: RunRecord) -> None: ...
    def get(self, tenant_id: str, run_id: str) -> RunRecord | None: ...
    def list(self, tenant_id: str, *, task_id: str | None = None) -> list[RunRecord]: ...


class InMemoryRunRecordStore:
    """The default: a dict keyed by ``(tenant_id, run_id)``, tenant-namespaced (no cross reads)."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], RunRecord] = {}

    def put(self, record: RunRecord) -> None:
        self._records[(record.tenant_id, record.run_id)] = record

    def get(self, tenant_id: str, run_id: str) -> RunRecord | None:
        return self._records.get((tenant_id, run_id))

    def list(self, tenant_id: str, *, task_id: str | None = None) -> list[RunRecord]:
        return [
            r for r in self._records.values()
            if r.tenant_id == tenant_id and (task_id is None or r.task_id == task_id)
        ]
