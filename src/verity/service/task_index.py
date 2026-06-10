"""The daemon-level task index (ROADMAP 8.1, ADR 0004 (b)/(h)): durable task definitions.

A task's per-task provenance store is created *from* its `TaskRequest`, so the request cannot live
only inside that store at create time (the chicken-and-egg). The :class:`TaskIndex` is the small
registry that resolves it: ``task_id -> TaskRequest``, persisted at the service **root** (not in any
per-task store), so a created task is durable and **rehydratable** after a restart — the daemon
re-reads the request and rebuilds the task's control plane on demand.

The default writes one JSON file per task at ``<root>/tasks/<task_id>/request.json``; the per-task
`SqliteStore` (``store.db`` + ``objects/``) lives alongside under the same directory, so a task's
definition and its provenance are co-located for restart discovery. With ``root=None`` the index is
in-memory (the test posture, mirroring `SqliteStore(":memory:")` — no restart, by definition).
"""

from __future__ import annotations

import json
from pathlib import Path

from verity.composition.task_request import TaskRequest

__all__ = ["TaskIndex"]


class TaskIndex:
    """A durable ``task_id -> TaskRequest`` registry (JSON files under root; in-memory if None)."""

    def __init__(self, root: Path | None) -> None:
        self._root = root
        self._mem: dict[str, TaskRequest] = {}

    def task_dir(self, task_id: str) -> Path:
        """The per-task directory (request + store + objects live here). Requires a disk root."""
        if self._root is None:
            raise ValueError("no on-disk root: this index is in-memory")
        return self._root / "tasks" / task_id

    def put(self, task_id: str, request: TaskRequest) -> None:
        if self._root is None:
            self._mem[task_id] = request
            return
        path = self.task_dir(task_id) / "request.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(request.to_dict(), indent=2), encoding="utf-8")

    def get(self, task_id: str) -> TaskRequest | None:
        if self._root is None:
            return self._mem.get(task_id)
        path = self.task_dir(task_id) / "request.json"
        if not path.exists():
            return None
        return TaskRequest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[str]:
        if self._root is None:
            return sorted(self._mem)
        tasks_dir = self._root / "tasks"
        if not tasks_dir.exists():
            return []
        return sorted(
            d.name for d in tasks_dir.iterdir() if (d / "request.json").exists()
        )
