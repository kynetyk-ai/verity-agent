"""The long-lived control-plane service (ROADMAP Phase 8, ADR 0004).

This package is the transport-agnostic core of the standing control-plane service: a
`ControlService` that multiplexes a generic `ControlPlane` + store per task instance, over a
`TaskCatalog` of pre-built task types and a durable `TaskIndex`. The daemon, the Unix-socket IPC,
the `verity` CLI (8.2), and the external HTTP API (8.4) are thin adapters over this core.

It imports `verity.composition` (the catalog, which knows domains) but **no domain directly** — the
genericity invariant holds: `verity.control_plane` stays task-agnostic, and the service drives it
only through its public API.
"""

from __future__ import annotations

from verity.service.control_service import ControlService, UnknownTask
from verity.service.task_index import TaskIndex

__all__ = ["ControlService", "UnknownTask", "TaskIndex"]
