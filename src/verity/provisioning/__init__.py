"""The provisioning layer (ROADMAP 7.4 / ADR 0003): launch isolated ephemeral workers.

A **neutral** package: it owns the `WorkerBackend` seam (launch / observe / destroy one labelled,
isolated unit of work) and nothing else. It imports nothing from `sandbox`, `verifier`,
`control_plane`, or `domains` — and the control plane never imports it. Provisioning lives *behind*
the `SandboxPort`/`VerifierPort`, so the control plane stays agnostic to how workers run or where
(local Docker now, Kubernetes later — a second backend, same port). See ADR 0003.
"""

from __future__ import annotations

from verity.provisioning.backend import (
    CompletedWorker,
    Labels,
    ResourceLimits,
    Tmpfs,
    WorkerBackend,
    WorkerHandle,
    WorkerSpec,
    WorkerStatus,
)
from verity.provisioning.fake import FakeBackend, WorkerScript

__all__ = [
    "WorkerBackend",
    "WorkerSpec",
    "WorkerHandle",
    "WorkerStatus",
    "CompletedWorker",
    "Labels",
    "ResourceLimits",
    "Tmpfs",
    "FakeBackend",
    "WorkerScript",
]
