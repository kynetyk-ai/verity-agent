"""The composition root (ROADMAP 7.4): apply a task to a generic control plane, declaratively.

This is the one layer allowed to import across services — control plane, sandbox, verifier,
provisioning, and domains — to configure a task onto an already-constructed, **task-agnostic**
`ControlPlane`. A caller builds a generic control plane, then applies a task through the CP's own
API (`configure_fe_kaggle_task` / `configure_code_task` register the task's providers + seed data +
`configure`). The control plane is never built around a task; `verity.control_plane` imports
nothing from `verity.domains` (enforced by a layering test), so the CP stays generic and long-lived.
"""

from __future__ import annotations

from verity.composition.catalog import (
    CatalogBuilder,
    Describer,
    TaskCatalog,
    UnknownTaskType,
    default_catalog,
)
from verity.composition.code import (
    CODE_GOAL,
    CODE_TASK_ID,
    build_code_task,
    configure_code_task,
    describe_code_task,
)
from verity.composition.description import OperationDescription, TaskTypeDescription
from verity.composition.fe import ProvisioningConfig, provisioning_config_from
from verity.composition.loader import TASK_TYPES_GROUP, load_task_plugins
from verity.composition.task_request import (
    DataRequest,
    PolicyRequest,
    SandboxRequest,
    TaskRequest,
    VerifierRequest,
)

__all__ = [
    "ProvisioningConfig",
    "provisioning_config_from",
    "configure_code_task",
    "build_code_task",
    "describe_code_task",
    "CODE_GOAL",
    "CODE_TASK_ID",
    # the catalog + declarative request (ROADMAP 8.1)
    "TaskCatalog",
    "CatalogBuilder",
    "Describer",
    "UnknownTaskType",
    "default_catalog",
    "load_task_plugins",
    "TASK_TYPES_GROUP",
    "TaskRequest",
    "SandboxRequest",
    "VerifierRequest",
    "PolicyRequest",
    "DataRequest",
    # task-type self-description (ROADMAP 8.2)
    "TaskTypeDescription",
    "OperationDescription",
]
