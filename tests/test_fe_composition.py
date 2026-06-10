"""Applying the FE task to a *generic* control plane (ROADMAP 7.4, decoupled). Offline.

Proves the control plane is **not** built around FE: a plain, task-agnostic `ControlPlane(store)` is
constructed first (it has no providers, knows no task), and `configure_fe_task` applies FE onto it
*through the CP's API* — registering the FE providers, seeding the dataset, and configuring. The
mechanical genericity guarantee (the CP package importing nothing from `domains`) is in
`test_control_plane_genericity.py`; here we prove the application path resolves end to end.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from verity.composition import ProvisioningConfig, configure_fe_task
from verity.control_plane.api import ControlPlane
from verity.control_plane.store import SqliteStore
from verity.provisioning import FakeBackend


@dataclass
class _Split:
    agent_train_csv: bytes = b"id,a,class\n1,2,X\n3,4,Y\n"
    reserved_test_csv: bytes = b"id,a\n5,6\n"
    reserved_labels: dict[str, str] = field(default_factory=lambda: {"5": "X"})


def test_configure_fe_task_applies_to_a_generic_control_plane() -> None:
    store = SqliteStore()
    cp = ControlPlane(store)  # GENERIC: no providers, no task knowledge at construction

    task_id = asyncio.run(configure_fe_task(cp, backend=FakeBackend(), split=_Split()))

    assert task_id == "fe"
    # the dataset root was seeded into the CP's own store via its API
    root = store.get_artifact("ds")
    assert root is not None and root.is_root
    # the task is fully configured end to end: configure() registered the schema version, and the
    # FE provider is now resolvable under its key (registered through the CP's API, not at build)
    assert store.current_schema_version() is not None
    assert cp._sandbox_providers.create("fe") is not None  # noqa: SLF001 - test inspects the registry


def test_provisioning_config_carries_substrate_shape_not_task_config() -> None:
    cfg = ProvisioningConfig(runtime="runsc", sandbox_memory="8g")
    assert cfg.runtime == "runsc" and cfg.sandbox_memory == "8g"
    # the model defaults to frontier Anthropic when unset
    assert cfg.model_spec is None
