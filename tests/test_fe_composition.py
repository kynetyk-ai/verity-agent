"""The FE composition builder (ROADMAP 7.4.f). Offline — `FakeBackend`, no Docker, no model.

Proves the ad-hoc `_build_fe_task` wiring is replaced by one declarative call:
`build_fe_control_plane` seeds the dataset, registers the FE sandbox + verifier providers, and
produces a `TaskConfig` the control plane configures — no hand-built drivers/runners/registries.
The full FE run is the 7.4.g capstone; here we prove the wiring resolves end to end.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from verity.composition import ProvisioningConfig, build_fe_control_plane
from verity.control_plane.registries import ObjectProvisionMode
from verity.control_plane.store import SqliteStore
from verity.provisioning import FakeBackend


@dataclass
class _Split:
    agent_train_csv: bytes = b"id,a,class\n1,2,X\n3,4,Y\n"
    reserved_test_csv: bytes = b"id,a\n5,6\n"
    reserved_labels: dict[str, str] = field(default_factory=lambda: {"5": "X"})


def test_build_fe_control_plane_wires_and_configures() -> None:
    store = SqliteStore()
    cp, config = build_fe_control_plane(backend=FakeBackend(), split=_Split(), store=store)

    # the declarative config — provisioning never leaked onto TaskConfig
    assert config.task_id == "fe"
    assert config.sandbox_key == "fe" and config.verifier_key == "fe"
    assert config.harvester is not None  # the FE Feature-harvest hook
    assert config.object_provisioning.mode is ObjectProvisionMode.LAST_REVISED_OR_ACCEPTED
    assert not hasattr(config, "backend_key")  # provisioning is NOT on the task config

    # the dataset root was seeded
    root = store.get_artifact("ds")
    assert root is not None and root.is_root

    # the providers resolve + provision: configure() succeeds end to end (the wiring is complete)
    asyncio.run(cp.configure(config))


def test_provisioning_config_carries_substrate_shape_not_task_config() -> None:
    cfg = ProvisioningConfig(runtime="runsc", sandbox_memory="8g")
    assert cfg.runtime == "runsc" and cfg.sandbox_memory == "8g"
    # the model defaults to frontier Anthropic when unset
    assert cfg.model_spec is None
