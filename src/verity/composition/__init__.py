"""The composition root (ROADMAP 7.4.f): wire a task's services declaratively.

This is the one layer allowed to import across services — control plane, sandbox, verifier,
provisioning, and domains — to assemble a runnable `ControlPlane` from declarative inputs (a domain,
a dataset, a `ProvisioningConfig` naming the backend + worker shape). It replaces the hand-wiring
that used to live in ``tools/benchmark_models.py`` and the live tests: a caller picks a backend and
calls a builder, instead of instantiating drivers / code-runners / registries by hand. The control
plane and `TaskConfig` stay provisioning-agnostic; everything substrate-specific lives here.
"""

from __future__ import annotations

from verity.composition.fe import FE_GOAL, ProvisioningConfig, build_fe_control_plane

__all__ = ["ProvisioningConfig", "build_fe_control_plane", "FE_GOAL"]
