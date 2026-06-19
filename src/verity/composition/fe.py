"""Shared feature-engineering provisioning helpers for the composition layer.

The substrate-and-worker shape (`ProvisioningConfig`) and the request -> substrate mapping
(`provisioning_config_from`) live here, kept deliberately **off** `TaskConfig` so the control plane
stays task-agnostic (ADR 0003). The FE-Kaggle task (`composition.fe_kaggle`) and the trivial `code`
task (`composition.code`) both import these. There is no standalone basic-`fe` task type;
`fe-kaggle` is the feature-engineering task.
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.composition.task_request import SandboxRequest
from verity.sandbox.model_spec import ModelSpec

__all__ = [
    "ProvisioningConfig",
    "provisioning_config_from",
]


@dataclass(frozen=True, slots=True)
class ProvisioningConfig:
    """The substrate-and-worker shape, kept OFF `TaskConfig` (the control plane stays agnostic).

    ``model_spec`` is the sandbox's model target (default frontier Anthropic). ``runtime`` selects
    the OCI runtime per worker (e.g. ``"runsc"`` for gVisor; ``None`` = the daemon default).
    """

    sandbox_image: str = "verity-sandbox:latest"
    code_image: str = "verity-code-runner:latest"
    model_spec: ModelSpec | None = None
    runtime: str | None = None
    sandbox_memory: str = "4g"
    code_memory: str = "2g"
    code_tmpfs_size: str = "1g"
    recursion_limit: int = 200
    sandbox_timeout_s: float = 1500.0
    code_timeout_s: float = 600.0


def provisioning_config_from(sandbox: SandboxRequest) -> ProvisioningConfig:
    """Reconstruct the (off-`TaskConfig`) `ProvisioningConfig` from a declarative `SandboxRequest`.

    Shared by the FE and `code` catalog builders so the request -> substrate mapping lives in one
    place. The control plane never sees this; it stays on the provisioning side (ADR 0003).
    """
    return ProvisioningConfig(
        sandbox_image=sandbox.sandbox_image,
        code_image=sandbox.code_image,
        model_spec=sandbox.to_model_spec(),
        runtime=sandbox.runtime,
        sandbox_memory=sandbox.sandbox_memory,
        code_memory=sandbox.code_memory,
        code_tmpfs_size=sandbox.code_tmpfs_size,
        recursion_limit=sandbox.recursion_limit,
        sandbox_timeout_s=sandbox.sandbox_timeout_s,
        code_timeout_s=sandbox.code_timeout_s,
    )
