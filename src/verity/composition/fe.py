"""Shared feature-engineering provisioning helpers for the composition layer.

The substrate-and-worker shape (`ProvisioningConfig`) and the request -> substrate mapping
(`provisioning_config_from`) live here, kept deliberately **off** `TaskConfig` so the control plane
stays task-agnostic (ADR 0003). The FE-Kaggle task (`composition.fe_kaggle`) and the trivial `code`
task (`composition.code`) both import these. There is no standalone basic-`fe` task type;
`fe-kaggle` is the feature-engineering task.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace

from verity.composition.task_request import SandboxRequest
from verity.sandbox.model_spec import ModelSpec

__all__ = [
    "ProvisioningConfig",
    "provisioning_config_from",
    "ANSWER_KEY_FILE",
    "assert_answer_key_isolated",
    "BASE_SANDBOX_IMAGE",
    "FE_SANDBOX_IMAGE",
    "with_fe_sandbox_image",
]

# The generic agent-runtime image (agent loop only) vs the feature-engineering image (base + the
# SYSTEM shared libraries the CPU ML wheels link, e.g. libgomp). FE agents install their own pinned
# requirements (mirroring the gate's clean install) and self-test their script — the base image
# can't (the wheels import-fail without the system libs). So an FE task defaults to the FE image;
# the Python ML packages are NOT baked in, on purpose (see Dockerfile.fe-sandbox).
BASE_SANDBOX_IMAGE = "verity-sandbox:latest"
FE_SANDBOX_IMAGE = "verity-fe-sandbox:latest"


def with_fe_sandbox_image(provisioning: ProvisioningConfig) -> ProvisioningConfig:
    """Default an FE task's sandbox to the FE runtime image, respecting an explicit override.

    The agent must be able to install its requirements and run/self-test its script (mirroring the
    gate's code-runner); the base image lacks the system libs the ML wheels link, so the imports
    fail. Swap the base default for :data:`FE_SANDBOX_IMAGE`; if the request named a *different*
    image, honour it (the operator chose a custom sandbox)."""
    if provisioning.sandbox_image == BASE_SANDBOX_IMAGE:
        return replace(provisioning, sandbox_image=FE_SANDBOX_IMAGE)
    return provisioning

# The verifier-role file holding the hold-out labels — the answer key. It must never reach the
# agent role; see ``assert_answer_key_isolated`` and ``tools/prepare_fe_data.py`` (ADR 0005).
ANSWER_KEY_FILE = "holdout_labels.csv"


def assert_answer_key_isolated(
    agent_files: Mapping[str, bytes], verifier_files: Mapping[str, bytes]
) -> None:
    """Defense-in-depth (I1): fail fast if the answer key's bytes appear in the agent role.

    ADR 0005 routes opaque role -> {filename: bytes} maps and only checks file *presence*, so a
    sloppy prep (a typo, a hand-rolled split dumping everything to one role) could route the
    hold-out labels into the agent's inputs with nothing to catch it. This turns the "isolation by
    construction" convention into an enforced invariant: since the store is content-addressed, a
    cheap content-hash comparison rejects the run loudly if the answer-key blob is also an agent
    blob — under *any* filename. (The subtler "labels embedded in a larger agent file" case is
    handled at the prep boundary by stripping the target column; see ``tools/prepare_fe_data.py``.)
    """
    answer_key = verifier_files.get(ANSWER_KEY_FILE)
    if answer_key is None:
        return  # presence is enforced separately; nothing to compare against here
    key_hash = hashlib.sha256(answer_key).hexdigest()
    leaked = sorted(
        name for name, blob in agent_files.items() if hashlib.sha256(blob).hexdigest() == key_hash
    )
    if leaked:
        raise ValueError(
            f"answer-key isolation violated: the hold-out labels ({ANSWER_KEY_FILE!r}) appear "
            f"verbatim in the agent role as {leaked} — the agent must never receive the labels "
            f"(ADR 0005). Re-prepare the data with the labels confined to the verifier role."
        )


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
    recursion_limit: int = 200  # a floor; the driver clamps it UP to step_budget*headroom (#103)
    sandbox_timeout_s: float = 1500.0
    # Gate per-script budget; calibrated 3x the slowest clean full-data run, 4312s (#111).
    code_timeout_s: float = 12960.0
    step_budget: int | None = None  # per-cycle model-step budget (#103); None -> driver-derived


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
        step_budget=sandbox.step_budget,
    )
