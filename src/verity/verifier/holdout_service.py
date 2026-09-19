"""Build the holdout-experiment verifier from a per-task :class:`VerifierSetup`, **verifier-side**.

The verifier for the ablation ladder (Exp 1–4b, ``docs/experimental-design.md`` §6.2): a controlled,
deterministic, local scorer — **no Kaggle, no competitive bar, no calibration**. It reuses the §12
FE holdout scorer (run the agent's script on a reserved split → balanced accuracy) and rides the
**existing verifier image** — ``balanced_accuracy`` is pure Python and the agent's script
pip-installs its own ML stack into the code-runner tmpfs at gate time, so this needs no new
container, just an entry point in the ``verity.verifier_types`` group.

The one experimental knob is the verdict policy (``accept_policy``), which has exactly two values
across all five conditions and selects which hard step the §12 scorer composes:

* ``improve_over_best_prior`` (Exp 3/4/4b) — ACCEPT iff the holdout score beats the best prior, else
  REJECT (binary; never ``refine`` — failures stay ``rejected`` and out of the curated context).
* ``always`` (Exp 1/2) — record the score and ACCEPT unconditionally.

**Setup contract** (ADR 0005: the verifier role's *files*, routed by filename, plus knobs):

* ``objects`` — three verifier-role CSV blobs, keyed by filename: ``train.csv`` (the agent's
  training set the gate re-runs the script on), ``holdout.csv`` (the reserved features),
  ``holdout_labels.csv`` (the answer key — present here, never in the agent role). No
  ``full_train.csv`` / real ``test.csv`` / competition slug (those are the fe-kaggle gate's).
* ``params`` — ``accept_policy`` (default ``improve_over_best_prior``), ``margin`` (what counts as
  an improvement, default ``0.0``), ``timeout_s`` (per-script code-runner budget).

The verifier's own code-runner provisioning (image / memory / staging) comes from the verifier
service's environment — its deployment concern, identical to the fe-kaggle verifier's runner.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from verity.contracts.ports import VerifierPort, VerifierSetup
from verity.logging import get_logger

if TYPE_CHECKING:
    from verity.verifier.registry import VerifierRegistry

log = get_logger("verity.verifier.holdout_service")

# The verifier name (matches the control-plane-side task config + the code-runner worker label).
HOLDOUT_CONFIG = "holdout-experiment"

# The same code-runner sizing the fe-kaggle verifier uses: the submitted FE script pip-installs a
# real ML stack into a RAM-backed exec tmpfs, then trains — the bare worker defaults cannot fit
# that. Each is overridable from the verifier service's env.
_DEFAULT_CODE_IMAGE = "python:3.12-slim"
_DEFAULT_CODE_MEMORY = "4g"
_DEFAULT_CODE_TMPFS = "2g"
_DEFAULT_CODE_CPUS = "16"
# Generous PID cap: the FE prompt mandates n_jobs=-1, which on 16 cores spawns far more than the
# bare-default 128 threads/procs → pthread_create deadlock (REPORT §2a). Overridable.
_DEFAULT_CODE_PIDS = "4096"


def _require_object(objects: Mapping[str, bytes], key: str) -> bytes:
    try:
        return objects[key]
    except KeyError:
        raise ValueError(f"holdout-experiment setup is missing the {key!r} object") from None


def build_holdout_verifier_from_setup(setup: VerifierSetup) -> VerifierPort:
    """Construct the holdout-experiment :class:`SdkVerifier` from a control-plane-shipped setup."""
    # Lazy imports: only the running verifier image needs the gate code; the module imports cleanly
    # without it so the entrypoint test stays light.
    from verity.domains.feature_engineering import (
        build_feature_engineering_verifier,
        parse_label_csv,
    )
    from verity.provisioning.docker import DockerBackend
    from verity.verifier import BackendCodeRunner

    objects, params = setup.objects, setup.params
    accept_policy = str(params.get("accept_policy", "improve_over_best_prior"))

    runner = BackendCodeRunner(
        backend=DockerBackend(),  # staging_root from VERITY_WORKER_STAGING (sibling-mount path)
        image=os.environ.get("VERITY_CODE_IMAGE", _DEFAULT_CODE_IMAGE),
        memory=os.environ.get("VERITY_CODE_MEMORY", _DEFAULT_CODE_MEMORY),
        tmpfs_size=os.environ.get("VERITY_CODE_TMPFS", _DEFAULT_CODE_TMPFS),
        cpus=os.environ.get("VERITY_CODE_CPUS", _DEFAULT_CODE_CPUS),
        pids_limit=int(os.environ.get("VERITY_CODE_PIDS", _DEFAULT_CODE_PIDS)),
        config=HOLDOUT_CONFIG,
    )

    log.info("holdout_verifier_built", accept_policy=accept_policy)
    return build_feature_engineering_verifier(
        runner,
        agent_train_csv=_require_object(objects, "train.csv"),
        reserved_test_csv=_require_object(objects, "holdout.csv"),
        reserved_labels=parse_label_csv(_require_object(objects, "holdout_labels.csv")),
        margin=float(_param(params, "margin", 0.0)),
        timeout_s=float(_param(params, "timeout_s", 900.0)),
        accept_policy=accept_policy,
    )


def _param(params: Mapping[str, Any], key: str, default: float) -> Any:
    value = params.get(key)
    return default if value is None else value


def register_verifier(registry: VerifierRegistry) -> None:
    """Register the ``holdout-experiment`` verifier (a ``verity.verifier_types`` entry point)."""
    registry.register_setup(HOLDOUT_CONFIG, build_holdout_verifier_from_setup)
