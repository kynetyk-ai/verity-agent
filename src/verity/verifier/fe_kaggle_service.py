"""Build the FE-Kaggle verifier from a per-task :class:`VerifierSetup`, **verifier-side** (§9.1).

This is the half of the FE-Kaggle verifier that lives in the *verifier image*, not the control
plane: it carries the domain gates, the `kaggle` extra, and the verifier's own code-runner
provisioning. The control plane ships only opaque data + knobs (the CSV splits, the competition
slug, the gate timeouts) in a :class:`VerifierSetup`; this builder turns them into a runnable
:class:`SdkVerifier`. Keeping it here is what lets the control-plane image drop ``--extra kaggle``
and every gate module — a new verifier ships its own image + this kind of builder, no CP rebuild.

**Setup contract** (mirrors `composition.fe_kaggle`'s in-process construction, now over the wire):

* ``objects`` — the four CSV blobs the gates run on:
  ``agent_train`` / ``reserved_test`` (the cheap local-proxy split) and
  ``full_train`` / ``real_test`` (the hard Kaggle run).
* ``params`` — ``reserved_labels`` (the held answer key, never exposed), ``competition`` (the slug),
  and the gate knobs (``timeout_s``, ``wait_deadline_s``, ``poll_interval_s``,
  ``score_poll_interval_s``, ``submit_message``, optional ``daily_submission_limit``).

The verifier's *own* code-runner provisioning (image / memory / staging) and the Kaggle creds come
from the verifier service's environment — its deployment concern, not the control plane's. Set
``VERITY_KAGGLE_FAKE=1`` to score against a deterministic :class:`FakeKaggleScorer` (the offline /
``@docker`` path, no creds), with scores from ``params['fake_scores']`` (default ``1.0``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from verity.contracts.ports import VerifierPort, VerifierSetup
from verity.logging import get_logger

log = get_logger("verity.verifier.fe_kaggle_service")

# The code-runner worker label `config` — matches the control-plane-side `FE_KAGGLE_TASK_ID`.
FE_KAGGLE_CONFIG = "fe-kaggle"

_DEFAULT_CODE_IMAGE = "python:3.12-slim"
_DEFAULT_CODE_MEMORY = "512m"


def _require_object(objects: Mapping[str, bytes], key: str) -> bytes:
    try:
        return objects[key]
    except KeyError:
        raise ValueError(f"fe-kaggle setup is missing the {key!r} object") from None


def _require_param(params: Mapping[str, Any], key: str) -> Any:
    try:
        return params[key]
    except KeyError:
        raise ValueError(f"fe-kaggle setup is missing the {key!r} param") from None


def build_fe_kaggle_verifier_from_setup(setup: VerifierSetup) -> VerifierPort:
    """Construct the FE-Kaggle :class:`SdkVerifier` from a control-plane-shipped setup payload."""
    # Lazy imports: only the running verifier image needs the gate code + the `kaggle` extra; the
    # module imports cleanly without them so the entrypoint test stays light.
    from verity.domains.feature_engineering_kaggle import (
        build_feature_engineering_kaggle_verifier,
    )
    from verity.provisioning.docker import DockerBackend
    from verity.verifier import BackendCodeRunner, FakeKaggleScorer, RealKaggleScorer
    from verity.verifier.kaggle import KaggleScorer

    objects, params = setup.objects, setup.params
    competition = str(_require_param(params, "competition"))

    runner = BackendCodeRunner(
        backend=DockerBackend(),  # staging_root from VERITY_WORKER_STAGING (sibling-mount path)
        image=os.environ.get("VERITY_CODE_IMAGE", _DEFAULT_CODE_IMAGE),
        memory=os.environ.get("VERITY_CODE_MEMORY", _DEFAULT_CODE_MEMORY),
        config=FE_KAGGLE_CONFIG,
    )

    fake = bool(os.environ.get("VERITY_KAGGLE_FAKE"))
    scorer: KaggleScorer
    if fake:
        fake_scores = params.get("fake_scores") or [1.0]
        scorer = FakeKaggleScorer(scores=[float(s) for s in fake_scores])
    else:
        limit = params.get("daily_submission_limit")
        scorer = RealKaggleScorer(
            competition=competition,
            poll_interval_s=float(params.get("score_poll_interval_s", 20.0)),
            daily_limit_override=int(limit) if limit is not None else None,
        )

    log.info("fe_kaggle_verifier_built", competition=competition, fake=fake)
    return build_feature_engineering_kaggle_verifier(
        runner,
        scorer=scorer,
        agent_train_csv=_require_object(objects, "agent_train"),
        reserved_test_csv=_require_object(objects, "reserved_test"),
        reserved_labels=dict(_require_param(params, "reserved_labels")),
        full_train_csv=_require_object(objects, "full_train"),
        real_test_csv=_require_object(objects, "real_test"),
        timeout_s=float(params.get("timeout_s", 900.0)),
        wait_deadline_s=float(params.get("wait_deadline_s", 86_400.0)),
        poll_interval_s=float(params.get("poll_interval_s", 60.0)),
        submit_message=str(params.get("submit_message", "verity fe-kaggle")),
    )
