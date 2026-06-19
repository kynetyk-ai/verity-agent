"""The FE-Kaggle verifier — the feature-engineering gate with the **real Kaggle leaderboard** as the
authoritative final test (the `fe-kaggle` task type).

Reuses the §12 feature-engineering domain wholesale (schema / shape / instructions and the shared
scoring helpers) and only swaps the verifier's gate stack. The ladder becomes two-tier:

* **cheap, per-cycle, unlimited (local proxy):** the gate re-runs the agent's *script* on a labeled
  hold-out carved from ``train.csv`` (never the agent's CSV — trusted regeneration) and scores
  balanced accuracy; a submission that doesn't beat our best **accepted** proxy is a cheap reject
  (so we never spend a Kaggle submission on a locally-worse attempt).
* **expensive, rate-limited (the real test):** for a survivor, regenerate the submission on the
  **full train + the real `test.csv`**, then submit to Kaggle and read the public score. Accept iff
  it **beats our best prior Kaggle-confirmed score** (the leaderboard is the incumbent ladder). The
  Kaggle cap is the competition's own daily limit (read from its metadata, default 5); when the
  budget is spent the gate **blocks** (polls) until it frees, up to
  ``wait_deadline_s`` (then a recoverable `GateUnavailable`). The control plane imposes no dispatch
  backstop by default, so the wait is legitimate, not a hang.

The submit happens here, in the **trusted in-process verifier** — creds never reach a worker; the
code-runner worker only produces predictions (`KaggleScorer`).
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    GateUnavailable,
    GateVerdict,
    SupportsRunContext,
    VerdictKind,
    VerifierRequest,
)
from verity.domains.feature_engineering import (
    _RUN_ENV,
    ENTRYPOINT,
    PREDICTIONS_OUTPUT,
    REQUIREMENTS,
    SUBMISSION,
    TEST_INPUT,
    TRAIN_INPUT,
    _parse_predictions,
    _stderr_tail,
    balanced_accuracy,
)
from verity.verifier import (
    CodeRunner,
    GateStep,
    RunRequest,
    RunResult,
    SdkVerifier,
)
from verity.verifier.kaggle import KaggleScorer

__all__ = [
    "KAGGLE_VERIFIER_IDENTITY",
    "build_feature_engineering_kaggle_verifier",
]

KAGGLE_VERIFIER_IDENTITY = "feature-engineering-kaggle-verifier"


@dataclass
class _KaggleGates:
    """The two-tier Submission gates over the agent's regenerated script (one cached run per role).

    Two evaluations of the *same* script, on different data the gate controls:
    - **proxy** (cheap): train on ``agent_train_csv``, predict ``reserved_test_csv``; scored vs the
      held ``reserved_labels`` (never exposed) — the local generalization estimate.
    - **kaggle** (hard): train on ``full_train_csv``, predict ``real_test_csv`` → the submission
      sent to Kaggle. ``_proxy_scores`` / ``_real_scores`` are the verifier's measurement ledgers
      (proposal id → score), read for incumbents by the store-slice's accepted ids (independence).
    """

    runner: CodeRunner
    scorer: KaggleScorer
    agent_train_csv: bytes
    reserved_test_csv: bytes
    reserved_labels: Mapping[str, str]
    full_train_csv: bytes
    real_test_csv: bytes
    timeout_s: float = 900.0
    wait_deadline_s: float = 86_400.0
    poll_interval_s: float = 60.0
    submit_message: str = "verity"
    _run_cache: dict[str, RunResult] = field(default_factory=dict)
    _proxy_scores: dict[str, float] = field(default_factory=dict)
    _real_scores: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def _digest(code: bytes) -> str:
        return hashlib.sha256(code).hexdigest()

    async def _run(
        self, code: bytes, *, inputs: dict[str, bytes], requirements: bytes | None, cache_key: str
    ) -> RunResult:
        cached = self._run_cache.get(cache_key)
        if cached is not None:
            return cached
        result = await self.runner.run(
            RunRequest(
                code=code, entrypoint=ENTRYPOINT, inputs=inputs, output_name=PREDICTIONS_OUTPUT,
                requirements=requirements, network=True, env=_RUN_ENV, timeout_s=self.timeout_s,
            )
        )
        self._run_cache[cache_key] = result
        return result

    async def _proxy_run(self, request: VerifierRequest) -> tuple[bytes | None, RunResult | None]:
        code = request.objects.get(ENTRYPOINT)
        if code is None:
            return None, None
        result = await self._run(
            code,
            inputs={TRAIN_INPUT: self.agent_train_csv, TEST_INPUT: self.reserved_test_csv},
            requirements=request.objects.get(REQUIREMENTS),
            cache_key=f"{self._digest(code)}:proxy",
        )
        return code, result

    def _best(
        self, ledger: dict[str, float], store_slice: tuple[Artifact, ...]
    ) -> tuple[str | None, float]:
        scored = [
            (a.id, ledger[a.id])
            for a in store_slice
            if a.status is ArtifactStatus.ACCEPTED and a.id in ledger
        ]
        if not scored:
            return None, 0.0
        return max(scored, key=lambda pair: pair[1])

    async def runnable(self, request: VerifierRequest) -> GateVerdict | None:
        """Cheap rung: the script executes on the hold-out and predicts every reserved row."""
        code, result = await self._proxy_run(request)
        if code is None or result is None:
            return GateVerdict(VerdictKind.REJECT, f"no {ENTRYPOINT!r} attachment to run")
        if result.timed_out:
            return GateVerdict(VerdictKind.REJECT, "submission timed out before completing")
        if result.exit_code != 0:
            return GateVerdict(
                VerdictKind.REJECT,
                f"submission exited {result.exit_code}: {_stderr_tail(result.stderr)}",
            )
        preds = _parse_predictions(result.output)
        if preds is None:
            return GateVerdict(VerdictKind.REJECT, f"no well-formed {PREDICTIONS_OUTPUT} produced")
        missing = self.reserved_labels.keys() - preds.keys()
        if missing:
            return GateVerdict(
                VerdictKind.REJECT,
                f"predictions cover {len(preds)} of {len(self.reserved_labels)} reserved rows",
            )
        return GateVerdict(VerdictKind.ACCEPT, "runs clean on the hold-out")

    async def proxy_improves(self, request: VerifierRequest) -> GateVerdict | None:
        """Cheap rung: the local proxy score must beat our best **accepted** proxy, else reject —
        the budget-saving filter so a Kaggle call is never spent on a locally-worse attempt."""
        _, result = await self._proxy_run(request)
        preds = _parse_predictions(result.output) if result is not None else None
        if preds is None:
            return GateVerdict(VerdictKind.REJECT, "no scorable hold-out predictions")
        net = balanced_accuracy(self.reserved_labels, preds)
        self._proxy_scores[request.proposal.id] = net
        _, best = self._best(self._proxy_scores, request.store_slice)
        detail = f"local proxy balanced-accuracy {net:.4f} vs best accepted {best:.4f}"
        if net <= best:
            return GateVerdict(VerdictKind.REJECT, f"no local improvement: {detail}")
        return GateVerdict(VerdictKind.ACCEPT, f"local improvement: {detail}")

    async def kaggle(self, request: VerifierRequest) -> GateVerdict | None:
        """Hard rung: regenerate on full/real data, submit to Kaggle, accept iff the public score
        beats our best prior Kaggle-confirmed score. Blocks on the daily budget; degrades on API
        failure (raised `GateUnavailable`)."""
        code = request.objects.get(ENTRYPOINT)
        if code is None:
            return GateVerdict(VerdictKind.REJECT, f"no {ENTRYPOINT!r} attachment to submit")
        result = await self._run(
            code,
            inputs={TRAIN_INPUT: self.full_train_csv, TEST_INPUT: self.real_test_csv},
            requirements=request.objects.get(REQUIREMENTS),
            cache_key=f"{self._digest(code)}:kaggle",
        )
        if result.timed_out or result.exit_code != 0 or _parse_predictions(result.output) is None:
            return GateVerdict(
                VerdictKind.REJECT, "submission did not regenerate a valid file on the real test"
            )
        await self._await_budget()  # block until the daily cap frees (else GateUnavailable)
        assert result.output is not None  # narrowed by the _parse_predictions check above
        real = await self.scorer.submit_and_score(
            result.output, message=self.submit_message, wait_deadline_s=self.wait_deadline_s
        )
        self._real_scores[request.proposal.id] = real
        best_id, best_real = self._best(self._real_scores, request.store_slice)
        detail = f"Kaggle public score {real:.5f} vs best accepted {best_real:.5f}"
        if real > best_real:
            return GateVerdict(
                VerdictKind.ACCEPT, f"improves on the leaderboard: {detail}",
                score=real, supersedes=best_id,
            )
        return GateVerdict(VerdictKind.REJECT, f"no leaderboard improvement: {detail}", score=real)

    async def _await_budget(self) -> None:
        """Block until Kaggle has submission budget today; the user's choice over resting tentative.
        Bounded by ``wait_deadline_s`` (then a recoverable `GateUnavailable`) so it never hangs."""
        waited = 0.0
        while True:
            if await self.scorer.remaining_budget() > 0:
                return
            if waited >= self.wait_deadline_s:
                raise GateUnavailable(
                    f"no Kaggle submission budget within {self.wait_deadline_s}s (daily cap)"
                )
            await asyncio.sleep(self.poll_interval_s)
            waited += self.poll_interval_s


def build_feature_engineering_kaggle_verifier(
    runner: CodeRunner,
    *,
    scorer: KaggleScorer,
    agent_train_csv: bytes,
    reserved_test_csv: bytes,
    reserved_labels: Mapping[str, str],
    full_train_csv: bytes,
    real_test_csv: bytes,
    timeout_s: float = 900.0,
    wait_deadline_s: float = 86_400.0,
    poll_interval_s: float = 60.0,
    submit_message: str = "verity",
) -> SdkVerifier:
    """The opaque FE-Kaggle verifier: cheap local proxy gates + the hard Kaggle leaderboard gate."""
    gates = _KaggleGates(
        runner=runner, scorer=scorer,
        agent_train_csv=agent_train_csv, reserved_test_csv=reserved_test_csv,
        reserved_labels=dict(reserved_labels),
        full_train_csv=full_train_csv, real_test_csv=real_test_csv,
        timeout_s=timeout_s,
        wait_deadline_s=wait_deadline_s, poll_interval_s=poll_interval_s,
        submit_message=submit_message,
    )
    sinks = (runner,) if isinstance(runner, SupportsRunContext) else ()
    return SdkVerifier(
        identity=KAGGLE_VERIFIER_IDENTITY,
        context_sinks=sinks,
        pipelines={
            SUBMISSION: (
                GateStep("runs-clean", gates.runnable, is_hard=False),
                GateStep("proxy-improves", gates.proxy_improves, is_hard=False),
                GateStep("kaggle", gates.kaggle, is_hard=True),
            ),
        },
    )
