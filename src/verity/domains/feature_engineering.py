"""The model-building discovery domain — the v1 validation target and MVP (spec §12, Phase 4).

The agent is given a dataset (one named column is the target ``class``, the rest are candidate
features) and asked to **build a single self-contained script that predicts better than the
incumbent**. Originally framed as strict feature engineering (spec §12); since broadened so any
single-script approach is fair game — engineered features, model choice, ensembling, calibration,
imbalance handling. The deliverable each cycle is a **self-contained script** (+ its pinned package
list) written to ``outbox/``, from which the harness harvests the code object (§3.4). It never
returns a trained model: the model is produced by the gate, from the script, on a **reserved dataset
the agent never sees** (§12) — so anything that peeks at the target inflates the agent's own score
but fails to generalize, and is caught on the reserved set.

Divergence from vendored §12: the original domain also asked the agent for a JSON feature
description, harvested one typed ``Feature`` child per declared feature, and ran a per-feature
grounding gate + a ``features-defined`` refine + a per-feature complexity penalty. With the task
broadened beyond features that machinery was dropped — the submission is the unit, and the gate
stack is runnable → ``tentative`` and selection-on-the-reserved-set → ``accepted`` (reject-only, no
refine). The kernel still supports harvested children / grounding / refine; this domain no longer
exercises them (§13.3/13.4/13.6 describe the original feature-harvest demonstration).

This module holds both sides for the domain. The **control-plane side**: the schema, the gated-type
coverage, the proposal-shape spec, and the domain instructions. The **verifier side**
(:func:`build_feature_engineering_verifier`, by ``verifier_key``): the two ``Submission`` gates —
runnable → ``tentative`` and selection-on-the-reserved-set → ``accepted``.

Schema (§12, broadened):
* ``DatasetVersion`` — the pinned dataset (the CSV, the named target, fixed folds); the larger
  reserved verification set is held by the gate, never exposed here. Root artifact.
* ``Submission`` — the agent's per-cycle proposal and the gated artifact: an object-bearing payload
  referencing the script + its declared package list.
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections.abc import Mapping
from dataclasses import dataclass, field

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    GateVerdict,
    SupportsRunContext,
    VerdictKind,
    VerifierRequest,
)
from verity.control_plane.commit import ShapeError, ShapeValidator
from verity.control_plane.independence import StoreInput
from verity.control_plane.registries import (
    ArtifactTypeDef,
    GatedTypeRegistry,
    OperationSignature,
    SchemaRegistry,
)
from verity.verifier import (
    CodeRunner,
    GateStep,
    RunRequest,
    RunResult,
    SdkVerifier,
)

__all__ = [
    "DATASET_VERSION",
    "SUBMISSION",
    "ENTRYPOINT",
    "REQUIREMENTS",
    "TRAIN_INPUT",
    "TEST_INPUT",
    "PREDICTIONS_OUTPUT",
    "FE_VERIFIER_IDENTITY",
    "FEATURE_ENGINEERING_INSTRUCTIONS",
    "FeatureEngineeringDomain",
    "build_feature_engineering_domain",
    "build_feature_engineering_verifier",
    "declared_objects",
    "balanced_accuracy",
]

DATASET_VERSION = "DatasetVersion"
SUBMISSION = "Submission"

# The reserved object names the agent writes to outbox/ (the script + its declared package list).
ENTRYPOINT = "submission.py"
REQUIREMENTS = "requirements.txt"

# The script I/O contract (the gate runs the script over these). The script reads its CSVs from the
# directory in ``$VERITY_DATA`` (default ``data``) and writes predictions to ``$VERITY_OUT``
# (default ``out``), so the same script runs unchanged in the agent's sandbox and the gate's runner.
TRAIN_INPUT = "train.csv"
TEST_INPUT = "test.csv"
PREDICTIONS_OUTPUT = "predictions.csv"
_RUN_ENV = {"VERITY_DATA": "/data", "VERITY_OUT": "/out"}

# The verifier package's identity — distinct from any proposer, so proposer != gate holds (§7.2).
FE_VERIFIER_IDENTITY = "feature-engineering-verifier"


@dataclass(frozen=True, slots=True)
class FeatureEngineeringDomain:
    """The control-plane side a task wires in (the verifier package is chosen by ``verifier_key``).

    ``domain_instructions`` is the domain layer of the composed system prompt (§3.4) — the script
    contract and the steer *we* choose to expose to the agent.
    """

    schema: SchemaRegistry
    gated_types: GatedTypeRegistry
    shape_validator: ShapeValidator
    domain_instructions: str


def build_feature_engineering_domain() -> FeatureEngineeringDomain:
    schema = SchemaRegistry()
    schema.register_type(ArtifactTypeDef(DATASET_VERSION, is_root=True))
    schema.register_type(ArtifactTypeDef(SUBMISSION))
    schema.register_operation(
        OperationSignature("submit", inputs=(DATASET_VERSION,), output=SUBMISSION)
    )
    # A revision of a prior submission enters via a 'revises' op (kernel-general, §6); this domain
    # never issues a refine, so it is unused here, but the op stays registered for lineage.
    schema.register_operation(
        OperationSignature("revises", inputs=(SUBMISSION,), output=SUBMISSION)
    )

    gated_types = GatedTypeRegistry()
    # 'Submission' is gated; its selection gate must beat the incumbents and deflate by the
    # rejected-log (the trial count, §12), so it declares both slices (§8.3, §10).
    gated_types.gate(
        SUBMISSION, declared_inputs=frozenset({StoreInput.INCUMBENTS, StoreInput.REJECTED_LOG})
    )

    return FeatureEngineeringDomain(
        schema=schema,
        gated_types=gated_types,
        shape_validator=_validate_shape,
        domain_instructions=FEATURE_ENGINEERING_INSTRUCTIONS,
    )


def _validate_shape(artifact: Artifact) -> ShapeError | None:
    """Presence/type only (§7.0, ADR 0001): a ``Submission`` declares its script + package list.

    Checks the *composition* the agent must produce — never quality. The actual object bytes (the
    script, the requirements) are checked by the verifier's runnable gate, which receives the
    harvested attachments; here we only validate the typed payload the agent declares.
    """
    if artifact.type != SUBMISSION:
        return None
    payload = artifact.payload
    if not isinstance(payload, dict):
        return ShapeError("a Submission payload must be an object")
    if not isinstance(payload.get("entrypoint"), str):
        return ShapeError("a Submission must declare a string 'entrypoint' (the script file name)")
    if not isinstance(payload.get("requirements"), str):
        return ShapeError(
            "a Submission payload must declare a string 'requirements' (the package-list file name)"
        )
    return None


def declared_objects(artifact: Artifact) -> frozenset[str]:
    """The object names a Submission declares — its ``entrypoint`` script + ``requirements`` file.

    The control plane harvests only these from the outbox (an undeclared file is dropped), so the
    set is read straight from the payload the agent submits. Shape validation has already ensured
    they are non-empty strings; anything else (or a non-Submission) declares no objects.
    """
    if artifact.type != SUBMISSION or not isinstance(artifact.payload, dict):
        return frozenset()
    candidates = (artifact.payload.get("entrypoint"), artifact.payload.get("requirements"))
    return frozenset(name for name in candidates if isinstance(name, str) and name)


# ----------------------------------------------------- the verifier package: two Submission gates


def build_feature_engineering_verifier(
    runner: CodeRunner,
    *,
    agent_train_csv: bytes,
    reserved_test_csv: bytes,
    reserved_labels: Mapping[str, str],
    margin: float = 0.0,
    trial_deflation: float = 0.0,
    timeout_s: float = 900.0,
) -> SdkVerifier:
    """The opaque verifier package for the §12 domain — the two ``Submission`` gates (ADR 0001).

    The gate trains the submitted script on ``agent_train_csv`` and predicts ``reserved_test_csv``,
    then scores **balanced accuracy** against ``reserved_labels`` — which it **holds and never
    exposes**, so a leaked-target submission inflates the agent's own score but is caught here (§12,
    §13.11). The cheap **runs-clean** gate (→ ``tentative``) requires the script to execute and
    predict every reserved row; the hard **selection** gate (→ ``accepted``) requires the score to
    beat the incumbent by ``margin``, with the bar raised ``trial_deflation`` per prior rejected
    submission (the trial count is the rejected-log, §12). Reject-only — this domain issues no
    ``refine``.

    The incumbents' scores come from the verifier's own measurement ledger (it scored them), keyed
    by the store-slice's *current* accepted ids — so independence holds (the slice carries status,
    the verifier the numbers it computed; no rationale or score round-trips through the store).
    """
    gates = _SubmissionGates(
        runner=runner,
        agent_train_csv=agent_train_csv,
        reserved_test_csv=reserved_test_csv,
        reserved_labels=dict(reserved_labels),
        margin=margin,
        trial_deflation=trial_deflation,
        timeout_s=timeout_s,
    )
    # If the runner provisions workers (the backend-backed runner), let the control plane's run
    # identity reach it, so its code-runner workers are labelled for audit + reaping (7.4.h). A pure
    # in-process runner (FakeCodeRunner) does not implement the capability and registers nothing.
    sinks = (runner,) if isinstance(runner, SupportsRunContext) else ()
    return SdkVerifier(
        identity=FE_VERIFIER_IDENTITY,
        context_sinks=sinks,
        pipelines={
            SUBMISSION: (
                GateStep("runs-clean", gates.runnable, is_hard=False),
                GateStep("selection", gates.selection, is_hard=True),
            ),
        },
    )


@dataclass
class _SubmissionGates:
    """The two ``Submission`` gates over one shared, cached run of the submitted script (§12).

    Both gates execute the *same* run (train on the agent's data, predict the reserved features), so
    a single submission trains once: ``runs-clean`` checks it executed and is well-formed,
    ``selection`` scores it. ``_scores`` is the verifier's measurement ledger (proposal id → score)
    used to read incumbents' scores back.
    """

    runner: CodeRunner
    agent_train_csv: bytes
    reserved_test_csv: bytes
    reserved_labels: Mapping[str, str]
    margin: float
    trial_deflation: float
    timeout_s: float
    _run_cache: dict[str, RunResult] = field(default_factory=dict)
    _scores: dict[str, float] = field(default_factory=dict)

    async def _run(self, request: VerifierRequest) -> RunResult | None:
        code = request.objects.get(ENTRYPOINT)
        if code is None:
            return None
        digest = hashlib.sha256(code).hexdigest()
        cached = self._run_cache.get(digest)
        if cached is not None:
            return cached
        result = await self.runner.run(
            RunRequest(
                code=code,
                entrypoint=ENTRYPOINT,
                inputs={TRAIN_INPUT: self.agent_train_csv, TEST_INPUT: self.reserved_test_csv},
                output_name=PREDICTIONS_OUTPUT,
                requirements=request.objects.get(REQUIREMENTS),
                network=True,
                env=_RUN_ENV,
                timeout_s=self.timeout_s,
            )
        )
        self._run_cache.clear()  # bounded to the most-recent submission's run
        self._run_cache[digest] = result
        return result

    async def runnable(self, request: VerifierRequest) -> GateVerdict | None:
        """Cheap rung: the script executes on the agent's data and predicts every reserved row."""
        result = await self._run(request)
        if result is None:
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
                f"predictions cover {len(preds)} of {len(self.reserved_labels)} reserved rows "
                f"({len(missing)} missing)",
            )
        return GateVerdict(VerdictKind.ACCEPT, "runs clean and predicts every reserved row")

    async def selection(self, request: VerifierRequest) -> GateVerdict | None:
        """Hard rung: balanced accuracy beats the incumbent + deflated margin."""
        result = await self._run(request)
        preds = _parse_predictions(result.output) if result is not None else None
        if preds is None or (self.reserved_labels.keys() - preds.keys()):
            return GateVerdict(VerdictKind.REJECT, "submission produced no scorable predictions")
        score = balanced_accuracy(self.reserved_labels, preds)
        self._scores[request.proposal.id] = score

        best_id, baseline = self._best_incumbent(request.store_slice)
        trials = sum(1 for a in request.store_slice if a.status is ArtifactStatus.REJECTED)
        bar = baseline + self.margin + self.trial_deflation * trials
        detail = (
            f"balanced-accuracy {score:.4f} vs bar {bar:.4f} "
            f"(incumbent {baseline:.4f}, {trials} prior trials)"
        )
        if score > bar:
            return GateVerdict(
                VerdictKind.ACCEPT, f"improves: {detail}", score=score, supersedes=best_id
            )
        return GateVerdict(VerdictKind.REJECT, f"does not improve: {detail}", score=score)

    def _best_incumbent(self, store_slice: tuple[Artifact, ...]) -> tuple[str | None, float]:
        """The top-scoring currently-accepted submission (status from the slice, score from us)."""
        scored = [
            (a.id, self._scores[a.id])
            for a in store_slice
            if a.status is ArtifactStatus.ACCEPTED and a.id in self._scores
        ]
        if not scored:
            return None, 0.0
        return max(scored, key=lambda pair: pair[1])


def _stderr_tail(stderr: str) -> str:
    """The most informative stderr line for a failed run — the Python exception if present, else
    the last real line. Skips pip's own noise (``[notice] …`` / ``WARNING: …``) so a pip-upgrade
    notice never masks the actual error (e.g. ``OSError: libgomp.so.1`` from a missing system
    library), which is what the agent needs in its reject feedback to fix the next submission."""
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    meaningful = [ln for ln in lines if not ln.startswith(("[notice]", "WARNING:"))]
    if not meaningful:
        return "no stderr"
    for ln in reversed(meaningful):
        if "Error" in ln or "Exception" in ln or "error:" in ln:
            return ln
    return meaningful[-1]


def _parse_predictions(output: bytes | None) -> dict[str, str] | None:
    """Parse a predictions CSV (``id,class``) into ``{id: class}``; ``None`` if not well-formed."""
    if not output:
        return None
    try:
        reader = csv.DictReader(io.StringIO(output.decode("utf-8")))
        names = reader.fieldnames
        if names is None or "id" not in names or "class" not in names:
            return None
        preds = {row["id"]: row["class"] for row in reader if row.get("id")}
    except (UnicodeDecodeError, csv.Error):
        return None
    return preds or None


def balanced_accuracy(truth: Mapping[str, str], preds: Mapping[str, str]) -> float:
    """The §12 metric: the unweighted mean of per-class recall, so every class counts equally."""
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for rid, label in truth.items():
        totals[label] = totals.get(label, 0) + 1
        if preds.get(rid) == label:
            correct[label] = correct.get(label, 0) + 1
    if not totals:
        return 0.0
    return sum(correct.get(c, 0) / totals[c] for c in totals) / len(totals)


FEATURE_ENGINEERING_INSTRUCTIONS = f"""\
You are a discovery agent on a tabular prediction task. Each cycle you produce ONE submission: a
self-contained Python script that reads the data, builds a predictive pipeline, and writes a
prediction for every test row. Any technique that fits in a single script and improves the held-out
score is fair game — engineered features, the model you pick, an ensemble, calibration, handling
class imbalance. There is no mandated method. Your aim across cycles is to climb into the TOP TIER
of the real leaderboard: each cycle either improves on your best or is sent back to revise toward a
competitive target (how that works is below).

The script contract (the gate runs your script; honour it exactly):
- Resolve the data directory as `os.environ.get("VERITY_DATA", "data")` and the output directory as
  `os.environ.get("VERITY_OUT", "out")`. (The defaults work in your sandbox; the gate sets these.)
- Read the labelled training data from `<VERITY_DATA>/{TRAIN_INPUT}` and the unlabelled rows to
  predict from `<VERITY_DATA>/{TEST_INPUT}`. The target is `class`; `{TEST_INPUT}` has no `class`.
- Clean the data, build your pipeline, train on the training data, then predict a `class` for every
  row of `{TEST_INPUT}`.
- Write predictions to `<VERITY_OUT>/{PREDICTIONS_OUTPUT}` with exactly two columns: `id,class`.
- The script must be deterministic (fix every random seed) and self-contained.

The runner (where the gate executes your script): a CPU-only Linux container, Python 3.12, with your
pinned `{REQUIREMENTS}` pip-installed (network on for the install). The common system libraries for
the CPU ML stack are present, so scikit-learn, LightGBM, XGBoost, pandas, and numpy all work.
Budget: a few minutes, ~2 GB RAM, ~1 GB scratch. So: prefer fast, wheel-installable CPU libraries
and a bounded model; do NOT use deep-learning frameworks (torch / tensorflow will not fit or
finish), and avoid libraries with no prebuilt wheel (there is no compiler in the runner).

Orient before you propose:
- If there is NO incumbent yet, EXPLORE THE DATA WITH CODE first — do not try to read the raw files,
  they are too large. Write a short script that loads the data and prints its shape, column dtypes,
  the target's class balance, summary statistics, missingness, and a few candidate signals
  (correlations or simple per-class means). Let what you find drive your first submission.
- If you have prior work, it is provided under `scratch/provided/` (see its `INDEX.md`): your best
  accepted submission, or — on a REVISE cycle — the exact script you were just asked to revise.
  Read it, work out WHY it scores as it does and where it is WEAK, then make a focused change that
  targets that weakness rather than starting from scratch. You may edit your workspace freely.

What to deliver to `outbox/` each cycle:
- `{ENTRYPOINT}` — the script above.
- `{REQUIREMENTS}` — the pinned package list your script needs (e.g. `pandas==2.2.2`), one per line.
  The gate installs exactly these before running your script, so pin versions for reproducibility.
- The proposal payload (via your submit/revises tool): `entrypoint` = "{ENTRYPOINT}" and
  `requirements` = "{REQUIREMENTS}". The submission is the unit — you do not report individual
  features or changes.

How you are judged (you never see the judge's data):
- Your goal is the TOP TIER of the real leaderboard. The gate runs your script on data you cannot
  see (a reserved hold-out) and scores BALANCED ACCURACY, then estimates whether that score would
  reach the competitive bar — a target read live from the leaderboard. If it would NOT, your
  submission is sent back to REVISE: you are told your estimated score, the target, and the gap —
  close it and resubmit. No real leaderboard submission is spent until you are competitive, so keep
  improving the held-out score.
- Once the estimate clears the bar, the script is regenerated on the full data and submitted to the
  REAL leaderboard; it is accepted only if its public score beats your best so far. Each accepted
  submission is a real step up the leaderboard — keep climbing across cycles.
- Anything that peeks at the target looks great on your own split but fails on the hold-out and the
  real data. A submission that errors, times out, or fails to predict every row is rejected
  outright; there is no partial credit. Confirm your script runs cleanly before you submit.

Data + speed (the amount of training data is the real lever here — use ALL of it):
- TRAIN ON THE FULL training data provided; do NOT subsample it. With balanced accuracy on this
  problem the quantity of training data is the dominant driver of the score, and a gradient-boosted
  tree model (LightGBM / XGBoost / CatBoost) fits the whole set in a couple of minutes — comfortably
  within the time budget.
- Optimize for BALANCED accuracy, not raw accuracy: validate with stratified cross-validation and
  handle the class imbalance (class weights / resampling / threshold tuning).
- The gate runs your script under a generous time budget, but a model that does not finish scores
  nothing — keep any ensemble or search bounded. To check it runs without burning your own budget,
  test on a small sample (e.g. a few thousand rows); the gate trains the real thing on the full
  data. Then submit.

Useful domain knowledge: differences between photometric bands ("colour indices", e.g. u-g, g-r,
r-i, i-z) and `redshift` carry most of the signal; encode the categorical `spectral_type` and the
`galaxy_population` column (both are present in train AND test — explore the columns to see exactly
what is available); and because scoring is balanced accuracy, handle class imbalance (class weights
/ resampling)."""
