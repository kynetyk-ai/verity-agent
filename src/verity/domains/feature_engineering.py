"""The feature-engineering domain — the v1 validation target and MVP (spec §12, Phase 4).

A feature-engineering agent is given a dataset (one named column is the target ``class``, the rest
are candidate features) and asked to **clean the data and design 1–5 new features that improve a
model**. Each cycle its deliverable is a **self-contained script** (cleans, engineers features,
trains, predicts) plus a **JSON description** of the features it added — both written to
``outbox/``, from which the harness harvests the code object (§3.4). It never returns a trained
model: the model is produced by the gate, from the script, on a **reserved dataset the agent never
sees** (§12) — so a feature that peeks at the target inflates the agent's own score but fails to
generalize, and is caught on the reserved set.

This module holds both sides for the domain. The **control-plane side**: the schema, the gated-type
coverage, the proposal-shape spec, the domain instructions, and the feature harvester. The
**verifier side** (:func:`build_feature_engineering_verifier`, by ``verifier_key``): the two
``Submission`` gates — runnable → ``tentative`` and selection-on-the-reserved-set → ``accepted``,
with a static ``features-defined`` refine — plus the ``Feature`` grounding gate.

Schema (§12):
* ``DatasetVersion`` — the pinned dataset (the CSV, the named target, fixed folds); the larger
  reserved verification set is held by the gate, never exposed here. Root artifact.
* ``Submission`` — the agent's per-cycle proposal and the gated artifact: an object-bearing payload
  referencing the script + its declared package list, plus the structured feature description.
* ``Feature`` — harvested by the harness from the submission's description, one typed artifact per
  declared feature (Phase 4.3), linked to its ``Submission`` by a ``harvest`` operation (§4.1).
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
    VerdictKind,
    VerifierRequest,
)
from verity.control_plane.commit import ShapeError, ShapeValidator
from verity.control_plane.config import HarvestedChild, Harvester
from verity.control_plane.independence import StoreInput
from verity.control_plane.registries import (
    ArtifactTypeDef,
    GatedTypeRegistry,
    OperationSignature,
    SchemaRegistry,
)
from verity.verifier import (
    CheckOutcome,
    CodeRunner,
    GateStep,
    RunRequest,
    RunResult,
    SdkVerifier,
    deterministic_check,
)

__all__ = [
    "DATASET_VERSION",
    "SUBMISSION",
    "FEATURE",
    "ENTRYPOINT",
    "REQUIREMENTS",
    "TRAIN_INPUT",
    "TEST_INPUT",
    "PREDICTIONS_OUTPUT",
    "MAX_FEATURES",
    "FE_VERIFIER_IDENTITY",
    "FEATURE_ENGINEERING_INSTRUCTIONS",
    "FeatureEngineeringDomain",
    "build_feature_engineering_domain",
    "build_feature_engineering_verifier",
    "harvest_features",
    "balanced_accuracy",
]

DATASET_VERSION = "DatasetVersion"
SUBMISSION = "Submission"
FEATURE = "Feature"

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

# A submission declares 1–5 engineered features (§12).
MAX_FEATURES = 5

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
    harvester: Harvester


def harvest_features(submission: Artifact) -> list[HarvestedChild]:
    """Derive one ``Feature`` child per declared feature from a submission's description (§12).

    The harness mints these as inspectable provenance nodes via a ``harvest`` operation; each then
    clears its own grounding gate (genuinely defined by the code) or is refused (§5.7, §13.6).
    """
    payload = submission.payload
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        return []
    return [
        HarvestedChild(artifact_type=FEATURE, op_name="harvest", payload=feature)
        for feature in features
        if isinstance(feature, dict)
    ]


def build_feature_engineering_domain() -> FeatureEngineeringDomain:
    schema = SchemaRegistry()
    schema.register_type(ArtifactTypeDef(DATASET_VERSION, is_root=True))
    schema.register_type(ArtifactTypeDef(SUBMISSION))
    schema.register_type(ArtifactTypeDef(FEATURE))
    schema.register_operation(
        OperationSignature("submit", inputs=(DATASET_VERSION,), output=SUBMISSION)
    )
    # A refine sends a submission back; its revision enters via a 'revises' op (§6, §7.5b).
    schema.register_operation(
        OperationSignature("revises", inputs=(SUBMISSION,), output=SUBMISSION)
    )
    # The harness harvests one Feature per declared feature from the submission's description (§12).
    schema.register_operation(OperationSignature("harvest", inputs=(SUBMISSION,), output=FEATURE))

    gated_types = GatedTypeRegistry()
    # 'Submission' is gated; its selection gate must beat the incumbents net of complexity and
    # deflate by the rejected-log (the trial count, §12), so it declares both slices (§8.3, §10).
    gated_types.gate(
        SUBMISSION, declared_inputs=frozenset({StoreInput.INCUMBENTS, StoreInput.REJECTED_LOG})
    )
    # 'Feature' is gated by a cheap grounding check (genuinely defined by the code); it sees only
    # the artifact under test (+ its code sidecar), so its slice is empty (§5.7 holds for it too).
    gated_types.gate(FEATURE, declared_inputs=frozenset())

    return FeatureEngineeringDomain(
        schema=schema,
        gated_types=gated_types,
        shape_validator=_validate_shape,
        domain_instructions=FEATURE_ENGINEERING_INSTRUCTIONS,
        harvester=harvest_features,
    )


def _validate_shape(artifact: Artifact) -> ShapeError | None:
    """Presence/type only (§7.0, ADR 0001): a ``Submission`` declares its script + 1–5 features.

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
    features = payload.get("features")
    if not isinstance(features, list) or not 1 <= len(features) <= MAX_FEATURES:
        return ShapeError(
            f"a Submission payload must carry a 'features' list of 1..{MAX_FEATURES} items"
        )
    for i, feature in enumerate(features):
        if not isinstance(feature, dict) or not all(
            isinstance(feature.get(k), str) and feature.get(k)
            for k in ("name", "definition", "rationale")
        ):
            return ShapeError(
                f"feature[{i}] must be an object with non-empty string 'name', 'definition', "
                f"and 'rationale'"
            )
    return None


# ----------------------------------------------------- the verifier package: two Submission gates


def build_feature_engineering_verifier(
    runner: CodeRunner,
    *,
    agent_train_csv: bytes,
    reserved_test_csv: bytes,
    reserved_labels: Mapping[str, str],
    complexity_penalty: float = 0.01,
    margin: float = 0.0,
    trial_deflation: float = 0.0,
    timeout_s: float = 900.0,
) -> SdkVerifier:
    """The opaque verifier package for the §12 domain — the two ``Submission`` gates (ADR 0001).

    The gate trains the submitted script on ``agent_train_csv`` and predicts ``reserved_test_csv``,
    then scores **balanced accuracy** against ``reserved_labels`` — which it **holds and never
    exposes**, so a leaked-target feature inflates the agent's own score but is caught here (§12,
    §13.11). The cheap **runs-clean** gate (→ ``tentative``) requires the script to execute and
    predict every reserved row; the hard **selection** gate (→ ``accepted``) requires the score net
    of a per-feature ``complexity_penalty`` to beat the incumbent by ``margin``, with the bar raised
    ``trial_deflation`` per prior rejected submission (the trial count is the rejected-log, §12).

    The incumbents' scores come from the verifier's own measurement ledger (it scored them), keyed
    by the store-slice's *current* accepted ids — so independence holds (the slice carries status,
    the verifier the numbers it computed; no rationale or score round-trips through the store).
    """
    gates = _SubmissionGates(
        runner=runner,
        agent_train_csv=agent_train_csv,
        reserved_test_csv=reserved_test_csv,
        reserved_labels=dict(reserved_labels),
        complexity_penalty=complexity_penalty,
        margin=margin,
        trial_deflation=trial_deflation,
        timeout_s=timeout_s,
    )
    return SdkVerifier(
        identity=FE_VERIFIER_IDENTITY,
        pipelines={
            SUBMISSION: (
                GateStep("runs-clean", gates.runnable, is_hard=False),
                GateStep("features-defined", deterministic_check(_features_defined), is_hard=False),
                GateStep("selection", gates.selection, is_hard=True),
            ),
            # The harvested-Feature grounding gate: cheap, structural, so "no implicit accept" holds
            # for Features too (§5.7, §13.6). Clearing it is the only check, so it earns acceptance.
            FEATURE: (
                GateStep("grounding", deterministic_check(_grounds_feature), is_hard=False),
            ),
        },
    )


def _features_defined(request: VerifierRequest) -> CheckOutcome:
    """Cheap, static refine: every declared feature must actually appear in the submitted code.

    A described-but-absent feature is a *localized* defect — most of the submission may be fine — so
    this returns ``refine`` naming the offending feature(s) (§7.5b, §12), recoverable by a revision.
    """
    code = request.objects.get(ENTRYPOINT)
    if code is None:
        return CheckOutcome(ok=False, rationale=f"no {ENTRYPOINT!r} to ground features against")
    payload = request.proposal.payload
    features = payload.get("features", []) if isinstance(payload, dict) else []
    absent: list[str] = []
    for feature in features if isinstance(features, list) else []:
        name = feature.get("name") if isinstance(feature, dict) else None
        if isinstance(name, str) and name.encode() not in code:
            absent.append(name)
    if absent:
        return CheckOutcome(
            ok=False,
            rationale=f"declared features not defined in the code: {', '.join(absent)}",
            defects=tuple(absent),
        )
    return CheckOutcome(ok=True, rationale="every declared feature appears in the code")


def _grounds_feature(request: VerifierRequest) -> CheckOutcome:
    """Cheap grounding (§12): a harvested ``Feature`` must be genuinely defined by the code."""
    code = request.objects.get(ENTRYPOINT)
    payload = request.proposal.payload
    name = payload.get("name") if isinstance(payload, dict) else None
    if not isinstance(name, str) or not name:
        return CheckOutcome(ok=False, rationale="feature has no name to ground")
    if code is None or name.encode() not in code:
        return CheckOutcome(
            ok=False, rationale=f"feature {name!r} is described but not defined in the code"
        )
    return CheckOutcome(ok=True, rationale=f"feature {name!r} is defined in the code")


@dataclass
class _SubmissionGates:
    """The two ``Submission`` gates over one shared, cached run of the submitted script (§12).

    Both gates execute the *same* run (train on the agent's data, predict the reserved features), so
    a single submission trains once: ``runs-clean`` checks it executed and is well-formed,
    ``selection`` scores it. ``_scores`` is the verifier's measurement ledger (proposal id → net
    score) used to read incumbents' scores back.
    """

    runner: CodeRunner
    agent_train_csv: bytes
    reserved_test_csv: bytes
    reserved_labels: Mapping[str, str]
    complexity_penalty: float
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
            tail = (result.stderr.strip().splitlines() or ["no stderr"])[-1]
            return GateVerdict(VerdictKind.REJECT, f"submission exited {result.exit_code}: {tail}")
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
        """Hard rung: balanced accuracy net of complexity beats the incumbent + deflated margin."""
        result = await self._run(request)
        preds = _parse_predictions(result.output) if result is not None else None
        if preds is None or (self.reserved_labels.keys() - preds.keys()):
            return GateVerdict(VerdictKind.REJECT, "submission produced no scorable predictions")
        raw = balanced_accuracy(self.reserved_labels, preds)
        n_features = _declared_feature_count(request.proposal)
        net = raw - self.complexity_penalty * n_features
        self._scores[request.proposal.id] = net

        best_id, baseline = self._best_incumbent(request.store_slice)
        trials = sum(1 for a in request.store_slice if a.status is ArtifactStatus.REJECTED)
        bar = baseline + self.margin + self.trial_deflation * trials
        detail = (
            f"balanced-accuracy {raw:.4f}, net {net:.4f} ({n_features} features) vs bar {bar:.4f} "
            f"(incumbent {baseline:.4f}, {trials} prior trials)"
        )
        if net > bar:
            return GateVerdict(
                VerdictKind.ACCEPT, f"improves: {detail}", score=net, supersedes=best_id
            )
        return GateVerdict(VerdictKind.REJECT, f"does not improve: {detail}", score=net)

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


def _declared_feature_count(artifact: Artifact) -> int:
    payload = artifact.payload
    features = payload.get("features") if isinstance(payload, dict) else None
    return len(features) if isinstance(features, list) else 0


FEATURE_ENGINEERING_INSTRUCTIONS = f"""\
You are a feature-engineering agent. Each cycle you produce ONE submission: a self-contained Python
script that cleans the data, engineers 1-{MAX_FEATURES} new features, trains a model, and predicts.

The script contract (the gate runs your script; honour it exactly):
- Resolve the data directory as `os.environ.get("VERITY_DATA", "data")` and the output directory as
  `os.environ.get("VERITY_OUT", "out")`. (The defaults work in your sandbox; the gate sets these.)
- Read the labelled training data from `<VERITY_DATA>/{TRAIN_INPUT}` and the unlabelled rows to
  predict from `<VERITY_DATA>/{TEST_INPUT}`. The target is `class`; `{TEST_INPUT}` has no `class`.
- Clean the data and engineer 1-{MAX_FEATURES} new features, train a model on the training data,
  then predict a `class` for every row of `{TEST_INPUT}`.
- Write predictions to `<VERITY_OUT>/{PREDICTIONS_OUTPUT}` with exactly two columns: `id,class`.
- The script must be deterministic (fix every random seed) and self-contained.

What to deliver to `outbox/` each cycle:
- `{ENTRYPOINT}` — the script above.
- `{REQUIREMENTS}` — the pinned package list your script needs (e.g. `pandas==2.2.2`), one per line.
  The gate installs exactly these before running your script, so pin versions for reproducibility.
- The proposal payload (via your submit/revises tool): `entrypoint` = "{ENTRYPOINT}",
  `requirements` = "{REQUIREMENTS}", and `features` = a list of 1-{MAX_FEATURES} objects, each
  with a `name`, a `definition` (how it is computed), and a `rationale` (why it should help).

How you are judged (you never see the judge's data):
- The gate runs your script on data you cannot see (a reserved hold-out), scores it on BALANCED
  ACCURACY, and only accepts a submission that improves on the best prior one net of complexity. A
  feature that leaks the target will look great on your own split but fail on the reserved set.
- If most features help but one is harmful or leaks, you get `refine` feedback naming the bad
  feature; produce a tracked revision that fixes exactly that feature.

Prior accepted submissions (if any) are provided under `scratch/provided/` — read them and build on
the best one rather than starting from scratch. You may edit anything in your workspace freely.

Speed and focus (IMPORTANT — the gate runs your script under a time budget; a submission that does
not finish in time is rejected):
- Your edge comes from FEATURE ENGINEERING, not from model size or tuning. Spend your effort
  designing and refining features, not searching for a bigger model.
- Train exactly ONE fast, modestly-sized model with fixed, sensible defaults (e.g. a small
  GradientBoosting / a LightGBM with default-ish settings, or even LogisticRegression). Do NOT run
  hyperparameter search, grid/random search, cross-validation sweeps, or large ensembles.
- Keep iterations quick: write the script, run it once to confirm it works, then submit. Don't
  repeatedly retrain to chase tiny gains — improve the FEATURES and resubmit instead.

Useful domain knowledge: differences between photometric bands ("colour indices", e.g. u-g, g-r,
r-i, i-z) and `redshift` carry most of the signal; encode the categorical `spectral_type`; and
because scoring is balanced accuracy, handle class imbalance (class weights / resampling)."""
