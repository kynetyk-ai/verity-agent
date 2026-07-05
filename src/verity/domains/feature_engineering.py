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
from verity.logging import get_logger
from verity.verifier import (
    CodeRunner,
    GateStep,
    RunRequest,
    RunResult,
    SdkVerifier,
)

log = get_logger("verity.domains.feature_engineering")

__all__ = [
    "DATASET_VERSION",
    "SUBMISSION",
    "ACCEPT_IMPROVE_OVER_BEST_PRIOR",
    "ACCEPT_ALWAYS",
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
    "parse_label_csv",
]

DATASET_VERSION = "DatasetVersion"
SUBMISSION = "Submission"

# The verdict-policy axis of the holdout scorer (the ablation knob, experimental-design §6.2). Both
# policies share the same scoring run; only the accept verdict differs.
#   ``improve_over_best_prior`` — ACCEPT iff the score beats the best prior, else REJECT (the §12
#     ``selection`` behaviour). Under a binary accept/reject gate that never refines, "best
#     accepted" and "best prior of any status" coincide, so the existing best-accepted baseline is
#     exact for the ablation (Exp 3/4/4b).
#   ``always`` — run, record the score, ACCEPT unconditionally (Exp 1/2). A non-runnable / non-
#     scorable submission still rejects (there is no score to record) — the fair, non-hobbled
#     construct, not "accept the failure" (experimental-design §7).
ACCEPT_IMPROVE_OVER_BEST_PRIOR = "improve_over_best_prior"
ACCEPT_ALWAYS = "always"
_ACCEPT_POLICIES = (ACCEPT_IMPROVE_OVER_BEST_PRIOR, ACCEPT_ALWAYS)

# The reserved object names the agent writes to outbox/ (the script + its declared package list).
ENTRYPOINT = "submission.py"
REQUIREMENTS = "requirements.txt"

# The payload KEYS whose VALUES name those outbox objects. Shared by `declared_objects` (the harvest
# filter) and the operations' `object_payload_keys` (the propose tool's in-cycle presence check), so
# the two never drift.
_SUBMISSION_OBJECT_KEYS = ("entrypoint", "requirements")

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
        OperationSignature(
            "submit", inputs=(DATASET_VERSION,), output=SUBMISSION,
            object_payload_keys=_SUBMISSION_OBJECT_KEYS,
        )
    )
    # A revision of a prior submission enters via a 'revises' op (kernel-general, §6); this domain
    # never issues a refine, so it is unused here, but the op stays registered for lineage.
    schema.register_operation(
        OperationSignature(
            "revises", inputs=(SUBMISSION,), output=SUBMISSION,
            object_payload_keys=_SUBMISSION_OBJECT_KEYS,
        )
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
    candidates = tuple(artifact.payload.get(key) for key in _SUBMISSION_OBJECT_KEYS)
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
    accept_policy: str = ACCEPT_IMPROVE_OVER_BEST_PRIOR,
) -> SdkVerifier:
    """The opaque verifier package for the §12 domain — the two ``Submission`` gates (ADR 0001).

    The gate trains the submitted script on ``agent_train_csv`` and predicts ``reserved_test_csv``,
    then scores **balanced accuracy** against ``reserved_labels`` — which it **holds and never
    exposes**, so a leaked-target submission inflates the agent's own score but is caught here (§12,
    §13.11). The cheap **runs-clean** gate (→ ``tentative``) requires the script to execute and
    predict every reserved row.

    ``accept_policy`` selects the hard verdict step over that one shared scoring run (the ablation
    knob, experimental-design §6.2) — both policies score every submission identically:

    * ``improve_over_best_prior`` (default) — the hard **selection** gate (→ ``accepted``) requires
      the score to beat the incumbent by ``margin``, with the bar raised ``trial_deflation`` per
      prior rejected submission (the trial count is the rejected-log, §12). Reject-only — no
      ``refine``.
    * ``always`` — the hard **score-and-accept** gate records the score and accepts unconditionally
      (Exp 1/2); a non-scorable submission still rejects.

    The incumbents' scores come from the verifier's own measurement ledger (it scored them), keyed
    by the store-slice's *current* accepted ids — so independence holds (the slice carries status,
    the verifier the numbers it computed; no rationale or score round-trips through the store).
    """
    if accept_policy not in _ACCEPT_POLICIES:
        raise ValueError(
            f"unknown accept_policy {accept_policy!r}; expected one of {_ACCEPT_POLICIES}"
        )
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
    hard_step = (
        GateStep("score-and-accept", gates.score_and_accept, is_hard=True)
        if accept_policy == ACCEPT_ALWAYS
        else GateStep("selection", gates.selection, is_hard=True)
    )
    return SdkVerifier(
        identity=FE_VERIFIER_IDENTITY,
        context_sinks=sinks,
        pipelines={
            SUBMISSION: (
                GateStep("runs-clean", gates.runnable, is_hard=False),
                hard_step,
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
                f"submission exited {result.exit_code}: {_stderr_tail(result.stderr)}"
                f"{_dependency_hint(result.stderr)}",
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

        best_id, baseline = self._best_incumbent(request)
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

    async def score_and_accept(self, request: VerifierRequest) -> GateVerdict | None:
        """Permissive policy: score the run, record it, ACCEPT unconditionally (Exp 1/2).

        Records the score on the same ledger ``selection`` uses, so a run under this policy is
        directly comparable to one under ``improve_over_best_prior``. A submission that produced no
        scorable predictions still rejects — there is no number to record, and an unrunnable script
        is not an "accept the failure" (the fair, non-hobbled construct, experimental-design §7).
        Does not supersede: every accepted attempt is retained (the accumulating-context conditions
        read all priors), and supersession is the improving ladder's concern, not this one.
        """
        result = await self._run(request)
        preds = _parse_predictions(result.output) if result is not None else None
        if preds is None or (self.reserved_labels.keys() - preds.keys()):
            return GateVerdict(VerdictKind.REJECT, "submission produced no scorable predictions")
        score = balanced_accuracy(self.reserved_labels, preds)
        self._scores[request.proposal.id] = score
        return GateVerdict(
            VerdictKind.ACCEPT,
            f"balanced-accuracy {score:.4f} (always-accept: score recorded, accepted)",
            score=score,
        )

    def _best_incumbent(self, request: VerifierRequest) -> tuple[str | None, float]:
        """The top-scoring currently-accepted submission (status from the slice, score from the
        control plane's durable record, falling back to our in-process ledger).

        Reading the recorded score from ``request.scores`` first means the baseline survives a
        verifier restart mid-run: a fresh verifier process has an empty ``_scores`` ledger, but the
        control plane still holds every accepted incumbent's recorded balanced accuracy (G3). The
        in-process ledger remains as a fast-path fallback for the steady state.
        """
        scored: list[tuple[str, float]] = []
        restored = False
        for a in request.store_slice:
            if a.status is not ArtifactStatus.ACCEPTED:
                continue
            recorded = request.scores.get(a.id)
            if a.id in self._scores:
                scored.append((a.id, self._scores[a.id]))
            elif recorded:
                # No in-process score for an accepted incumbent → this verifier did not score it
                # (a restart, almost always). Recover the baseline from the control plane's record.
                scored.append((a.id, max(recorded.values())))
                restored = True
        if restored:
            log.warning("verifier_restart_detected", recovered_from="control_plane_scores")
        if not scored:
            return None, 0.0
        return max(scored, key=lambda pair: pair[1])


# Stderr markers of a dependency-shaped failure: missing imports and pip's resolution errors.
# Case-insensitive; deliberately narrow — a hint on a non-dependency failure is worse than none.
_DEPENDENCY_MARKERS = (
    "modulenotfounderror",
    "importerror",
    "no matching distribution found",
    "resolutionimpossible",
    "could not find a version",
    "error: pip",
)

_DEPENDENCY_HINT = (
    " Likely a sandbox↔gate environment mismatch: the gate installs EXACTLY what "
    f"{REQUIREMENTS!r} lists into a fresh container — nothing else. Pin the exact versions the "
    "script was actually tested with (`pip freeze` in the sandbox) for every import, and list "
    "every package the script needs."
)


def _dependency_hint(stderr: str) -> str:
    """The pin-your-deps remedy for a dependency-shaped failure, else ``""`` (#134).

    The reactive-hint principle: the domain instructions already say "pin recent versions", but
    the remedy at the MOMENT of failure is what breaks the silent retry loop — a script that ran
    in the agent's sandbox and died in the gate's clean container is a version/deps mismatch far
    more often than a code bug.
    """
    lowered = stderr.lower()
    if any(marker in lowered for marker in _DEPENDENCY_MARKERS):
        return _DEPENDENCY_HINT
    return ""


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


def parse_label_csv(data: bytes, *, id_column: str = "id", target: str = "class") -> dict[str, str]:
    """Parse a ``holdout_labels.csv`` (the answer key) into an ``{id: target}`` map (verifier-side).

    Domain knowledge that lives with the verifier: since ADR 0005 the answer key arrives as a
    routed verifier-role *file* (not a control-plane-derived dict), and the verifier image — which
    owns the domain — turns it back into the lookup the gates score against. Falls back to the first
    two columns when the header names differ, so a prep tool need not match the gate's column names.
    """
    reader = csv.reader(io.StringIO(data.decode("utf-8")))
    rows = list(reader)
    if not rows:
        return {}
    header = rows[0]
    try:
        id_idx, target_idx = header.index(id_column), header.index(target)
    except ValueError:
        id_idx, target_idx = 0, 1
    return {r[id_idx]: r[target_idx] for r in rows[1:] if len(r) > max(id_idx, target_idx)}


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
Produce a single self-contained Python script that reads the data, builds a predictive pipeline, and
predicts every test row. Any technique that fits in a single script and improves the held-out score
is fair game — features, model choice, an ensemble, calibration, imbalance handling. No method is
mandated.

### Script Contract
The script contract (the gate runs the script; honour it exactly):
- Data dir = `os.environ.get("VERITY_DATA", "data")`; output dir =
  `os.environ.get("VERITY_OUT", "out")` (the defaults work in the sandbox; the gate sets these).
- Read training data from `<VERITY_DATA>/{TRAIN_INPUT}` and the rows to predict from
  `<VERITY_DATA>/{TEST_INPUT}`. The target is `class`; `{TEST_INPUT}` has none.
- Train, then write predictions to `<VERITY_OUT>/{PREDICTIONS_OUTPUT}` with exactly two columns:
  `id,class`.
- Deterministic (fix every seed) and self-contained.

The `{TEST_INPUT}` in the sandbox is a small **unlabelled sample** — just enough to verify the
script runs end to end and writes a well-formed `{PREDICTIONS_OUTPUT}` over the real schema. It is
**not** representative and **cannot be scored** (no labels). Estimate balanced accuracy with
**stratified cross-validation on the training data**, and report that as
`ESTIMATED_BALANCED_ACCURACY`. The gate re-runs the exact script on a larger reserved set, so do not
spend the time budget predicting the sample repeatedly — get the pipeline right and submit.

### Dependencies
No packages are pre-installed — install them with `pip` during the session and test the script
end-to-end before submitting. Deliver a clean, pinned `{REQUIREMENTS}` alongside `{ENTRYPOINT}`: the
code-runner installs exactly it before running the script, so list **exactly** what `{ENTRYPOINT}`
imports (nothing unused — every listed package must install) and pin **recent** versions with
prebuilt py3.12 wheels. `{ENTRYPOINT}` must only `import` its libraries — never `pip install` from
inside it.

The runner (where the gate executes the script): a CPU-only Linux container, Python 3.12, ~16
cores, ~16 GB RAM, with the pinned `{REQUIREMENTS}` pip-installed (network is on for the install
only). scikit-learn, LightGBM, XGBoost, CatBoost, pandas, numpy all work. Do NOT use deep-learning
frameworks (torch / tensorflow won't finish) or libraries with no prebuilt wheel (no compiler in
the runner).

### Evaluation Criteria
- The proposal will be evaluated based on BALANCED ACCURACY.
- The proposed script is retrained on additional data not available in this environment; anything
  that peeks at the target fails on the hold-out and real data.
- The proposal should improve upon prior results, if prior submissions and their outcomes are
  provided.

### Process Instructions

Work the problem within the time allotted:
- If no prior submissions have been provided, explore the data with code to understand its
  structure, where the signal is, and non-obvious relationships (don't open the raw files directly;
  they're large — load and summarize them in a script).
- If prior submissions are provided, consider ways to optimize and/or combine the best approaches.
- If prior scores appear to have plateaued, consider new directions (e.g. new features, different
  models).

Train on ALL the data, and finish in time:
- TRAIN ON THE FULL training set — do NOT subsample. 
- USE ALL THE CORES: `n_jobs=-1` (scikit-learn / XGBoost / LightGBM), `thread_count=-1` (CatBoost).
- Optimize for BALANCED accuracy: validate with stratified cross-validation and handle class
imbalance (class weights / resampling / threshold tuning).
- Keep any ensemble or search bounded — a model that doesn't finish scores nothing. And note: some
libraries that manage their own thread pools don't co-exist cleanly in one process (CatBoost
alongside LightGBM/XGBoost is a known stall) — so to use several, train each in its own
process (`multiprocessing` / `ProcessPoolExecutor`) and combine predictions.

Deliver to `/work/outbox/` each cycle: `{ENTRYPOINT}` (the script) and `{REQUIREMENTS}` (its pinned
packages, e.g. `pandas==2.2.2`, one per line — the gate installs exactly these). The proposal
payload (via the submit/revises tool): `entrypoint` = "{ENTRYPOINT}", `requirements` =
"{REQUIREMENTS}". The submission is the unit — do not report individual features.

In the submit/revises tool's `rationale` (a private note — recorded for the audit trail, NEVER
shown to the gate), include a line exactly of the form `ESTIMATED_BALANCED_ACCURACY: <float>` with
an honest best estimate of the held-out balanced accuracy this submission will score (e.g.
`ESTIMATED_BALANCED_ACCURACY: 0.964`). It does not affect the verdict — report the real estimate,
not an optimistic one.

"""
