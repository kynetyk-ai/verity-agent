"""The feature-engineering domain — the v1 validation target and MVP (spec §12, Phase 4).

A feature-engineering agent is given a dataset (one named column is the target ``class``, the rest
are candidate features) and asked to **clean the data and design 1–5 new features that improve a
model**. Each cycle its deliverable is a **self-contained script** (cleans, engineers features,
trains, predicts) plus a **JSON description** of the features it added — both written to
``outbox/``, from which the harness harvests the code object (§3.4). It never returns a trained
model: the model is produced by the gate, from the script, on a **reserved dataset the agent never
sees** (§12) — so a feature that peeks at the target inflates the agent's own score but fails to
generalize, and is caught on the reserved set.

This module is the **control-plane side**: the schema, the gated-type coverage, the proposal-shape
spec, and the domain instructions. The **verifier side** — the two ``Submission`` gates (runnable →
``tentative``; selection-on-the-reserved-set → ``accepted``) and the ``Feature`` grounding gate —
lands in Phase 4.2/4.3 as the opaque verifier package selected by ``verifier_key``.

Schema (§12):
* ``DatasetVersion`` — the pinned dataset (the CSV, the named target, fixed folds); the larger
  reserved verification set is held by the gate, never exposed here. Root artifact.
* ``Submission`` — the agent's per-cycle proposal and the gated artifact: an object-bearing payload
  referencing the script + its declared package list, plus the structured feature description.
* ``Feature`` — harvested by the harness from the submission's description, one typed artifact per
  declared feature (Phase 4.3), linked to its ``Submission`` by a ``harvest`` operation (§4.1).
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.contracts import Artifact
from verity.control_plane.commit import ShapeError, ShapeValidator
from verity.control_plane.independence import StoreInput
from verity.control_plane.registries import (
    ArtifactTypeDef,
    GatedTypeRegistry,
    OperationSignature,
    SchemaRegistry,
)

__all__ = [
    "DATASET_VERSION",
    "SUBMISSION",
    "FEATURE",
    "ENTRYPOINT",
    "REQUIREMENTS",
    "MAX_FEATURES",
    "FE_VERIFIER_IDENTITY",
    "FEATURE_ENGINEERING_INSTRUCTIONS",
    "FeatureEngineeringDomain",
    "build_feature_engineering_domain",
]

DATASET_VERSION = "DatasetVersion"
SUBMISSION = "Submission"
FEATURE = "Feature"

# The reserved object names the agent writes to outbox/ (the script + its declared package list).
ENTRYPOINT = "submission.py"
REQUIREMENTS = "requirements.txt"

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


FEATURE_ENGINEERING_INSTRUCTIONS = f"""\
You are a feature-engineering agent. Each cycle you produce ONE submission: a self-contained Python
script that cleans the data, engineers 1-{MAX_FEATURES} new features, trains a model, and predicts.

The script contract (the gate runs your script; honour it exactly):
- Read the labelled training data from `data/train.csv` and the unlabelled rows to predict from
  `data/test.csv`. The target column is `class`; `test.csv` has no `class` column.
- Clean the data and engineer 1-{MAX_FEATURES} new features, train a model on the training data,
  then predict a `class` for every row of `data/test.csv`.
- Write predictions to `out/predictions.csv` with exactly two columns: `id,class`.
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

Useful domain knowledge: differences between photometric bands ("colour indices", e.g. u-g, g-r,
r-i, i-z) and `redshift` carry most of the signal; encode the categorical `spectral_type`; and
because scoring is balanced accuracy, handle class imbalance (class weights / resampling)."""
