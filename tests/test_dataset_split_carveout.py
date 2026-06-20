"""The user-side data-prep derivation: the stratified split + answer-key (ADR 0005 / §9.2).

Since #74 the dataset split + ``reserved_labels`` derivation is the **user's** responsibility, done
outside Verity by ``tools/prepare_fe_data.py`` over the ``tools.harness.dataset`` helper — the
control plane does no splitting (it routes opaque role-keyed blobs). These tests pin the derivation
itself — the piece a bad split or an answer-key leak would corrupt — independent of the control
plane: agent and reserved rows are disjoint and exhaustive, ``reserved_labels`` exactly matches the
reserved rows' true targets, the target column is dropped from the reserved set the agent's script
will predict, and the split is deterministic (reproducible verdicts, §5.8). A final test pins the
**prep boundary**: the prepared agent bundle never carries the answer key (§3.5 isolation by
construction).
"""

from __future__ import annotations

import csv
import io

from tools.harness.dataset import stratified_split, subsample
from tools.prepare_fe_data import prepare


def _make_csv(n_per_class: int, classes: tuple[str, ...]) -> bytes:
    """A labelled CSV with globally-unique ids, so set-membership checks are unambiguous."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["id", "a", "class"])
    writer.writeheader()
    rid = 0
    for cls in classes:
        for _ in range(n_per_class):
            writer.writerow({"id": str(rid), "a": str(rid * 2), "class": cls})
            rid += 1
    return buf.getvalue().encode()


def _ids(data: bytes) -> list[str]:
    return [r["id"] for r in csv.DictReader(io.StringIO(data.decode()))]


def test_split_is_disjoint_and_exhaustive() -> None:
    raw = _make_csv(5, ("STAR", "GALAXY", "QSO", "AGN"))
    split = stratified_split(raw, target="class", id_column="id", reserved_fraction=0.5)
    agent_ids, reserved_ids = set(_ids(split.agent_train_csv)), set(_ids(split.reserved_test_csv))
    assert agent_ids.isdisjoint(reserved_ids), "a row leaked into both the agent and reserved sets"
    assert agent_ids | reserved_ids == set(_ids(raw)), "the split lost or duplicated rows"


def test_reserved_labels_match_truth_and_target_is_dropped() -> None:
    raw = _make_csv(5, ("STAR", "GALAXY", "QSO", "AGN"))
    truth = {r["id"]: r["class"] for r in csv.DictReader(io.StringIO(raw.decode()))}
    split = stratified_split(raw, target="class", id_column="id", reserved_fraction=0.5)

    assert set(split.reserved_labels) == set(_ids(split.reserved_test_csv))
    assert all(split.reserved_labels[i] == truth[i] for i in split.reserved_labels)
    # the answer key is dropped from what the agent's script sees, so a feature cannot read it back.
    reserved_header = next(csv.reader(io.StringIO(split.reserved_test_csv.decode())))
    assert "class" not in reserved_header


def test_stratified_proportions_per_class() -> None:
    # reserved_fraction 0.5 over 4 rows/class -> 2 reserved, 2 agent per class (even interleave).
    raw = _make_csv(4, ("STAR", "GALAXY"))
    split = stratified_split(raw, target="class", id_column="id", reserved_fraction=0.5)
    # reserved_test_csv has no class column, so count via the answer key (one per reserved row).
    per_class: dict[str, int] = {}
    for cls in split.reserved_labels.values():
        per_class[cls] = per_class.get(cls, 0) + 1
    assert per_class == {"STAR": 2, "GALAXY": 2}


def test_split_is_deterministic() -> None:
    raw = _make_csv(5, ("STAR", "GALAXY", "QSO"))
    a = stratified_split(raw, target="class", id_column="id", reserved_fraction=0.6)
    b = stratified_split(raw, target="class", id_column="id", reserved_fraction=0.6)
    assert a.agent_train_csv == b.agent_train_csv
    assert a.reserved_test_csv == b.reserved_test_csv
    assert a.reserved_labels == b.reserved_labels


def test_subsample_caps_per_class() -> None:
    raw = _make_csv(10, ("STAR", "GALAXY", "QSO"))
    capped = subsample(raw, per_class=3, target="class")
    counts: dict[str, int] = {}
    for r in csv.DictReader(io.StringIO(capped.decode())):
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    assert counts == {"STAR": 3, "GALAXY": 3, "QSO": 3}


def test_prep_isolates_the_answer_key_in_the_verifier_bundle(tmp_path: object) -> None:
    """ADR 0005: the user-side prep places the answer key **only** in the verifier role bundle.

    Isolation by construction at the prep boundary — the agent bundle the control plane will route
    to the sandbox carries no labels file, and the agent's ``train.csv`` has the target dropped from
    no rows it shares with the hold-out (the hold-out rows are simply absent). The verifier's
    ``holdout_labels.csv`` matches the reserved rows' true targets.
    """
    import pathlib

    raw = _make_csv(6, ("STAR", "GALAXY"))
    real_test = b"id,a\n100,200\n101,202\n"
    out = pathlib.Path(str(tmp_path))
    written = prepare(
        train=raw, test=real_test, out=out,
        target="class", id_column="id", per_class=100, reserved_fraction=0.5,
    )

    # The agent bundle never contains the answer key.
    assert "holdout_labels.csv" not in written["agent"]
    assert set(written["agent"]) == {"train.csv", "test.csv"}
    assert "holdout_labels.csv" in written["verifier"]

    # The verifier's answer key matches truth for exactly the reserved rows; the agent's train rows
    # are disjoint from the hold-out rows (the leak surface is empty by construction).
    truth = {r["id"]: r["class"] for r in csv.DictReader(io.StringIO(raw.decode()))}
    labels = {
        r["id"]: r["class"]
        for r in csv.DictReader(io.StringIO((out / "verifier" / "holdout_labels.csv").read_text()))
    }
    assert labels and all(labels[i] == truth[i] for i in labels)
    agent_train_ids = set(_ids((out / "agent" / "train.csv").read_bytes()))
    assert agent_train_ids.isdisjoint(labels), "a hold-out row leaked into the agent's train set"
