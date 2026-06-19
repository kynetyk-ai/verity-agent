"""Deterministic, stratified train/reserved split for the feature-engineering task (ROADMAP §12).

Splits a labelled CSV into the three pieces §12 needs:

* ``agent_train_csv`` — the labelled rows handed to the agent (mounted read-only into ``data/``);
* ``reserved_test_csv`` — the **reserved** rows with the target column dropped, handed to the gate
  as the unlabelled set the submitted script must predict;
* ``reserved_labels`` — the reserved rows' true targets, **held by the gate** and never exposed, so
  a feature that leaks the target is caught on this set (§12 leakage-safety, §13.11).

Stratified by the target so balanced accuracy is well-defined on both sides, and fully deterministic
(no RNG — reproducible verdicts, §5.8). Pure stdlib (``csv``) so the harness needs no pandas; the
*agent's* script installs its own libraries in-container. Lives in ``composition`` so the FE-Kaggle
catalog builder (`verity.composition.fe_kaggle`) can prepare its split when configuring the task.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

__all__ = ["DatasetSplit", "stratified_split", "subsample"]


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """The three pieces of a stratified split (§12). The reserved labels never reach the agent."""

    agent_train_csv: bytes
    reserved_test_csv: bytes
    reserved_labels: dict[str, str]
    header: tuple[str, ...]
    target: str
    id_column: str


def stratified_split(
    data: bytes,
    *,
    target: str = "class",
    id_column: str = "id",
    reserved_fraction: float = 0.6,
) -> DatasetSplit:
    """Split ``data`` (CSV bytes) into agent-train + a larger reserved set, stratified by target.

    ``reserved_fraction`` is the share of each class routed to the reserved set (default 0.6 — the
    reserved set is deliberately *larger* than the agent's, §12). Raises ``ValueError`` if the
    target or id column is absent or the fraction is out of ``(0, 1)``.
    """
    if not 0.0 < reserved_fraction < 1.0:
        raise ValueError(f"reserved_fraction must be in (0, 1), got {reserved_fraction}")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    header = tuple(reader.fieldnames or ())
    if target not in header:
        raise ValueError(f"target column {target!r} not in header {header}")
    if id_column not in header:
        raise ValueError(f"id column {id_column!r} not in header {header}")

    rows = list(reader)
    per_class_seen: dict[str, int] = {}
    agent_rows: list[dict[str, str]] = []
    reserved_rows: list[dict[str, str]] = []
    for row in rows:
        label = row[target]
        i = per_class_seen.get(label, 0)
        per_class_seen[label] = i + 1
        # Even interleave: route this row to reserved when the running count of reserved rows for
        # this class would advance — spreads the reserved share uniformly through the class.
        to_reserved = int((i + 1) * reserved_fraction) != int(i * reserved_fraction)
        (reserved_rows if to_reserved else agent_rows).append(row)

    test_header = tuple(c for c in header if c != target)
    return DatasetSplit(
        agent_train_csv=_write_csv(header, agent_rows),
        reserved_test_csv=_write_csv(test_header, reserved_rows),
        reserved_labels={row[id_column]: row[target] for row in reserved_rows},
        header=header,
        target=target,
        id_column=id_column,
    )


def subsample(data: bytes, *, per_class: int, target: str = "class") -> bytes:
    """Keep up to ``per_class`` rows per target class — a fast, stratified slice for a live run."""
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    header = tuple(reader.fieldnames or ())
    seen: dict[str, int] = {}
    kept: list[dict[str, str]] = []
    for row in reader:
        label = row[target]
        if seen.get(label, 0) < per_class:
            seen[label] = seen.get(label, 0) + 1
            kept.append(row)
    return _write_csv(header, kept)


def _write_csv(header: tuple[str, ...], rows: list[dict[str, str]]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(header), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")
