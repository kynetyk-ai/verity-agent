"""Deterministic stratified train/reserved split — a **user-side** data-prep helper (ADR 0005).

This is the §12 feature-engineering split, kept as a *dev/prep tool* and imported by **no Verity
service**. Since ROADMAP 9.2 / #74 data preparation is the user's responsibility, performed outside
the control plane: the CP routes opaque, role-keyed blobs and interprets no dataset semantics (no
``target``/``id_column``, no answer-key derivation). This helper is what a user runs to *produce*
those role-keyed inputs (see ``tools/prepare_fe_data.py``); it deliberately lives under ``tools/``,
not in the installed ``verity`` package, so a guard test can prove the control-plane image carries
no split logic.

It splits a labelled CSV into the three pieces §12 needs:

* ``agent_train_csv`` — the labelled rows handed to the agent (the agent's training set);
* ``reserved_test_csv`` — the **reserved** rows with the target column dropped, the unlabelled set
  the submitted script must predict for the cheap local proxy;
* ``reserved_labels`` — the reserved rows' true targets, the answer key the proxy scores against,
  routed only to the verifier role so a feature that leaks the target is caught (§12, §13.11).

Stratified by the target so balanced accuracy is well-defined on both sides, fully deterministic (no
RNG — reproducible, §5.8), pure stdlib (``csv``) so prep needs no pandas.
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


def subsample(data: bytes, *, per_class: int | None, target: str = "class") -> bytes:
    """Keep up to ``per_class`` rows per target class — a fast, stratified slice for a live run.

    ``per_class=None`` keeps **all** rows: the full competitive train (the data is the bottleneck
    for reaching a top-N% leaderboard score, so a serious attempt trains on everything). Rows are
    always re-written through the same writer, so the full-train bytes are byte-stable with the
    split path.
    """
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    header = tuple(reader.fieldnames or ())
    seen: dict[str, int] = {}
    kept: list[dict[str, str]] = []
    for row in reader:
        label = row[target]
        if per_class is None or seen.get(label, 0) < per_class:
            seen[label] = seen.get(label, 0) + 1
            kept.append(row)
    return _write_csv(header, kept)


def _write_csv(header: tuple[str, ...], rows: list[dict[str, str]]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(header), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")
