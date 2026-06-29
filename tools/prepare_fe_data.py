"""Prepare role-keyed FE-Kaggle inputs from a raw labelled train + unlabelled test (ADR 0005).

**User-side data prep.** Since ROADMAP 9.2 / #74 the control plane does no dataset splitting; the
*user* prepares the per-role inputs and the CP routes them as opaque blobs. This script is the
reference prep for the `fe-kaggle` task: it takes the competition's ``train.csv`` (labelled) and
``test.csv`` (the real, unlabelled leaderboard set) and writes two self-contained role bundles:

    <out>/agent/                          what the agent's sandbox sees
        train.csv        the agent's training set (a stratified subsample, hold-out removed)
        test.csv         a small UNLABELLED sample of the test (--agent-test-rows, default 200) — a
                         wiring/format check only; the agent estimates score via CV on train, the
                         gate regenerates on the full test (target stripped if present, I2)
    <out>/verifier/                       what the verifier (gates) sees
        train.csv        the SAME agent training set — the cheap proxy re-runs the script on it
        holdout.csv      the reserved rows, target column dropped (the cheap proxy's eval set)
        holdout_labels.csv   the reserved rows' true targets — the answer key (id,<target>)
        full_train.csv   the full labelled subsample — the hard Kaggle gate trains on this
        test.csv         the real, unlabelled Kaggle test set

The answer key (``holdout_labels.csv``) is written **only** under ``verifier/`` — it is never placed
in the agent bundle, so the §3.5 isolation invariant holds by construction at the prep boundary, not
by any control-plane carving. ``agent/train.csv`` and ``verifier/train.csv`` are byte-identical, so
the content-addressed store dedupes them. The agent's ``test.csv`` is a small sample, so it differs
from the verifier's full ``test.csv`` (pass ``--agent-test-rows 0`` / ``None`` to give the agent the
full test, the legacy behaviour).

Usage (competitive defaults — full train, ~15% stratified hold-out). Run as a **module** (it imports
``tools.harness``), not by path — ``python tools/prepare_fe_data.py`` fails with "No module named
'tools'":
    uv run python -m tools.prepare_fe_data \
        --train prototyping_datasci_test/train.csv \
        --test  prototyping_datasci_test/test.csv \
        --out   prototyping_datasci_test
    # add --per-class N for a quick smaller smoke; --reserved-fraction F to resize the hold-out
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from tools.harness.dataset import stratified_split, subsample


def _labels_csv(reserved_labels: dict[str, str], *, id_column: str, target: str) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([id_column, target])
    writer.writerows(reserved_labels.items())
    return buf.getvalue().encode("utf-8")


def _head_rows(csv_bytes: bytes, n: int) -> bytes:
    """Return the header + the first ``n`` data rows of ``csv_bytes`` (schema-preserving sample).

    The agent's ``test.csv`` is unlabelled, so it can only verify *wiring* (does my script run and
    write a well-formed predictions file over this schema?) — it cannot score on it. A small sample
    verifies wiring just as well as the full set, at a fraction of the cost, and keeps a capable
    agent from burning its whole budget predicting hundreds of thousands of rows. The real, full
    test stays in the verifier role (the fe-kaggle gate regenerates predictions on it to submit)."""
    rows = list(csv.reader(io.StringIO(csv_bytes.decode("utf-8"))))
    if len(rows) <= n + 1:
        return csv_bytes
    buf = io.StringIO()
    csv.writer(buf).writerows(rows[: n + 1])
    return buf.getvalue().encode("utf-8")


def _drop_target_column(csv_bytes: bytes, target: str) -> bytes:
    """Return ``csv_bytes`` with the ``target`` column removed if present (idempotent).

    Defensive (I2): the agent's ``test.csv`` must be unlabelled. A real competition test set already
    has no target, so this is a no-op; but if a user points ``--test`` at a labelled file (a re-used
    train, or a competition whose test ships the target), this strips the answer at the prep
    boundary rather than leaking it into the agent's inputs.
    """
    rows = list(csv.reader(io.StringIO(csv_bytes.decode("utf-8"))))
    if not rows or target not in rows[0]:
        return csv_bytes
    idx = rows[0].index(target)
    buf = io.StringIO()
    writer = csv.writer(buf)
    for row in rows:
        writer.writerow([value for i, value in enumerate(row) if i != idx])
    return buf.getvalue().encode("utf-8")


def prepare(
    *,
    train: bytes,
    test: bytes,
    out: Path,
    target: str = "class",
    id_column: str = "id",
    per_class: int | None = None,
    reserved_fraction: float = 0.15,
    agent_test_rows: int | None = 200,
) -> dict[str, dict[str, Path]]:
    """Write the agent + verifier role bundles under ``out``; return {role: {name: path}}.

    Competitive sizing by default: ``per_class=None`` trains on the **full** labelled train (the
    data is the bottleneck for a top-N% score) and ``reserved_fraction=0.15`` reserves a stratified
    hold-out — big enough that its balanced accuracy is a tight, well-calibrated estimate of the
    public score (the competitive gate compares against it). Pass a ``per_class`` cap for a quick
    manual smoke; the gate logic is identical, just noisier.
    """
    sub = subsample(train, per_class=per_class, target=target)
    split = stratified_split(
        sub, target=target, id_column=id_column, reserved_fraction=reserved_fraction
    )
    agent_dir, verifier_dir = out / "agent", out / "verifier"
    agent_dir.mkdir(parents=True, exist_ok=True)
    verifier_dir.mkdir(parents=True, exist_ok=True)

    # The agent gets a small, target-stripped SAMPLE of the test (wiring/format check only — it is
    # unlabelled, so it can't score on it; it estimates with CV on train). The full test stays in
    # the verifier role. ``agent_test_rows=None`` gives the agent the full test (legacy). The I2
    # target-strip is a no-op for a normal unlabelled test but never leaks labels if --test was
    # pointed at a labelled file.
    agent_test_src = _head_rows(test, agent_test_rows) if agent_test_rows is not None else test
    agent_test = _drop_target_column(agent_test_src, target)
    files: dict[str, dict[str, bytes]] = {
        "agent": {"train.csv": split.agent_train_csv, "test.csv": agent_test},
        "verifier": {
            "train.csv": split.agent_train_csv,
            "holdout.csv": split.reserved_test_csv,
            "holdout_labels.csv": _labels_csv(
                split.reserved_labels, id_column=id_column, target=target
            ),
            "full_train.csv": sub,
            "test.csv": test,
        },
    }
    written: dict[str, dict[str, Path]] = {}
    for role, named in files.items():
        written[role] = {}
        for name, blob in named.items():
            path = (out / role / name)
            path.write_bytes(blob)
            written[role][name] = path
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True, help="raw labelled train.csv")
    p.add_argument("--test", type=Path, required=True, help="real unlabelled test.csv")
    p.add_argument("--out", type=Path, required=True, help="output dir (holds agent/ + verifier/)")
    p.add_argument("--target", default="class")
    p.add_argument("--id-column", default="id")
    p.add_argument(
        "--per-class", type=int, default=None,
        help="rows/class to keep (omit = full competitive train; set for a quick smoke)",
    )
    p.add_argument("--reserved-fraction", type=float, default=0.15)
    p.add_argument(
        "--agent-test-rows", type=int, default=200,
        help="rows of the test to give the AGENT (a wiring/format sample; it can't score on the "
             "unlabelled test — it estimates via CV on train). 0 = the full test (legacy).",
    )
    args = p.parse_args()

    written = prepare(
        train=args.train.read_bytes(),
        test=args.test.read_bytes(),
        out=args.out,
        target=args.target,
        id_column=args.id_column,
        per_class=args.per_class,
        reserved_fraction=args.reserved_fraction,
        agent_test_rows=args.agent_test_rows or None,
    )
    for role, named in written.items():
        for name, path in named.items():
            print(f"{role}/{name} -> {path}")


if __name__ == "__main__":
    main()
