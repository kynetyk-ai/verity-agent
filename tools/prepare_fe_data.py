"""Prepare role-keyed FE-Kaggle inputs from a raw labelled train + unlabelled test (ADR 0005).

**User-side data prep.** Since ROADMAP 9.2 / #74 the control plane does no dataset splitting; the
*user* prepares the per-role inputs and the CP routes them as opaque blobs. This script is the
reference prep for the `fe-kaggle` task: it takes the competition's ``train.csv`` (labelled) and
``test.csv`` (the real, unlabelled leaderboard set) and writes two self-contained role bundles:

    <out>/agent/                          what the agent's sandbox sees
        train.csv        the agent's training set (a stratified subsample, hold-out removed)
        test.csv         the real, unlabelled Kaggle test set
    <out>/verifier/                       what the verifier (gates) sees
        train.csv        the SAME agent training set — the cheap proxy re-runs the script on it
        holdout.csv      the reserved rows, target column dropped (the cheap proxy's eval set)
        holdout_labels.csv   the reserved rows' true targets — the answer key (id,<target>)
        full_train.csv   the full labelled subsample — the hard Kaggle gate trains on this
        test.csv         the real, unlabelled Kaggle test set

The answer key (``holdout_labels.csv``) is written **only** under ``verifier/`` — it is never placed
in the agent bundle, so the §3.5 isolation invariant holds by construction at the prep boundary, not
by any control-plane carving. ``agent/train.csv`` and ``verifier/train.csv`` are byte-identical (and
likewise the two ``test.csv``), so the content-addressed store dedupes them.

Usage (competitive defaults — full train, ~15% stratified hold-out):
    uv run python tools/prepare_fe_data.py \
        --train feature-engineering-test/train.csv \
        --test  feature-engineering-test/test.csv \
        --out   feature-engineering-test
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


def prepare(
    *,
    train: bytes,
    test: bytes,
    out: Path,
    target: str = "class",
    id_column: str = "id",
    per_class: int | None = None,
    reserved_fraction: float = 0.15,
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

    files: dict[str, dict[str, bytes]] = {
        "agent": {"train.csv": split.agent_train_csv, "test.csv": test},
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
    args = p.parse_args()

    written = prepare(
        train=args.train.read_bytes(),
        test=args.test.read_bytes(),
        out=args.out,
        target=args.target,
        id_column=args.id_column,
        per_class=args.per_class,
        reserved_fraction=args.reserved_fraction,
    )
    for role, named in written.items():
        for name, path in named.items():
            print(f"{role}/{name} -> {path}")


if __name__ == "__main__":
    main()
