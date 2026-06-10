"""Re-export of the stratified-split helper, now packaged in ``verity.composition.dataset``.

Kept as a stable import path for the dev/benchmark tooling and tests that referenced
``tools.harness.dataset``; the implementation moved into the installed package so the containerized
FE entrypoint can use it.
"""

from __future__ import annotations

from verity.composition.dataset import DatasetSplit, stratified_split, subsample

__all__ = ["DatasetSplit", "stratified_split", "subsample"]
