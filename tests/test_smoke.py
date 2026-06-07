"""Smoke tests — prove the scaffold imports and day-one logging works (ROADMAP Phase 0)."""

from __future__ import annotations

import verity
from verity.logging import configure_logging, get_logger


def test_package_imports_and_has_version() -> None:
    assert isinstance(verity.__version__, str)
    assert verity.__version__


def test_logging_configures_and_binds() -> None:
    configure_logging()
    log = get_logger("verity.tests")
    # Binding context and emitting must not raise once configured.
    log.bind(check="smoke").info("scaffold_ok")
