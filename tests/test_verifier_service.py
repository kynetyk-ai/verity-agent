"""The verifier service entrypoint (ROADMAP Phase 7.1). Construct-from-env only — no socket.

``build_verifier_from_env`` is the dependency-wiring the service image runs; the serving (uvicorn /
FastAPI) is lazy inside ``main``, so this needs neither the ``service`` extra nor a network.
"""

from __future__ import annotations

import pytest

from verity.verifier.__main__ import build_verifier_from_env


def test_build_verifier_from_env_defaults_to_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERITY_VERIFIER", raising=False)
    verifier = build_verifier_from_env()
    assert verifier.identity == "fake-verifier"


def test_build_verifier_from_env_selects_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERITY_VERIFIER", "fake")
    assert build_verifier_from_env().identity == "fake-verifier"


def test_unknown_verifier_name_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERITY_VERIFIER", "nope")
    with pytest.raises(SystemExit, match="unknown VERITY_VERIFIER"):
        build_verifier_from_env()
