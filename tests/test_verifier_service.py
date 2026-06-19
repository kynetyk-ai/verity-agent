"""The verifier service entrypoint (ROADMAP Phase 7.1 → 9.1). Construct-from-env only — no socket.

``build_server_from_env`` is the dependency-wiring the service image runs; the serving (uvicorn /
FastAPI) is lazy inside ``main``, so this needs neither the ``service`` extra nor a network. A
dataless verifier (``fake``) is served as a fixed impl; a data-bearing one (``fe-kaggle``) is served
as a *builder* that constructs the impl from the per-task setup payload (§9.1).
"""

from __future__ import annotations

import pytest

from verity.verifier.__main__ import build_server_from_env


def test_defaults_to_fake_fixed_impl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERITY_VERIFIER", raising=False)
    server = build_server_from_env()
    # A dataless verifier is ready immediately: a fixed impl, no builder.
    assert server._impl is not None and server._impl.identity == "fake-verifier"
    assert server._builder is None


def test_fe_kaggle_is_a_builder_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERITY_VERIFIER", "fe-kaggle")
    server = build_server_from_env()
    # A data-bearing verifier waits for the per-task `setup`: a builder, no impl yet.
    assert server._impl is None
    assert server._builder is not None


def test_unknown_verifier_name_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERITY_VERIFIER", "nope")
    with pytest.raises(SystemExit, match="unknown VERITY_VERIFIER"):
        build_server_from_env()
