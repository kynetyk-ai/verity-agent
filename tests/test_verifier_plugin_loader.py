"""Unit tests for the verifier-type plugin loader (ROADMAP 9.3, ADR 0006).

The verifier image discovers its verifiers from the ``verity.verifier_types`` group; these pass fake
entry points (no install) to pin both shapes (dataless ``impl`` + data-bearing ``setup``), the
``VERITY_VERIFIER`` selection via ``build_server_from_env``, and degrade-don't-crash.
"""

from __future__ import annotations

import structlog

from verity.verifier.registry import VerifierRegistry, load_verifier_plugins


class FakeEntryPoint:
    def __init__(self, name, loader, *, value="fake_pkg:register_verifier", dist_name=""):
        self.name = name
        self.value = value
        self._loader = loader
        self.dist = type("Dist", (), {"name": dist_name})()

    def load(self):
        return self._loader()


def _source(*eps):
    return lambda: list(eps)


class _FakePort:  # a stand-in VerifierPort (the loader/registry never calls into it here)
    pass


def test_discovers_impl_and_setup_verifiers() -> None:
    def register(registry):
        registry.register_impl("my-impl", lambda: _FakePort())
        registry.register_setup("my-setup", lambda setup: _FakePort())

    reg = load_verifier_plugins(
        VerifierRegistry(), entry_points=_source(FakeEntryPoint("v", lambda: register))
    )
    assert reg.names() == ["my-impl", "my-setup"]
    assert reg.impl("my-impl") is not None
    assert reg.setup("my-setup") is not None
    assert reg.impl("my-setup") is None  # a setup verifier is not an impl


def test_load_failure_is_skipped_and_logged() -> None:
    def boom():
        raise ImportError("no module")

    with structlog.testing.capture_logs() as logs:
        reg = load_verifier_plugins(
            VerifierRegistry(), entry_points=_source(FakeEntryPoint("b", boom))
        )
    assert reg.names() == []
    assert any(e["event"] == "verifier_plugin_load_failed" for e in logs)


def test_collision_keeps_incumbent() -> None:
    base = VerifierRegistry()
    base.register_impl("fake", lambda: _FakePort())  # the incumbent

    def register(registry):
        registry.register_setup("fake", lambda setup: _FakePort())  # tries to shadow

    with structlog.testing.capture_logs() as logs:
        reg = load_verifier_plugins(
            base, entry_points=_source(FakeEntryPoint("s", lambda: register))
        )
    assert reg.impl("fake") is not None  # incumbent (the impl) survives
    assert reg.setup("fake") is None  # the shadowing setup was skipped
    assert any(e["event"] == "verifier_plugin_collision" and e["type"] == "fake" for e in logs)


def test_build_server_selects_discovered_verifier(monkeypatch) -> None:
    from verity.verifier import __main__ as entry

    def register(registry):
        registry.register_impl("my-impl", lambda: _FakePort())

    monkeypatch.setattr(
        entry,
        "load_verifier_plugins",
        lambda registry: load_verifier_plugins(
            registry, entry_points=_source(FakeEntryPoint("v", lambda: register))
        ),
    )
    monkeypatch.setenv("VERITY_VERIFIER", "my-impl")
    server = entry.build_server_from_env()
    assert server is not None


def test_build_server_unknown_name_exits_with_known_list(monkeypatch) -> None:
    import pytest

    from verity.verifier import __main__ as entry

    def register(registry):
        registry.register_impl("only-this", lambda: _FakePort())

    monkeypatch.setattr(
        entry,
        "load_verifier_plugins",
        lambda registry: load_verifier_plugins(
            registry, entry_points=_source(FakeEntryPoint("v", lambda: register))
        ),
    )
    monkeypatch.setenv("VERITY_VERIFIER", "nope")
    with pytest.raises(SystemExit) as exc:
        entry.build_server_from_env()
    assert "only-this" in str(exc.value)
