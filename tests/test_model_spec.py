"""The provider-agnostic model seam (ROADMAP Phase 6.1/6.2).

Pure-function tests — no network, no Docker, no real model. The one test that touches the real
``langchain_openai.ChatOpenAI`` is guarded by ``importorskip`` (it lives only in the ``sandbox``
extra), mirroring the deepagents/pypdf skips elsewhere.
"""

from __future__ import annotations

import pytest

from verity.sandbox.model_spec import (
    ENV_API_KEY_ENV,
    ENV_BASE_URL,
    ENV_EXTRA,
    ENV_MODEL,
    ModelSpec,
    coerce_model,
    resolve_model,
)

_ANTHROPIC = ModelSpec("anthropic", "claude-sonnet-4-6")
_LOCAL = ModelSpec(
    "openai-compatible", "qwen2.5-coder",
    base_url="http://host.docker.internal:8000/v1", api_key_env="LOCAL_KEY",
    extra={"temperature": 0},
)


def test_from_provider_string_parses_provider_and_model() -> None:
    spec = ModelSpec.from_provider_string("anthropic:claude-sonnet-4-6")
    assert spec == _ANTHROPIC and spec.base_url is None


def test_from_provider_string_without_prefix_defaults_provider() -> None:
    spec = ModelSpec.from_provider_string("claude-sonnet-4-6")
    assert spec.provider == "anthropic" and spec.model == "claude-sonnet-4-6"


def test_anthropic_spec_env_is_backward_compatible() -> None:
    # The pre-Phase-6 behavior: exactly one env var, the bare provider string — nothing else, so the
    # existing docker command is byte-identical.
    assert _ANTHROPIC.to_env() == {ENV_MODEL: "anthropic:claude-sonnet-4-6"}


def test_model_spec_env_roundtrips() -> None:
    for spec in (_ANTHROPIC, _LOCAL):
        assert ModelSpec.from_env(spec.to_env()) == spec


def test_local_spec_emits_all_env_vars() -> None:
    env = _LOCAL.to_env()
    assert env[ENV_MODEL] == "openai-compatible:qwen2.5-coder"
    assert env[ENV_BASE_URL] == "http://host.docker.internal:8000/v1"
    assert env[ENV_API_KEY_ENV] == "LOCAL_KEY"
    assert env[ENV_EXTRA] == '{"temperature": 0}'


def test_from_env_defaults_to_anthropic_when_unset() -> None:
    assert ModelSpec.from_env({}) == _ANTHROPIC


def test_key_env_resolves_explicit_then_provider_default() -> None:
    assert _LOCAL.key_env() == "LOCAL_KEY"  # explicit wins
    assert _ANTHROPIC.key_env() == "ANTHROPIC_API_KEY"  # provider default
    assert ModelSpec("openai", "gpt-4o-mini").key_env() == "OPENAI_API_KEY"
    assert ModelSpec("openai-compatible", "llama").key_env() is None  # keyless


def test_needs_host_gateway_only_for_host_local_base_url() -> None:
    assert _LOCAL.needs_host_gateway is True
    assert _ANTHROPIC.needs_host_gateway is False
    hosted = ModelSpec("openai", "gpt-4o", base_url="https://api.openai.com/v1")
    assert hosted.needs_host_gateway is False


def test_resolve_model_returns_bare_string_without_base_url() -> None:
    # The native path is unchanged: a string for init_chat_model (Anthropic today).
    assert resolve_model(_ANTHROPIC) == "anthropic:claude-sonnet-4-6"


def test_resolve_model_builds_chat_model_for_base_url() -> None:
    seen: dict[str, object] = {}

    def fake_factory(**kwargs: object) -> str:
        seen.update(kwargs)
        return "CHAT"

    out = resolve_model(_LOCAL, chat_openai_factory=fake_factory)
    assert out == "CHAT"
    assert seen["model"] == "qwen2.5-coder"
    assert seen["base_url"] == "http://host.docker.internal:8000/v1"
    assert seen["temperature"] == 0  # extra kwargs forwarded


def test_resolve_model_reads_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_KEY", "sk-local")
    seen: dict[str, object] = {}
    resolve_model(_LOCAL, chat_openai_factory=lambda **kw: seen.update(kw))
    assert seen["api_key"] == "sk-local"


def test_resolve_model_uses_dummy_key_for_keyless_local() -> None:
    keyless = ModelSpec("openai-compatible", "llama", base_url="http://host.docker.internal:8000/v1")
    seen: dict[str, object] = {}
    resolve_model(keyless, chat_openai_factory=lambda **kw: seen.update(kw))
    assert seen["api_key"] == "EMPTY"  # ChatOpenAI requires *some* key; vLLM/Ollama convention


def test_coerce_model_passes_through_non_specs() -> None:
    assert coerce_model("anthropic:claude-sonnet-4-6") == "anthropic:claude-sonnet-4-6"
    sentinel = object()
    assert coerce_model(sentinel) is sentinel  # an already-constructed chat model is untouched


def test_coerce_model_resolves_a_spec() -> None:
    assert coerce_model(_ANTHROPIC) == "anthropic:claude-sonnet-4-6"


def test_resolve_model_real_chat_openai_attrs(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("LOCAL_KEY", "sk-local")
    chat = resolve_model(_LOCAL)
    assert chat.model_name == "qwen2.5-coder"
    assert chat.openai_api_base == "http://host.docker.internal:8000/v1"
