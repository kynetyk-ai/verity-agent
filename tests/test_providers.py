"""Ready-made ModelSpec constructors (ROADMAP Phase 6). Pure-function; no network, no Docker."""

from __future__ import annotations

from verity.sandbox.providers import DEFAULT_LOCAL_BASE_URL, local_spec, openai_spec


def test_local_spec_targets_the_host_gateway_keyless() -> None:
    spec = local_spec("qwen2.5-coder")
    assert spec.provider == "openai-compatible" and spec.model == "qwen2.5-coder"
    assert spec.base_url == DEFAULT_LOCAL_BASE_URL
    assert spec.needs_host_gateway is True  # default base_url is on the Docker host
    assert spec.key_env() is None  # keyless local server -> resolver sends EMPTY


def test_local_spec_accepts_a_custom_base_url_and_key() -> None:
    spec = local_spec(
        "qwen2.5-coder", base_url="http://host.docker.internal:11434/v1", api_key_env="OLLAMA_KEY"
    )
    assert spec.base_url == "http://host.docker.internal:11434/v1"  # e.g. Ollama's port
    assert spec.key_env() == "OLLAMA_KEY"
    assert spec.needs_host_gateway is True


def test_local_spec_forwards_extra_kwargs() -> None:
    spec = local_spec("qwen2.5-coder", temperature=0)
    assert spec.extra == {"temperature": 0}


def test_local_spec_roundtrips_through_env() -> None:
    from verity.sandbox.model_spec import ModelSpec

    spec = local_spec("qwen2.5-coder", api_key_env="LOCAL_KEY", temperature=0)
    assert ModelSpec.from_env(spec.to_env()) == spec


def test_openai_spec_is_native_with_default_key_no_gateway() -> None:
    spec = openai_spec("gpt-4o-mini")
    assert spec.provider == "openai" and spec.model == "gpt-4o-mini"
    assert spec.base_url is None  # hosted -> the native openai:<model> path, no base_url
    assert spec.needs_host_gateway is False  # public endpoint, no --add-host
    assert spec.key_env() == "OPENAI_API_KEY"  # forwarded automatically
    assert spec.to_env() == {"VERITY_SANDBOX_MODEL": "openai:gpt-4o-mini"}
