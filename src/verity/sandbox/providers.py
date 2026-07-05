"""Ready-made :class:`~verity.sandbox.model_spec.ModelSpec` constructors (ROADMAP Phase 6).

Thin, framework-neutral convenience builders for the model targets Phase 6 cares about. They hold no
logic the seam doesn't already have — they just name the conventions (the default local ``base_url``
and the ``openai-compatible`` provider tag) so a task registration reads as one line. A sandbox is
registered by handing one of these to the production driver via ``build_sandbox``::

    from verity.provisioning import DockerBackend
    from verity.sandbox.backend_driver import BackendSandboxDriver
    from verity.sandbox.providers import local_spec
    from verity.sandbox.registration import build_sandbox, register_sandbox

    register_sandbox(sandbox_providers, "deepagents-local", lambda: build_sandbox(
        schema=schema, root=root, proposer_identity="deepagents-worker:local",
        driver=BackendSandboxDriver(backend=DockerBackend(), spec=local_spec("qwen2.5-coder")),
    ))

The system depends on the **OpenAI-compatible contract**, not on any one server — vLLM, Ollama, and
llama.cpp are interchangeable behind the same ``base_url`` (see ``docs/local-models.md``).
"""

from __future__ import annotations

from typing import Any

from verity.sandbox.model_spec import ModelSpec

__all__ = ["DEFAULT_LOCAL_BASE_URL", "LOCAL_PROVIDER", "local_spec", "openai_spec"]

# A host-local OpenAI-compatible server, reached from inside the sandbox container via the Docker
# host gateway. The port is the common default; override per server (Ollama serves on 11434).
DEFAULT_LOCAL_BASE_URL = "http://host.docker.internal:8000/v1"
LOCAL_PROVIDER = "openai-compatible"


def local_spec(
    model: str,
    *,
    base_url: str = DEFAULT_LOCAL_BASE_URL,
    api_key_env: str | None = None,
    **extra: Any,
) -> ModelSpec:
    """A :class:`ModelSpec` for a host-local OpenAI-compatible server (vLLM / Ollama / llama.cpp).

    ``model`` is the name the server serves (e.g. ``"qwen2.5-coder"``). The default ``base_url``
    points at the Docker host; the driver adds ``--add-host`` automatically (host-gateway). Most
    local servers are keyless — leave ``api_key_env`` unset and the resolver sends ``"EMPTY"``.
    """
    return ModelSpec(
        provider=LOCAL_PROVIDER,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        extra=extra,
    )


def openai_spec(model: str, **extra: Any) -> ModelSpec:
    """A :class:`ModelSpec` for a cheaper **hosted OpenAI** model (Phase 6.2).

    No ``base_url`` -> the native ``openai:<model>`` path (langchain ``init_chat_model``), so
    ``OPENAI_API_KEY`` is forwarded automatically (the provider's default key env) and no host
    gateway is added. ``model`` is the OpenAI model id, e.g. ``"gpt-4o-mini"``.
    """
    return ModelSpec(provider="openai", model=model, extra=extra)
