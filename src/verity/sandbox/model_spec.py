"""The provider-agnostic model seam (ROADMAP Phase 6.1/6.2).

A :class:`ModelSpec` names a model target richly enough to span three cases behind one seam:

- **native provider** (today's Anthropic) — a bare ``provider:model`` string handed to Deep Agents /
  langchain ``init_chat_model`` unchanged;
- any **OpenAI-compatible endpoint** with a custom ``base_url`` — a host-local server (vLLM / Ollama
  / llama.cpp, reached via ``host.docker.internal``) **or** a cheaper hosted provider.

The module is **framework-neutral**: it imports no Deep Agents / langchain at load time
(``langchain_openai`` is imported lazily inside :func:`resolve_model`, mirroring how
:func:`verity.sandbox.registration.build_deepagents_sandbox` lazily imports deepagents), so the host
control plane can build a spec without the ``sandbox`` extra. The spec crosses the container border
as env vars (callables/objects can't), reconstructed by :mod:`verity.sandbox.container_entry`.

Backward-compat is a hard design constraint: an Anthropic spec (``base_url=None``) serializes to
**exactly** ``{"VERITY_SANDBOX_MODEL": "anthropic:claude-sonnet-4-6"}`` and resolves to that bare
string — so the existing container command and in-process path are byte-for-byte unchanged.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import urlparse

__all__ = ["ModelSpec", "resolve_model", "coerce_model"]

# Env vars carrying the spec across the host -> container boundary. VERITY_SANDBOX_MODEL is the
# pre-Phase-6 name (a plain provider string); the rest are emitted only when set, so the Anthropic
# command stays identical.
ENV_MODEL = "VERITY_SANDBOX_MODEL"
ENV_BASE_URL = "VERITY_SANDBOX_BASE_URL"
ENV_API_KEY_ENV = "VERITY_SANDBOX_API_KEY_ENV"
ENV_EXTRA = "VERITY_SANDBOX_MODEL_EXTRA"

_DEFAULT_MODEL_STRING = "anthropic:claude-sonnet-4-6"
# A provider's conventional API-key env var, used when a spec doesn't name one explicitly.
_DEFAULT_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
# Keyless OpenAI-compatible servers (vLLM / Ollama) still require *some* api_key argument.
_KEYLESS_PLACEHOLDER = "EMPTY"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A provider-agnostic model target.

    ``base_url=None`` is the native-provider path (a bare ``provider:model`` string); a non-None
    ``base_url`` targets any OpenAI-compatible endpoint. ``api_key_env`` names the env var with the
    key (``None`` -> the provider default, or no key for a keyless local server). ``extra`` are
    JSON-serializable kwargs forwarded to the chat-model constructor (e.g. ``temperature``).
    """

    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_provider_string(cls, s: str) -> ModelSpec:
        """Parse a langchain-style ``"anthropic:claude-sonnet-4-6"`` into a native-provider spec."""
        provider, sep, model = s.partition(":")
        if not sep:  # no provider prefix — treat the whole string as the model, default provider
            return cls(provider="anthropic", model=s)
        return cls(provider=provider, model=model)

    def provider_string(self) -> str:
        """The ``provider:model`` string (what native providers hand to ``init_chat_model``)."""
        return f"{self.provider}:{self.model}"

    def key_env(self) -> str | None:
        """The env var holding this spec's API key, or ``None`` (provider default / keyless)."""
        if self.api_key_env:
            return self.api_key_env
        return _DEFAULT_KEY_ENV.get(self.provider)

    @property
    def gateway_host(self) -> str | None:
        """The ``*.docker.internal`` hostname to map to the host gateway, or ``None``.

        Any ``*.docker.internal`` name (``host.docker.internal``; Docker Model Runner's
        ``model-runner.docker.internal``; …) is a Docker Desktop special host alias for the machine
        the daemon runs on; a container needs ``--add-host=<it>:host-gateway`` to resolve it
        (required on Linux, a no-op on macOS). Public endpoints return ``None``.
        """
        if self.base_url is None:
            return None
        host = urlparse(self.base_url).hostname
        if host is None:  # base_url without a scheme — best-effort token scan
            host = self.base_url.split("/", 1)[0].split(":", 1)[0] or None
        return host if host and host.endswith(".docker.internal") else None

    @property
    def needs_host_gateway(self) -> bool:
        """True iff the endpoint is a Docker-host-internal address (needs ``--add-host``)."""
        return self.gateway_host is not None

    def to_env(self) -> dict[str, str]:
        """Project to env vars for the container boundary (only set fields are emitted)."""
        env = {ENV_MODEL: self.provider_string()}
        if self.base_url is not None:
            env[ENV_BASE_URL] = self.base_url
        if self.api_key_env is not None:
            env[ENV_API_KEY_ENV] = self.api_key_env
        if self.extra:
            env[ENV_EXTRA] = json.dumps(dict(self.extra), sort_keys=True)
        return env

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> ModelSpec:
        """Reconstruct from env (the container side). Empty env -> the default Anthropic spec."""
        spec = cls.from_provider_string(env.get(ENV_MODEL, _DEFAULT_MODEL_STRING))
        extra_raw = env.get(ENV_EXTRA)
        return replace(
            spec,
            base_url=env.get(ENV_BASE_URL),
            api_key_env=env.get(ENV_API_KEY_ENV),
            extra=json.loads(extra_raw) if extra_raw else {},
        )


def resolve_model(spec: ModelSpec, *, chat_openai_factory: Callable[..., Any] | None = None) -> Any:
    """Resolve a spec to whatever ``create_deep_agent(model=...)`` accepts.

    No ``base_url`` -> the bare ``provider:model`` string (langchain's ``init_chat_model`` resolves
    it; Anthropic is unchanged). With a ``base_url`` -> a constructed ``ChatOpenAI`` pointed at the
    endpoint, its key read from :meth:`ModelSpec.key_env` (falling back to ``"EMPTY"`` for keyless
    local servers). ``chat_openai_factory`` is an injection seam for offline tests; in production
    the real ``langchain_openai.ChatOpenAI`` is imported lazily (only in the ``sandbox`` extra).
    """
    if spec.base_url is None:
        return spec.provider_string()
    factory: Callable[..., Any]
    if chat_openai_factory is not None:
        factory = chat_openai_factory
    else:
        from langchain_openai import ChatOpenAI  # only importable with the sandbox extra

        factory = ChatOpenAI  # ChatOpenAI coerces a str api_key to SecretStr at runtime
    key_env = spec.key_env()
    api_key = (os.environ.get(key_env) if key_env else None) or _KEYLESS_PLACEHOLDER
    # Unpack a dict so ChatOpenAI's typed ctor doesn't reject the str api_key (it coerces to
    # SecretStr at runtime); local servers also accept the "EMPTY" placeholder.
    kwargs: dict[str, Any] = {
        "model": spec.model, "base_url": spec.base_url, "api_key": api_key, **dict(spec.extra)
    }
    return factory(**kwargs)


def coerce_model(model: Any) -> Any:
    """The single resolution edge for both drivers: resolve a :class:`ModelSpec`, else passthrough.

    A bare string (``"anthropic:..."``) or an already-constructed chat model (the fake-model tests)
    passes through untouched, so existing call sites and offline tests are unaffected.
    """
    return resolve_model(model) if isinstance(model, ModelSpec) else model
