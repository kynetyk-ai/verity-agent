"""The declarative, JSON-serializable task definition (ROADMAP 8.1, ADR 0004 (b)).

A :class:`TaskRequest` is *what a client sends* to define a task instance: a catalog ``type_name``
(`"fe"`, `"code"`), the sandbox config (a `ModelSpec` + the provisioning knobs), the verifier
approach (a selector + knobs), the orchestration policy, and references to the input data.
It carries **only data** — the callables (gates, ``shape_validator``, ``harvester``) come from the
*named domain* via the catalog builder, never the wire — which is the whole point of the catalog
model (a client can't smuggle code through a request).

It is the durable, rehydratable definition of a task: persisted by the service's task index, it is
re-read after a restart to rebuild the task's control plane (ADR 0004 (b)/(h)). ``to_dict`` /
``from_dict`` round-trip through plain JSON so the index can store it as a file.

Lives in :mod:`verity.composition` (the one layer allowed to know domains), but is itself pure data:
its only imports are the light, framework-neutral :class:`ModelSpec` and the kernel
:class:`OrchestrationPolicy` — no domain, no cycle with the builders that consume it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from verity.control_plane.api import OrchestrationPolicy
from verity.sandbox.model_spec import ModelSpec
from verity.sandbox.providers import LOCAL_PROVIDER

__all__ = [
    "SandboxRequest",
    "VerifierRequest",
    "PolicyRequest",
    "DataRequest",
    "TaskRequest",
]


@dataclass(frozen=True, slots=True)
class SandboxRequest:
    """The sandbox selection: a model target (`ModelSpec` fields) + the provisioning knobs.

    ``model`` is a langchain-style ``"provider:model"`` string (``None`` -> the builder's default);
    ``base_url`` targets an OpenAI-compatible endpoint (local/hosted) per Phase 6. The remaining
    fields are the `ProvisioningConfig` knobs the builder reconstructs (images, memory, timeouts).
    """

    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    sandbox_image: str = "verity-sandbox:latest"
    code_image: str = "verity-code-runner:latest"
    runtime: str | None = None
    sandbox_memory: str = "4g"
    code_memory: str = "2g"
    code_tmpfs_size: str = "1g"
    recursion_limit: int = 200
    sandbox_timeout_s: float = 1500.0
    code_timeout_s: float = 600.0
    # Per-cycle model-step budget (ROADMAP 5.1, #103). ``None`` lets the driver derive a graceful
    # default from ``recursion_limit`` so every task gets the soft "wrap up" step nudge before the
    # framework's hard ``GraphRecursionError``; an explicit value overrides the derivation.
    step_budget: int | None = None

    def to_model_spec(self) -> ModelSpec | None:
        """Reconstruct the `ModelSpec`, or ``None`` to let the builder pick its default model.

        ``None`` only when *nothing* about the model was specified (no ``model``, no ``base_url``);
        any local/hosted endpoint forces a concrete spec so the default provider string is not used.

        When a ``base_url`` is set the target is an **OpenAI-compatible endpoint**, so ``model`` is
        a *literal* server-side name — which may itself contain colons (e.g. Ollama's
        ``qwen3.6:27b-coding-mxfp8``). It must **not** be split on ``:`` into provider/model (that
        mangles the name); we build the spec exactly as `local_spec` does. The ``provider:model``
        split applies only to the native-provider path (no ``base_url``).
        """
        if self.model is None and self.base_url is None:
            return None
        if self.base_url is not None:  # OpenAI-compatible endpoint: literal model name, no split
            base = ModelSpec(provider=LOCAL_PROVIDER, model=self.model or "")
        elif self.model is not None:
            base = ModelSpec.from_provider_string(self.model)
        else:
            base = ModelSpec(provider="anthropic", model="claude-sonnet-4-6")
        return replace(
            base, base_url=self.base_url, api_key_env=self.api_key_env, extra=dict(self.extra)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "extra": dict(self.extra),
            "sandbox_image": self.sandbox_image,
            "code_image": self.code_image,
            "runtime": self.runtime,
            "sandbox_memory": self.sandbox_memory,
            "code_memory": self.code_memory,
            "code_tmpfs_size": self.code_tmpfs_size,
            "recursion_limit": self.recursion_limit,
            "sandbox_timeout_s": self.sandbox_timeout_s,
            "code_timeout_s": self.code_timeout_s,
            "step_budget": self.step_budget,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SandboxRequest:
        return cls(**{k: v for k, v in d.items() if k in cls.__slots__})


@dataclass(frozen=True, slots=True)
class VerifierRequest:
    """The verifier selection: an ``approach`` selector + free-form ``knobs`` the builder reads.

    8.1 has a single FE approach, so ``approach`` is informational (the builder ignores unknown
    selectors and uses its default); it is the seam for offering alternative gate stacks later.
    """

    approach: str = "default"
    knobs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"approach": self.approach, "knobs": dict(self.knobs)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VerifierRequest:
        return cls(approach=d.get("approach", "default"), knobs=dict(d.get("knobs", {})))


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    """The orchestration policy, declaratively (mirrors `OrchestrationPolicy`)."""

    max_cycles: int = 4
    refine_cap: int = 3
    stop_on_accept: bool = False
    max_consecutive_sandbox_failures: int = 3
    # Object-provisioning spec — a preset name (e.g. "best_revised_or_accepted") or an explicit
    # ``{"statuses": [...], "select": "all|last|best"}``. Lives here because provisioning is
    # orchestration (what the agent sees next cycle), not gating; ``None`` = the task's default.
    provisioning: str | dict[str, Any] | None = None

    def to_orchestration_policy(self) -> OrchestrationPolicy:
        return OrchestrationPolicy(
            max_cycles=self.max_cycles,
            refine_cap=self.refine_cap,
            stop_on_accept=self.stop_on_accept,
            max_consecutive_sandbox_failures=self.max_consecutive_sandbox_failures,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_cycles": self.max_cycles,
            "refine_cap": self.refine_cap,
            "stop_on_accept": self.stop_on_accept,
            "max_consecutive_sandbox_failures": self.max_consecutive_sandbox_failures,
            "provisioning": self.provisioning,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PolicyRequest:
        return cls(**{k: v for k, v in d.items() if k in cls.__slots__})


@dataclass(frozen=True, slots=True)
class DataRequest:
    """References to the task's **pre-prepared, role-keyed** input files (ADR 0005 / #74).

    ``roles`` maps a worker role (``"agent"`` / ``"verifier"``) to ``{filename: content_hash}`` —
    each value the content hash of an ingested object in the task's store (immutable, set by
    ``ControlService.create_task``). The control plane routes a role's file set, by name, into that
    role's worker and **interprets none of it**: data preparation (splitting, the answer key) is the
    user's responsibility, performed outside Verity, so the CP holds no dataset semantics (no
    ``target``/``id_column``, no split). An empty map means the task needs no client data.
    """

    roles: dict[str, dict[str, str]] = field(default_factory=dict)

    def files_for(self, role: str) -> dict[str, str]:
        """The ``{filename: content_hash}`` map for ``role`` (empty if the role has no files)."""
        return dict(self.roles.get(role, {}))

    def with_role_file(self, role: str, name: str, ref: str) -> DataRequest:
        """Return a copy with ``ref`` stamped at ``roles[role][name]`` (used at ingest time)."""
        roles = {r: dict(files) for r, files in self.roles.items()}
        roles.setdefault(role, {})[name] = ref
        return replace(self, roles=roles)

    def to_dict(self) -> dict[str, Any]:
        return {"roles": {r: dict(files) for r, files in self.roles.items()}}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataRequest:
        roles = d.get("roles", {})
        return cls(roles={r: dict(files) for r, files in roles.items()})


@dataclass(frozen=True, slots=True)
class TaskRequest:
    """A durable, JSON-serializable task definition: the catalog selection + its declarative knobs.

    ``type_name`` selects the catalog builder; ``goal`` is the run goal; the four sub-requests carry
    the sandbox / verifier / policy / data selections. ``tenant_id`` is the single default tenant in
    v1 (cross-tenant isolation is the deferred engine, #58).
    """

    type_name: str
    goal: str = ""
    sandbox: SandboxRequest = field(default_factory=SandboxRequest)
    verifier: VerifierRequest = field(default_factory=VerifierRequest)
    policy: PolicyRequest = field(default_factory=PolicyRequest)
    data: DataRequest = field(default_factory=DataRequest)
    tenant_id: str = "default"

    def with_role_file(self, role: str, name: str, ref: str) -> TaskRequest:
        """Return a copy with ``ref`` stamped at ``data.roles[role][name]`` (at ingest time)."""
        return replace(self, data=self.data.with_role_file(role, name, ref))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type_name": self.type_name,
            "goal": self.goal,
            "sandbox": self.sandbox.to_dict(),
            "verifier": self.verifier.to_dict(),
            "policy": self.policy.to_dict(),
            "data": self.data.to_dict(),
            "tenant_id": self.tenant_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskRequest:
        return cls(
            type_name=d["type_name"],
            goal=d.get("goal", ""),
            sandbox=SandboxRequest.from_dict(d.get("sandbox", {})),
            verifier=VerifierRequest.from_dict(d.get("verifier", {})),
            policy=PolicyRequest.from_dict(d.get("policy", {})),
            data=DataRequest.from_dict(d.get("data", {})),
            tenant_id=d.get("tenant_id", "default"),
        )
