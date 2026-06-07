"""The extension-point registries (spec §8).

The four per-domain seams the kernel manipulates only through typed contracts:

* **Schema registry** (§8.1) — artifact types + typed operation signatures, versioned. The
  kernel reads only the *meta-schema* (names and signatures); it never interprets payload
  semantics.
* **Tool registry** (§8.2) — typed plugins (a declared input/output signature referencing the
  schema). A tool produces a *proposal*; it carries no acceptance authority. Signatures are
  typed on **both** sides — the [SC] lesson: declared input types make operation composition
  checkable rather than coincidental.
* **Gate registry** (§8.3) — **bindings** (an ordered per-type pipeline) live here; the gate
  **plugins** that actually judge live in the verifier (§3.6). :meth:`GateRegistry.resolve` is
  the :class:`~verity.control_plane.commit.GateBindingResolver` the commit path consumes, and a
  type with no binding resolves to ``None`` — the "no implicit accept" trigger (§5.7).
* **Retrieval / planner policy** (§8.4) — ``(goal, store) → ranked artifacts``. The kernel ships
  a status-aware default (prefers ``accepted`` over ``tentative``); a domain may replace it.

The registries are configuration the control plane reads; they hold no LLM and make no
judgment (Principle 9). Gate *bindings* are reused from :mod:`.commit` so the commit path stays
self-contained and the registry simply stores and resolves them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from verity.control_plane.commit import GateBinding, GateSpec
from verity.control_plane.store import (
    Artifact,
    ArtifactStatus,
    SchemaVersion,
    Store,
)
from verity.logging import get_logger

__all__ = [
    "ArtifactTypeDef",
    "OperationSignature",
    "SchemaRegistry",
    "ToolSpec",
    "ToolRegistry",
    "GateRegistry",
    "RetrievalPolicy",
    "DefaultRetrievalPolicy",
    "RegistryError",
]

log = get_logger("verity.control_plane.registries")


class RegistryError(RuntimeError):
    """Raised on a registry-contract violation (an unknown type, a duplicate name)."""


# --------------------------------------------------------------------- schema registry (§8.1)


@dataclass(frozen=True, slots=True)
class ArtifactTypeDef:
    """A declared artifact type: a name plus whether it is a root (raw-input) type (§8.1).

    The kernel reads only the name (the meta-schema); the payload shape is the domain's to
    interpret. ``is_root`` types are the §5.4 provenance exception.
    """

    name: str
    is_root: bool = False


@dataclass(frozen=True, slots=True)
class OperationSignature:
    """A typed operation signature: name, input types → output type (§8.2).

    Typed on both sides so operation composition is checkable ([SC], §8.2).
    """

    name: str
    inputs: tuple[str, ...]
    output: str


class SchemaRegistry:
    """The declared types and operation signatures for a domain, versioned (§8.1).

    Pure configuration — no store dependency. The control plane persists a snapshot as a
    ``schema_versions`` row when a task is configured (see :meth:`snapshot`).
    """

    def __init__(self) -> None:
        self._types: dict[str, ArtifactTypeDef] = {}
        self._operations: dict[str, OperationSignature] = {}

    def register_type(self, type_def: ArtifactTypeDef) -> None:
        if type_def.name in self._types:
            raise RegistryError(f"artifact type already registered: {type_def.name!r}")
        self._types[type_def.name] = type_def
        log.debug("type_registered", type=type_def.name, is_root=type_def.is_root)

    def register_operation(self, signature: OperationSignature) -> None:
        if signature.name in self._operations:
            raise RegistryError(f"operation already registered: {signature.name!r}")
        for referenced in (*signature.inputs, signature.output):
            if referenced not in self._types:
                raise RegistryError(
                    f"operation {signature.name!r} references unknown type {referenced!r}"
                )
        self._operations[signature.name] = signature
        log.debug("operation_registered", op=signature.name)

    def type(self, name: str) -> ArtifactTypeDef | None:
        return self._types.get(name)

    def is_registered(self, name: str) -> bool:
        return name in self._types

    def types(self) -> tuple[ArtifactTypeDef, ...]:
        return tuple(self._types.values())

    def operation(self, name: str) -> OperationSignature | None:
        return self._operations.get(name)

    def operations(self) -> tuple[OperationSignature, ...]:
        return tuple(self._operations.values())

    def snapshot(self, *, version: int, created_at: str) -> SchemaVersion:
        """Build a stampable :class:`SchemaVersion` of the current schema (§4.1, §8.1)."""
        return SchemaVersion(
            version=version,
            types=tuple(sorted(self._types)),
            op_signatures=tuple(sorted(self._operations)),
            created_at=created_at,
        )


# ----------------------------------------------------------------------- tool registry (§8.2)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A typed tool plugin: a name and an input/output signature over schema types (§8.2).

    The callable executes in the **sandbox** (§3.5), not here — the registry holds the typed
    signature so composition is checkable and the invocation can be recorded as an
    ``operations`` row. A tool carries no acceptance authority; it produces a *proposal*.
    """

    name: str
    inputs: tuple[str, ...]
    output: str


class ToolRegistry:
    """The domain's tools, keyed by name, validated against the schema (§8.2)."""

    def __init__(self, schema: SchemaRegistry) -> None:
        self._schema = schema
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise RegistryError(f"tool already registered: {tool.name!r}")
        for referenced in (*tool.inputs, tool.output):
            if not self._schema.is_registered(referenced):
                raise RegistryError(
                    f"tool {tool.name!r} references unknown type {referenced!r}"
                )
        self._tools[tool.name] = tool
        log.debug("tool_registered", tool=tool.name)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def tools(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tools.values())


# ----------------------------------------------------------------------- gate registry (§8.3)


class GateRegistry:
    """Per-type gate **bindings** (§8.3); the plugins that judge live in the verifier (§3.6).

    A type with no binding resolves to ``None`` — the commit path turns that into a
    ``NoImplicitAccept`` error (§5.7). This registry is exactly where "no implicit accept" is
    satisfied or violated.
    """

    def __init__(self) -> None:
        self._bindings: dict[str, GateBinding] = {}

    def bind(self, artifact_type: str, *gates: GateSpec) -> None:
        """Bind an ordered (cheap-first, then hard) gate pipeline to a type (§8.3)."""
        if artifact_type in self._bindings:
            raise RegistryError(f"gate binding already registered for type: {artifact_type!r}")
        if not gates:
            raise RegistryError(
                f"refusing to bind an empty pipeline to {artifact_type!r}: a binding must "
                f"declare at least one gate, else it is an implicit accept (§5.7)"
            )
        self._bindings[artifact_type] = GateBinding(artifact_type=artifact_type, gates=gates)
        log.debug("gate_bound", type=artifact_type, gates=[g.name for g in gates])

    def register_binding(self, binding: GateBinding) -> None:
        self.bind(binding.artifact_type, *binding.gates)

    def resolve(self, artifact_type: str) -> GateBinding | None:
        """The :class:`GateBindingResolver`: a type's pipeline, or ``None`` (§5.7, §8.3)."""
        return self._bindings.get(artifact_type)


# ------------------------------------------------------------------ retrieval policy (§8.4)


class RetrievalPolicy(Protocol):
    """``(goal, store) → ranked artifacts`` — which artifacts to preload, and how ranked (§8.4).

    Feeds context assembly (§9) but does not change the bounded-context guarantee — callers
    pass a ``limit``.
    """

    def select(self, goal: str, store: Store, *, limit: int) -> list[Artifact]: ...


# Status precedence: trusted artifacts first (§6, §8.4). 'accepted' outranks 'tentative'.
_STATUS_RANK: dict[ArtifactStatus, int] = {
    ArtifactStatus.ACCEPTED: 0,
    ArtifactStatus.TENTATIVE: 1,
    ArtifactStatus.PROPOSED: 2,
    ArtifactStatus.REVISED: 3,
    ArtifactStatus.SUPERSEDED: 4,
    ArtifactStatus.REJECTED: 5,
}


@dataclass(frozen=True, slots=True)
class DefaultRetrievalPolicy:
    """The kernel default: status-aware, recency-tiebroken retrieval (§8.4).

    Prefers ``accepted`` over ``tentative`` over the rest, then more-recent (``created_at``)
    first. Terminal-rejected artifacts rank last but are not excluded — a domain policy may
    refine this. ``goal`` is unused by the default (it carries no domain knowledge).
    """

    include_terminal: bool = False

    def select(self, goal: str, store: Store, *, limit: int) -> list[Artifact]:
        candidates = store.query_artifacts()
        if not self.include_terminal:
            terminal = (
                ArtifactStatus.REJECTED,
                ArtifactStatus.SUPERSEDED,
                ArtifactStatus.REVISED,
            )
            candidates = [a for a in candidates if a.status not in terminal]
        # Two stable passes: recency descending, then status ascending — within each status
        # band the more-recent artifact stays first (Python's sort is stable).
        candidates.sort(key=lambda a: a.created_at, reverse=True)
        candidates.sort(key=lambda a: _STATUS_RANK[a.status])
        return candidates[:limit]
