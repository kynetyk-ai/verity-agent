"""The extension-point registries (spec §8).

The per-domain seams the kernel manipulates only through typed contracts:

* **Schema registry** (§8.1) — artifact types + typed operation signatures, versioned. The
  kernel reads only the *meta-schema* (names and signatures); it never interprets payload
  semantics. Operation signatures are typed on **both** sides — the [SC] lesson: declared input
  types make operation composition checkable rather than coincidental.
* **Gate registry** (§8.3) — **bindings** (an ordered per-type pipeline) live here; the gate
  **plugins** that actually judge live in the verifier (§3.6). :meth:`GateRegistry.resolve` is
  the :class:`~verity.control_plane.commit.GateBindingResolver` the commit path consumes, and a
  type with no binding resolves to ``None`` — the "no implicit accept" trigger (§5.7).
* **Retrieval / planner policy** (§8.4) — ``(goal, store) → ranked artifacts``. The kernel ships
  a status-aware default (prefers ``accepted`` over ``tentative``); a domain may replace it.

**On the §8.2 tool registry — intentionally collapsed.** A tool is the *executable* realization
of an operation, and the executable form is harness-specific (a pi-mono tool, a Claude Agent SDK
tool def, a LangChain tool, an MCP server — different schemas and calling conventions). Strip the
harness binding out and a tool's only harness-agnostic content is its typed signature — which is
*already* the :class:`OperationSignature` above (a tool invocation is recorded as an ``operations``
row, §8.2). So a separate control-plane tool registry would merely duplicate op signatures while
implying the control plane owns tool execution (it does not). The typed/provenance half lives here
in the schema registry; the harness-bound half belongs to the **sandbox adapter** (Phase 3),
parallel to ``WorkspaceLayout``. This is a deliberate divergence from §8.2's four-extension-point
shape, to be synced back to the spec.

The registries are configuration the control plane reads; they hold no LLM and make no judgment
(Principle 9). Gate *bindings* are reused from :mod:`.commit` so the commit path stays
self-contained and the registry simply stores and resolves them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from verity.control_plane.independence import StoreInput
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
    "GatedTypeRegistry",
    "RetrievalPolicy",
    "DefaultRetrievalPolicy",
    "ObjectProvisionMode",
    "ProvisionStatusSet",
    "ProvisionSelect",
    "ObjectProvisioningPolicy",
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
    ``required_payload_keys`` is an *optional* per-operation payload schema (presence-only): the
    keys the proposal payload must carry, checked at intake (ROADMAP 5.1 #13). Structural, not
    semantic — the control plane never interprets payload values, only that declared keys exist.
    """

    name: str
    inputs: tuple[str, ...]
    output: str
    required_payload_keys: tuple[str, ...] = ()


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


# Note: there is no tool registry (§8.2) here — see the module docstring. A tool's typed
# signature is an :class:`OperationSignature` (above); its harness-bound, executable form is a
# Phase-3 sandbox-adapter concern, not a control-plane construct.


# ----------------------------------------------------------------------- gate registry (§8.3)


class GatedTypeRegistry:
    """The control plane's coverage + independence record per gated type (§8.3, ADR 0001).

    Under ADR 0001 the gate **pipeline** (which checks, in what order, cheap/hard) lives in the
    verifier package; the control plane keeps only this: *which types are gated*, and *the exact
    store-slice the verifier may see* for each (the §10 independence allowlist). A type with **no**
    entry resolves to ``None`` — committing it is a ``NoImplicitAccept`` error (§5.7). A gated
    type's declared inputs may be empty (it sees only the artifact under test); that is still gated.
    """

    def __init__(self) -> None:
        self._gated: dict[str, frozenset[StoreInput]] = {}

    def gate(
        self, artifact_type: str, *, declared_inputs: frozenset[StoreInput] = frozenset()
    ) -> None:
        """Register ``artifact_type`` as gated, with the store-inputs its verifier sees (§8.3)."""
        if artifact_type in self._gated:
            raise RegistryError(f"type already registered as gated: {artifact_type!r}")
        self._gated[artifact_type] = declared_inputs
        log.debug("type_gated", type=artifact_type, declared_inputs=sorted(declared_inputs))

    def resolve(self, artifact_type: str) -> frozenset[StoreInput] | None:
        """The type's declared store-inputs, or ``None`` if it is not gated (§5.7, §10)."""
        return self._gated.get(artifact_type)

    def is_gated(self, artifact_type: str) -> bool:
        return artifact_type in self._gated


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


# ------------------------------------------------------ object-provisioning policy (durable refs)


class ProvisionStatusSet(StrEnum):
    """Which artifact statuses qualify for provisioning — one axis of :class:`ObjectProvisionMode`.

    Control-plane-native (status only); never verifier semantics. ``ANY`` is **literally every
    status** (incl. ``rejected``/``superseded``/in-flight ``proposed``/``tentative``) — the agent's
    full history. ``ACCEPTED_OR_REVISED`` keeps ``revised`` "live" for provisioning (the #51
    refine-edit target) even though :class:`DefaultRetrievalPolicy` treats ``revised`` as terminal —
    a deliberate asymmetry: an in-flight refine is worth iterating on, a terminal one is not.
    """

    ACCEPTED = "accepted"
    ACCEPTED_OR_REVISED = "accepted_or_revised"
    ANY = "any"


class ProvisionSelect(StrEnum):
    """How to pick among the qualifying artifacts — the other axis of :class:`ObjectProvisionMode`.

    ``ALL`` provisions every qualifier (recency-ranked); ``LAST`` the single most-recent; ``BEST``
    the single highest **recorded score** (``_artifact_score`` — a recorded measurement, not
    verifier state, so still control-plane-native). ``BEST`` breaks ties by recency and, when *no*
    score is recorded for any candidate, degrades to ``LAST`` (measured beats unmeasured; newest
    wins absent any measurement).
    """

    ALL = "all"
    LAST = "last"
    BEST = "best"


#: Statuses each set admits. ``ANY`` carries ``None`` — "no status filter, every status qualifies".
_STATUS_SET_MEMBERS: dict[ProvisionStatusSet, frozenset[ArtifactStatus] | None] = {
    ProvisionStatusSet.ACCEPTED: frozenset({ArtifactStatus.ACCEPTED}),
    ProvisionStatusSet.ACCEPTED_OR_REVISED: frozenset(
        {ArtifactStatus.ACCEPTED, ArtifactStatus.REVISED}
    ),
    ProvisionStatusSet.ANY: None,
}


class ObjectProvisionMode(StrEnum):
    """How many durable objects to materialize into the workspace as references each cycle.

    Stated in **control-plane-native terms** (status + recency + recorded score) — never verifier
    semantics. Each named mode is one cell of a two-axis grid, *which statuses qualify*
    (:class:`ProvisionStatusSet`) × *how to select among them* (:class:`ProvisionSelect`), resolved
    via :data:`_MODE_AXES`. Only the sensible cells are named (the validity guard) — e.g. there is
    no ``ANY × LAST``/``ANY × BEST`` mode (building on a possibly-rejected artifact is rarely apt).

    ``LAST_REVISED_OR_ACCEPTED`` (``ACCEPTED_OR_REVISED × LAST``) provisions the single most-recent
    ``accepted``/``revised`` artifact, so a refine cycle **edits its prior script** instead of
    reconstructing it from the defect text (issue #51). ``BEST_REVISED_OR_ACCEPTED``
    (``ACCEPTED_OR_REVISED × BEST``) instead hands the agent its highest-*scoring* prior so it never
    iterates away from its peak in an all-``revised`` run (issue #95); when nothing is scored yet it
    behaves like ``LAST_REVISED_OR_ACCEPTED``.
    """

    NONE = "none"
    ALL = "all"
    ALL_ACCEPTED = "all_accepted"
    ALL_REVISED_OR_ACCEPTED = "all_revised_or_accepted"
    LAST_ACCEPTED = "last_accepted"
    LAST_REVISED_OR_ACCEPTED = "last_revised_or_accepted"
    BEST_ACCEPTED = "best_accepted"
    BEST_REVISED_OR_ACCEPTED = "best_revised_or_accepted"


#: Each mode's ``(status set, selection)`` axes; ``NONE`` provisions nothing. Total over the enum
#: (asserted by ``test_mode_axes_total``) so a new mode cannot silently resolve to "nothing".
_MODE_AXES: dict[ObjectProvisionMode, tuple[ProvisionStatusSet | None, ProvisionSelect | None]] = {
    ObjectProvisionMode.NONE: (None, None),
    ObjectProvisionMode.ALL: (ProvisionStatusSet.ANY, ProvisionSelect.ALL),
    ObjectProvisionMode.ALL_ACCEPTED: (ProvisionStatusSet.ACCEPTED, ProvisionSelect.ALL),
    ObjectProvisionMode.ALL_REVISED_OR_ACCEPTED: (
        ProvisionStatusSet.ACCEPTED_OR_REVISED, ProvisionSelect.ALL
    ),
    ObjectProvisionMode.LAST_ACCEPTED: (ProvisionStatusSet.ACCEPTED, ProvisionSelect.LAST),
    ObjectProvisionMode.LAST_REVISED_OR_ACCEPTED: (
        ProvisionStatusSet.ACCEPTED_OR_REVISED, ProvisionSelect.LAST
    ),
    ObjectProvisionMode.BEST_ACCEPTED: (ProvisionStatusSet.ACCEPTED, ProvisionSelect.BEST),
    ObjectProvisionMode.BEST_REVISED_OR_ACCEPTED: (
        ProvisionStatusSet.ACCEPTED_OR_REVISED, ProvisionSelect.BEST
    ),
}


@dataclass(frozen=True, slots=True)
class ObjectProvisioningPolicy:
    """Selects durable objects to drop into a writable workspace role each cycle (§3.4, §9).

    The control plane resolves this against the store by **status / recency / recorded score / type
    only** (never verifier semantics) and reads the selected artifacts' object sidecars via
    ``Store.get_object``. The result is materialized into
    ``target_role`` (a writable role) under ``name_prefix`` and **re-materialized every cycle**, so
    agent edits never persist. ``type_filter`` restricts selection to one artifact type;
    ``max_objects`` bounds total objects so a large store cannot blow up the workspace.
    """

    mode: ObjectProvisionMode = ObjectProvisionMode.NONE
    type_filter: str | None = None
    target_role: str = "scratch"
    name_prefix: str = "provided/"
    max_objects: int = 16

    def _select(self, store: Store) -> list[Artifact]:
        status_set, select = _MODE_AXES[self.mode]
        if status_set is None or select is None:  # NONE — provision nothing
            return []
        # ``query_artifacts`` filters on a single status, so the set is filtered in Python; ``ANY``
        # (members is None) keeps every status, including terminal/in-flight.
        members = _STATUS_SET_MEMBERS[status_set]
        candidates = [
            a
            for a in store.query_artifacts(type=self.type_filter)
            if members is None or a.status in members
        ]
        if select is ProvisionSelect.BEST:
            # Highest recorded score wins, ties (and the no-score-anywhere case) broken by recency —
            # a recorded measurement, not verifier state, so still control-plane-native (#95).
            scores = {a.id: _artifact_score(store, a) for a in candidates}
            ranked = {aid: (s if s is not None else float("-inf")) for aid, s in scores.items()}
            candidates.sort(key=lambda a: a.created_at, reverse=True)
            candidates.sort(key=lambda a: ranked[a.id], reverse=True)
            return candidates[:1]
        candidates.sort(key=lambda a: a.created_at, reverse=True)  # most-recent first
        return candidates[:1] if select is ProvisionSelect.LAST else candidates

    def materialize(self, store: Store) -> dict[str, dict[str, bytes]]:
        """Resolve the policy to ``{role: {name: bytes}}`` ready for ``ServedContext`` (§9).

        Names are **ordered + meaningful** (issue #20). Naming is **count-based**: when ≤1 artifact
        is selected there can be no clash (a single artifact's object names are unique), so names
        stay flat (``<prefix><name>``); when ≥2 are selected each artifact is namespaced under a
        recency-ranked subdir ``<prefix><NN>-<id>/<name>`` (``01`` = first by the mode's order) so
        two artifacts that declare the *same* object name cannot clobber, and the ``<NN>`` rank ties
        each file to its ``INDEX.md`` row. A ``<prefix>INDEX.md`` manifest accompanies the bytes,
        listing each provisioned artifact (rank, id, status, recency, type, score). Harvested object
        *names* stay domain-fixed; disambiguation is the subdir scheme + manifest.
        """
        selected = self._select(store)
        single = len(selected) <= 1
        role: dict[str, bytes] = {}
        entries: list[_ManifestEntry] = []
        for rank, artifact in enumerate(selected, start=1):
            subdir = "" if single else f"{rank:02d}-{artifact.id}/"
            files: list[str] = []
            for name, ref in artifact.objects:
                if len(role) >= self.max_objects:
                    break
                key = f"{self.name_prefix}{subdir}{name}"
                role[key] = store.get_object(ref.content_hash)
                files.append(key)
            if files:
                score = _artifact_score(store, artifact)
                entries.append(_ManifestEntry(rank, artifact, score, files))
        if not role:
            return {}
        role[f"{self.name_prefix}INDEX.md"] = _render_manifest(entries)
        return {self.target_role: role}


@dataclass(frozen=True, slots=True)
class _ManifestEntry:
    rank: int
    artifact: Artifact
    score: float | None
    files: list[str]


def _artifact_score(store: Store, artifact: Artifact) -> float | None:
    """The artifact's best recorded gate score (e.g. the accepting selection score), or ``None``.

    A recorded measurement — not verifier-internal state — so surfacing it keeps the control plane
    generic (status/recency/recorded score), consistent with the no-verifier-semantics rule.
    """
    scores = [d.score for d in store.decisions_for(artifact.id) if d.score is not None]
    return max(scores) if scores else None


def _render_manifest(entries: list[_ManifestEntry]) -> bytes:
    """A dumb Markdown index of the provisioned durable objects: a title + a metadata table only.

    Carries **no instructions** — what to do with these objects is the task's configurable concern
    (its composed instructions), never hardwired into this generic control-plane renderer.
    """
    lines = [
        "# Provisioned durable objects",
        "",
        "| rank | artifact | type | status | created_at | score | files |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        art = entry.artifact
        score = f"{entry.score:.4f}" if entry.score is not None else "-"
        lines.append(
            f"| {entry.rank:02d} | {art.id} | {art.type} | {art.status.value} | "
            f"{art.created_at} | {score} | {', '.join(entry.files)} |"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")
