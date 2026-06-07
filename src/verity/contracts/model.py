"""The cross-service value model — the nouns that travel between services (spec §4.1).

These are the immutable value types that cross a service boundary: the control plane hands
:class:`Artifact` slices and :class:`Operation` edges to the sandbox and verifier, and the verifier
hands back a :class:`GateVerdict`. They are the *published vocabulary* of the system, so they live
here in :mod:`verity.contracts` — depended on by every service, owned by none.

This is deliberately **only** the boundary-crossing closure. Control-plane-internal value types
that never reach the verifier or sandbox — :class:`~verity.control_plane.store.Decision`,
``SchemaVersion``, ``Provenance`` — stay with the store, alongside the storage mechanism. The cut
is "what crosses a wire" vs "what the control plane persists and extracts".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "ArtifactStatus",
    "OperationStatus",
    "VerdictKind",
    "JSONValue",
    "ObjectRef",
    "Payload",
    "Artifact",
    "Operation",
    "GateVerdict",
]


# --------------------------------------------------------------------------- enums


class ArtifactStatus(StrEnum):
    """The artifact lifecycle states (spec §4.1, §6)."""

    PROPOSED = "proposed"
    TENTATIVE = "tentative"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    REVISED = "revised"


class OperationStatus(StrEnum):
    """The status of a provenance edge (spec §4.1)."""

    SUCCESS = "success"
    FAILED = "failed"
    RETRIED = "retried"


class VerdictKind(StrEnum):
    """The three answers a gate can give (spec §4.1, §7, §11)."""

    ACCEPT = "accept"
    REJECT = "reject"
    REFINE = "refine"


# --------------------------------------------------------------------- value model

JSONValue = bool | int | float | str | None | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True, slots=True)
class ObjectRef:
    """A reference to a content-addressed object in the object store (spec §4.1).

    Artifact payloads hold this instead of inlining bytes, so code, a data file, or a
    serialized model fits the relational/JSON backends: the row carries the hash and the
    pointer, the bytes live in the object store (§4.2).
    """

    blob_ref: str
    content_hash: str


# A payload is either an inline JSON value or a reference to a stored object (§4.1).
Payload = JSONValue | ObjectRef


@dataclass(frozen=True, slots=True)
class Artifact:
    """A typed payload with a lifecycle status — the durable noun (spec §4.1, §6).

    Frozen: a status transition produces a *new* snapshot via the commit path, never an
    in-place payload edit (§5.2). ``superseded_by`` / ``revised_by`` point to the artifact
    that replaced or revised this one, once that happens (§6, §7).
    """

    id: str
    type: str
    payload: Payload
    status: ArtifactStatus
    created_by: str
    created_at: str
    superseded_by: str | None = None
    revised_by: str | None = None
    is_root: bool = False


@dataclass(frozen=True, slots=True)
class Operation:
    """A typed provenance edge: parents → output, with a success/failed/retried status.

    A ``revises`` operation (parent = the flagged artifact, output = its revision) records a
    refine as provenance-bearing work rather than an in-place edit (spec §4.1, §6).
    """

    op_id: str
    op_name: str
    parents: tuple[str, ...]
    output_id: str
    status: OperationStatus
    created_at: str


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """One gate's advisory verdict (spec §3.6, §7.3). ``defects`` localizes a ``refine``.

    The verifier's reply across the service boundary: the control plane, not the verifier, acts
    on it (§3.6). A ``score`` is present where the gate produced one (the numeric rung, §11).
    """

    kind: VerdictKind
    rationale: str
    defects: tuple[str, ...] | None = None
    score: float | None = None
