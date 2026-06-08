"""The store — durable source of truth, behind a backend-agnostic interface (spec §4).

artifacts · operations · decisions · schema_versions, plus the object store
(``put_object`` / ``get_object`` by content hash). Default reference backend: SQLite;
git-on-disk JSON is a valid v0 (deferred).

The data model (§4.1) is a set of **frozen** value objects: a read returns an immutable
snapshot, and there is no method on the public :class:`Store` surface that overwrites an
accepted artifact's payload or hard-deletes it (§5.2, §5.3). Status transitions and
decision/lineage writes are **privileged** — they live on the separate :class:`CommitSink`
surface, which only the commit path (§7) holds. Domain and agent code see :class:`Store`
(reads + ``propose`` + objects + provenance) and nothing that can mutate durable state —
the privileged-mutator boundary of §3.3 made structural rather than conventional.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

# The boundary-crossing value model lives in verity.contracts and is re-exported here, so the store
# still surfaces the nouns it persists (§4.1). Control-plane-internal value types — Decision,
# SchemaVersion, Provenance — are defined below, alongside the storage mechanism.
from verity.contracts.model import (
    Artifact,
    ArtifactStatus,
    JSONValue,
    ObjectRef,
    Operation,
    OperationStatus,
    Payload,
    VerdictKind,
)
from verity.logging import get_logger

__all__ = [
    "ArtifactStatus",
    "OperationStatus",
    "VerdictKind",
    "ObjectRef",
    "Artifact",
    "Operation",
    "Decision",
    "SchemaVersion",
    "Provenance",
    "JSONValue",
    "Payload",
    "Clock",
    "IdFactory",
    "Store",
    "CommitSink",
    "SqliteStore",
    "StoreError",
    "default_clock",
    "default_id_factory",
]

log = get_logger("verity.control_plane.store")


# The cross-service value model (ArtifactStatus / OperationStatus / VerdictKind / ObjectRef /
# Payload / Artifact / Operation) is imported from verity.contracts.model above and re-exported via
# __all__. What follows are the control-plane-internal value types — never sent to a gate or the
# sandbox — defined here alongside the storage mechanism.


@dataclass(frozen=True, slots=True)
class Decision:
    """A recorded gate verdict on an artifact (spec §4.1, §7).

    One row per gate that ruled. ``defects`` names the localized failures a ``refine``
    revision must address; ``score`` is present where the gate produced one.
    """

    artifact_id: str
    gate: str
    verdict: VerdictKind
    rationale: str
    defects: tuple[str, ...] | None = None
    score: float | None = None
    created_at: str = ""


@dataclass(frozen=True, slots=True)
class SchemaVersion:
    """A stamped snapshot of the registered schema in force (spec §4.1, §8.1)."""

    version: int
    types: tuple[str, ...]
    op_signatures: tuple[str, ...]
    created_at: str


@dataclass(frozen=True, slots=True)
class Provenance:
    """The transitive lineage behind an artifact — the basis for "why do we believe X".

    ``operations`` and ``ancestors`` are the transitive provenance DAG (§4.2); ``decisions``
    are the gate verdicts gathered across that lineage, so the answer needs no transcript.
    """

    artifact: Artifact
    operations: tuple[Operation, ...]
    ancestors: tuple[Artifact, ...]
    decisions: tuple[Decision, ...]


# ------------------------------------------------------------------- injected clock/id


class Clock(Protocol):
    """Returns an ISO-8601 timestamp. Injected so verdicts/tests are reproducible (§5.8)."""

    def __call__(self) -> str: ...


class IdFactory(Protocol):
    """Returns a fresh, never-reused identifier (spec §5.1). Injected for determinism."""

    def __call__(self) -> str: ...


def default_clock() -> str:
    return datetime.now(UTC).isoformat()


def default_id_factory() -> str:
    return uuid.uuid4().hex


class StoreError(RuntimeError):
    """Raised on a store-contract violation (a reused id, an unknown object/artifact)."""


# ------------------------------------------------------------------------- interfaces


@runtime_checkable
class Store(Protocol):
    """The world-facing store surface: reads, ``propose``, objects, provenance (spec §4.2).

    This is what domain and agent code see. It carries **no** method that overwrites a
    payload, hard-deletes an accepted artifact, or sets status/decisions — those are
    privileged and live on :class:`CommitSink` (§3.3, §4.2).
    """

    # reads
    def get_artifact(self, artifact_id: str) -> Artifact | None: ...
    def query_artifacts(
        self, *, type: str | None = None, status: ArtifactStatus | None = None
    ) -> list[Artifact]: ...
    def operations_into(self, artifact_id: str) -> list[Operation]: ...
    def operations_out_of(self, artifact_id: str) -> list[Operation]: ...
    def decisions_for(self, artifact_id: str) -> list[Decision]: ...
    def current_schema_version(self) -> SchemaVersion | None: ...
    def schema_versions(self) -> list[SchemaVersion]: ...

    # proposing records an intention; it accepts nothing (§4.2)
    def propose(self, artifact: Artifact, operation: Operation) -> Artifact: ...

    # objects (§4.2)
    def put_object(self, data: bytes) -> ObjectRef: ...
    def get_object(self, content_hash: str) -> bytes: ...

    # provenance / audit (§4.2)
    def get_provenance(self, artifact_id: str) -> Provenance: ...
    def rejected_log(self, *, type: str | None = None) -> list[Artifact]: ...
    def superseded_log(self, *, type: str | None = None) -> list[Artifact]: ...


@runtime_checkable
class CommitSink(Protocol):
    """The privileged write surface — held **only** by the commit path (spec §4.2, §7).

    Every method here mutates durable status or lineage. Nothing on :class:`Store` reaches
    these, so neither domain code nor the agent can set status or write decisions; the
    commit path is the single privileged route (Principle 3/9).
    """

    def record_decision(self, decision: Decision) -> None: ...
    def set_status(self, artifact_id: str, status: ArtifactStatus) -> None: ...
    def accept_superseding(self, new_id: str, old_id: str) -> None: ...
    def link_revision(self, old_id: str, new_id: str) -> None: ...
    def register_schema_version(self, schema: SchemaVersion) -> None: ...

    # one atomic unit spanning several of the writes above (ROADMAP 5.1): a commit's decision rows
    # and its terminal status land together, or not at all — no partial commit on a mid-path fail.
    def transaction(self) -> AbstractContextManager[None]: ...


# ---------------------------------------------------------------------- sqlite backend

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    id            TEXT PRIMARY KEY,
    type          TEXT NOT NULL,
    payload       TEXT NOT NULL,
    status        TEXT NOT NULL,
    created_by    TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    superseded_by TEXT,
    revised_by    TEXT,
    is_root       INTEGER NOT NULL DEFAULT 0,
    objects       TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS operations (
    op_id      TEXT PRIMARY KEY,
    op_name    TEXT NOT NULL,
    parents    TEXT NOT NULL,
    output_id  TEXT NOT NULL,
    status     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    artifact_id TEXT NOT NULL,
    gate        TEXT NOT NULL,
    verdict     TEXT NOT NULL,
    rationale   TEXT NOT NULL,
    defects     TEXT,
    score       REAL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS schema_versions (
    version       INTEGER PRIMARY KEY,
    types         TEXT NOT NULL,
    op_signatures TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_type_status ON artifacts(type, status);
CREATE INDEX IF NOT EXISTS idx_operations_output ON operations(output_id);
CREATE INDEX IF NOT EXISTS idx_decisions_artifact ON decisions(artifact_id);
"""

_OBJECT_MARKER = "__object_ref__"


def _dump_objects(objects: tuple[tuple[str, ObjectRef], ...]) -> str:
    """Serialize an artifact's object sidecar (name → ref) for the ``objects`` column (§4.1)."""
    return json.dumps([[name, ref.blob_ref, ref.content_hash] for name, ref in objects])


def _load_objects(raw: str) -> tuple[tuple[str, ObjectRef], ...]:
    return tuple(
        (name, ObjectRef(blob_ref=blob_ref, content_hash=content_hash))
        for name, blob_ref, content_hash in json.loads(raw)
    )


@dataclass
class SqliteStore:
    """The reference store backend: a single SQLite file plus a dir of hash-named objects.

    Implements both :class:`Store` (world-facing) and :class:`CommitSink` (privileged). A
    factory hands the privileged surface only to the commit path; everyone else is typed to
    :class:`Store`, so the boundary holds without trusting callers (§4.2).

    ``path=":memory:"`` (the default) keeps the whole store in-process — used by tests; a
    file path persists behind a mounted volume for container portability (§3.9). When
    ``object_dir`` is set, objects are hash-named files there; otherwise they live in memory.
    """

    path: str = ":memory:"
    object_dir: Path | None = None
    clock: Clock = field(default=default_clock)
    new_id: IdFactory = field(default=default_id_factory)
    _objects: dict[str, bytes] = field(default_factory=dict, init=False, repr=False)
    _conn: sqlite3.Connection = field(init=False, repr=False)
    _tx_depth: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        # check_same_thread=False: the control plane runs the (sync) commit path in a worker
        # thread while it awaits the async verifier (§3.4). It is the sole, serialized mutator
        # (Principle 9), so the connection is never touched concurrently — cross-thread use is safe.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        if self.object_dir is not None:
            self.object_dir.mkdir(parents=True, exist_ok=True)
        log.debug("store_opened", path=self.path, on_disk_objects=self.object_dir is not None)

    # -- transactions -------------------------------------------------------------

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """A single atomic unit, **re-entrant**: when several writes nest under one
        :meth:`transaction`, only the outermost commits (or rolls back), so a multi-write commit
        lands all-or-nothing (ROADMAP 5.1). Supersession/revision stay atomic with their cause (§6).
        """
        self._tx_depth += 1
        outermost = self._tx_depth == 1
        try:
            yield self._conn
        except Exception:
            if outermost:
                self._conn.rollback()
            raise
        else:
            if outermost:
                self._conn.commit()
        finally:
            self._tx_depth -= 1

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run several :class:`CommitSink` writes as one atomic unit (ROADMAP 5.1).

        All of a commit's decision rows + its terminal status write land together, or none do — a
        failure part-way through (a failed provenance check, an illegal transition, a store error)
        rolls the whole thing back instead of leaving decisions on a still-``proposed`` artifact.
        """
        with self._tx():
            yield

    # -- (de)serialization --------------------------------------------------------

    @staticmethod
    def _dump_payload(payload: Payload) -> str:
        if isinstance(payload, ObjectRef):
            ref = {"blob_ref": payload.blob_ref, "content_hash": payload.content_hash}
            return json.dumps({_OBJECT_MARKER: ref})
        return json.dumps(payload)

    @staticmethod
    def _load_payload(raw: str) -> Payload:
        value = json.loads(raw)
        if isinstance(value, dict) and _OBJECT_MARKER in value:
            ref = value[_OBJECT_MARKER]
            return ObjectRef(blob_ref=ref["blob_ref"], content_hash=ref["content_hash"])
        return cast("Payload", value)

    @staticmethod
    def _row_to_artifact(row: sqlite3.Row) -> Artifact:
        return Artifact(
            id=row["id"],
            type=row["type"],
            payload=SqliteStore._load_payload(row["payload"]),
            status=ArtifactStatus(row["status"]),
            created_by=row["created_by"],
            created_at=row["created_at"],
            superseded_by=row["superseded_by"],
            revised_by=row["revised_by"],
            is_root=bool(row["is_root"]),
            objects=_load_objects(row["objects"]),
        )

    @staticmethod
    def _row_to_operation(row: sqlite3.Row) -> Operation:
        return Operation(
            op_id=row["op_id"],
            op_name=row["op_name"],
            parents=tuple(json.loads(row["parents"])),
            output_id=row["output_id"],
            status=OperationStatus(row["status"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_decision(row: sqlite3.Row) -> Decision:
        raw_defects = row["defects"]
        return Decision(
            artifact_id=row["artifact_id"],
            gate=row["gate"],
            verdict=VerdictKind(row["verdict"]),
            rationale=row["rationale"],
            defects=tuple(json.loads(raw_defects)) if raw_defects is not None else None,
            score=row["score"],
            created_at=row["created_at"],
        )

    # -- reads --------------------------------------------------------------------

    def get_artifact(self, artifact_id: str) -> Artifact | None:
        row = self._conn.execute(
            "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
        ).fetchone()
        return self._row_to_artifact(row) if row is not None else None

    def query_artifacts(
        self, *, type: str | None = None, status: ArtifactStatus | None = None
    ) -> list[Artifact]:
        clauses: list[str] = []
        params: list[str] = []
        if type is not None:
            clauses.append("type = ?")
            params.append(type)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM artifacts{where} ORDER BY created_at, id", params
        ).fetchall()
        return [self._row_to_artifact(r) for r in rows]

    def operations_into(self, artifact_id: str) -> list[Operation]:
        rows = self._conn.execute(
            "SELECT * FROM operations WHERE output_id = ? ORDER BY created_at, op_id",
            (artifact_id,),
        ).fetchall()
        return [self._row_to_operation(r) for r in rows]

    def operations_out_of(self, artifact_id: str) -> list[Operation]:
        rows = self._conn.execute("SELECT * FROM operations ORDER BY created_at, op_id").fetchall()
        ops = [self._row_to_operation(r) for r in rows]
        return [op for op in ops if artifact_id in op.parents]

    def decisions_for(self, artifact_id: str) -> list[Decision]:
        rows = self._conn.execute(
            "SELECT * FROM decisions WHERE artifact_id = ? ORDER BY created_at",
            (artifact_id,),
        ).fetchall()
        return [self._row_to_decision(r) for r in rows]

    def current_schema_version(self) -> SchemaVersion | None:
        versions = self.schema_versions()
        return versions[-1] if versions else None

    def schema_versions(self) -> list[SchemaVersion]:
        rows = self._conn.execute("SELECT * FROM schema_versions ORDER BY version").fetchall()
        return [
            SchemaVersion(
                version=r["version"],
                types=tuple(json.loads(r["types"])),
                op_signatures=tuple(json.loads(r["op_signatures"])),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # -- propose (public; records an intention only, §4.2) ------------------------

    def propose(self, artifact: Artifact, operation: Operation) -> Artifact:
        if artifact.status is not ArtifactStatus.PROPOSED:
            raise StoreError(
                f"propose() requires status 'proposed', got '{artifact.status.value}'"
            )
        stamped = artifact if artifact.created_at else dataclasses.replace(
            artifact, created_at=self.clock()
        )
        with self._tx() as conn:
            self._assert_fresh_id(conn, stamped.id)
            self._assert_fresh_op_id(conn, operation.op_id)
            self._insert_artifact(conn, stamped)
            self._insert_operation(conn, operation)
        log.info("proposed", artifact_id=stamped.id, type=stamped.type, op=operation.op_name)
        return stamped

    # -- privileged writes (CommitSink; commit path only, §7) ---------------------

    def record_decision(self, decision: Decision) -> None:
        stamped = (
            decision
            if decision.created_at
            else dataclasses.replace(decision, created_at=self.clock())
        )
        with self._tx() as conn:
            self._insert_decision(conn, stamped)
        log.info(
            "decision_recorded",
            artifact_id=stamped.artifact_id,
            gate=stamped.gate,
            verdict=stamped.verdict.value,
        )

    def set_status(self, artifact_id: str, status: ArtifactStatus) -> None:
        with self._tx() as conn:
            self._require_artifact(conn, artifact_id)
            conn.execute(
                "UPDATE artifacts SET status = ? WHERE id = ?", (status.value, artifact_id)
            )
        log.info("status_set", artifact_id=artifact_id, status=status.value)

    def accept_superseding(self, new_id: str, old_id: str) -> None:
        """Accept ``new_id`` and supersede ``old_id`` atomically (spec §5.2, §6, §7.5a)."""
        with self._tx() as conn:
            self._require_artifact(conn, new_id)
            self._require_artifact(conn, old_id)
            conn.execute(
                "UPDATE artifacts SET status = ? WHERE id = ?",
                (ArtifactStatus.ACCEPTED.value, new_id),
            )
            conn.execute(
                "UPDATE artifacts SET status = ?, superseded_by = ? WHERE id = ?",
                (ArtifactStatus.SUPERSEDED.value, new_id, old_id),
            )
        log.info("accepted_superseding", new_id=new_id, old_id=old_id)

    def link_revision(self, old_id: str, new_id: str) -> None:
        """Point a ``revised`` artifact at its revision, atomically (spec §6, §7.5b)."""
        with self._tx() as conn:
            self._require_artifact(conn, old_id)
            self._require_artifact(conn, new_id)
            conn.execute(
                "UPDATE artifacts SET status = ?, revised_by = ? WHERE id = ?",
                (ArtifactStatus.REVISED.value, new_id, old_id),
            )
        log.info("revision_linked", old_id=old_id, new_id=new_id)

    def register_schema_version(self, schema: SchemaVersion) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO schema_versions (version, types, op_signatures, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    schema.version,
                    json.dumps(list(schema.types)),
                    json.dumps(list(schema.op_signatures)),
                    schema.created_at or self.clock(),
                ),
            )

    # -- objects (§4.2) -----------------------------------------------------------

    def put_object(self, data: bytes) -> ObjectRef:
        content_hash = hashlib.sha256(data).hexdigest()
        if self.object_dir is not None:
            (self.object_dir / content_hash).write_bytes(data)
        else:
            self._objects[content_hash] = data
        return ObjectRef(blob_ref=f"sha256:{content_hash}", content_hash=content_hash)

    def get_object(self, content_hash: str) -> bytes:
        if self.object_dir is not None:
            target = self.object_dir / content_hash
            if not target.exists():
                raise StoreError(f"object not found: {content_hash}")
            return target.read_bytes()
        if content_hash not in self._objects:
            raise StoreError(f"object not found: {content_hash}")
        return self._objects[content_hash]

    # -- provenance / audit (§4.2) ------------------------------------------------

    def get_provenance(self, artifact_id: str) -> Provenance:
        root = self.get_artifact(artifact_id)
        if root is None:
            raise StoreError(f"unknown artifact: {artifact_id}")
        seen_ops: dict[str, Operation] = {}
        seen_artifacts: dict[str, Artifact] = {}
        frontier = [artifact_id]
        while frontier:
            current = frontier.pop()
            for op in self.operations_into(current):
                if op.op_id in seen_ops:
                    continue
                seen_ops[op.op_id] = op
                for parent in op.parents:
                    if parent in seen_artifacts:
                        continue
                    parent_artifact = self.get_artifact(parent)
                    if parent_artifact is not None:
                        seen_artifacts[parent] = parent_artifact
                        frontier.append(parent)
        decisions: list[Decision] = list(self.decisions_for(artifact_id))
        for ancestor_id in seen_artifacts:
            decisions.extend(self.decisions_for(ancestor_id))
        return Provenance(
            artifact=root,
            operations=tuple(seen_ops.values()),
            ancestors=tuple(seen_artifacts.values()),
            decisions=tuple(decisions),
        )

    def rejected_log(self, *, type: str | None = None) -> list[Artifact]:
        return self.query_artifacts(type=type, status=ArtifactStatus.REJECTED)

    def superseded_log(self, *, type: str | None = None) -> list[Artifact]:
        return self.query_artifacts(type=type, status=ArtifactStatus.SUPERSEDED)

    # -- low-level helpers --------------------------------------------------------

    def _assert_fresh_id(self, conn: sqlite3.Connection, artifact_id: str) -> None:
        if conn.execute("SELECT 1 FROM artifacts WHERE id = ?", (artifact_id,)).fetchone():
            raise StoreError(f"id already in use (ids are never reused, §5.1): {artifact_id}")

    def _assert_fresh_op_id(self, conn: sqlite3.Connection, op_id: str) -> None:
        if conn.execute("SELECT 1 FROM operations WHERE op_id = ?", (op_id,)).fetchone():
            raise StoreError(f"op_id already in use (ids are never reused, §5.1): {op_id}")

    def _require_artifact(self, conn: sqlite3.Connection, artifact_id: str) -> None:
        if not conn.execute("SELECT 1 FROM artifacts WHERE id = ?", (artifact_id,)).fetchone():
            raise StoreError(f"unknown artifact: {artifact_id}")

    def _insert_artifact(self, conn: sqlite3.Connection, artifact: Artifact) -> None:
        conn.execute(
            "INSERT INTO artifacts "
            "(id, type, payload, status, created_by, created_at, superseded_by, revised_by, "
            "is_root, objects) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                artifact.id,
                artifact.type,
                self._dump_payload(artifact.payload),
                artifact.status.value,
                artifact.created_by,
                artifact.created_at,
                artifact.superseded_by,
                artifact.revised_by,
                int(artifact.is_root),
                _dump_objects(artifact.objects),
            ),
        )

    def _insert_operation(self, conn: sqlite3.Connection, operation: Operation) -> None:
        conn.execute(
            "INSERT INTO operations (op_id, op_name, parents, output_id, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                operation.op_id,
                operation.op_name,
                json.dumps(list(operation.parents)),
                operation.output_id,
                operation.status.value,
                operation.created_at or self.clock(),
            ),
        )

    def _insert_decision(self, conn: sqlite3.Connection, decision: Decision) -> None:
        conn.execute(
            "INSERT INTO decisions "
            "(artifact_id, gate, verdict, rationale, defects, score, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                decision.artifact_id,
                decision.gate,
                decision.verdict.value,
                decision.rationale,
                json.dumps(list(decision.defects)) if decision.defects is not None else None,
                decision.score,
                decision.created_at,
            ),
        )

    def close(self) -> None:
        self._conn.close()
