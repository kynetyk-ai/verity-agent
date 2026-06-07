"""The store — durable source of truth, behind a backend-agnostic interface (spec §4).

artifacts · operations · decisions · schema_versions, plus the object store
(``put_object`` / ``get_object`` by content hash). Default reference backend: SQLite;
git-on-disk JSON is a valid v0.

TODO (Phase 1.1): Store interface + data model; object store; provenance/audit queries.
"""
