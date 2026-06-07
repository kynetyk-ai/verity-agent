"""Audit-contract invariants — properties that must hold over the store (spec §5).

Stable/unique/never-reused ids; append-only payloads; no hard delete of accepted
artifacts; provenance for every non-root accepted/tentative artifact; failures
recorded; no silent merge; no implicit accept; reproducible gate verdicts.

These ship as **property tests**, not prose (ROADMAP cross-cutting principle).

TODO (Phase 1.1): encode each invariant as an executable property/acceptance check.
"""
