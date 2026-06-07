"""The lifecycle state machine — legal status transitions and who may trigger them (spec §6).

``proposed → {tentative | rejected | revised}``; ``tentative → {accepted | rejected |
revised | superseded}``; ``accepted → superseded``. Only the commit path triggers
transitions out of ``proposed``.

TODO (Phase 1.1): explicit, testable transition table.
"""
