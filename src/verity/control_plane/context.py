"""Context assembly — regenerate the volatile context from the store each turn (spec §9).

Stable prefix (the composed system prompt + tool defs + slow-changing manifest) plus a
volatile tail (artifacts retrieved for this step). Regenerate-never-append; flush-on-commit.
Carries the bounded-context guarantee: assembled context size must not grow with store size.

TODO (Phase 1.3): stable-prefix/volatile-tail split, manifest, bounded-context test.
"""
