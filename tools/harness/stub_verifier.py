"""Stub verifier — a control-plane integration double (ROADMAP Phase 1.5).

NON-PRODUCT. Stands in for the real advisory verifier (spec §3.6): receives a shaped
proposal plus its declared store-slice and any object attachments, and returns a *scripted*
verdict (``accept`` / ``reject`` / ``refine``, with rationale and optional defects). No
real evaluation. Used to drive every commit-path branch deterministically.

TODO (Phase 1.5): scripted-verdict verifier behind the control plane's verifier-dispatch API.
"""
