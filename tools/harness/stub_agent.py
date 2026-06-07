"""Stub agent / workspace — a control-plane integration double (ROADMAP Phase 1.5).

NON-PRODUCT. Stands in for the real sandbox (spec §3.5): reads the served context, emits
*scripted* proposals, writes object attachments to the outbox, and responds to shape-error
and refine feedback — all deterministically, with no LLM. Used to exercise the control
plane's propose → gate → commit cycle and the object harvest path.

TODO (Phase 1.5): drive the control-plane API with scripted proposal sequences.
"""
