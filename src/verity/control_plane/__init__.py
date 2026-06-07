"""The control plane — the sole mutator of durable state (spec §3.4).

Deliberately unintelligent: it runs the mechanical commit protocol, assembles context,
and applies a declarative orchestration policy; it hosts no LLM and makes no judgment
calls. This package is the ROADMAP Phase 1 deliverable. Modules here are documented
stubs until Phase 1 lands their real signatures + tests.
"""
