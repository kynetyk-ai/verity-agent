"""Integration test doubles for the control plane (ROADMAP Phase 1.5).

NON-PRODUCT. These stand in for the real sandbox/agent and verifier so the control plane
can be exercised end-to-end before those services exist. They are deliberately dumb:
scripted, deterministic, no LLM and no real evaluation. Later phases retire them as the
real services land (Phase 2: verifier, Phase 3: sandbox).
"""
