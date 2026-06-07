# tools/harness — control-plane integration doubles

**Non-product test tooling.** These are throwaway stand-ins for the real services, built so
the **control plane** (ROADMAP Phase 1) can be exercised and validated end-to-end before the
real sandbox and verifier exist.

- `stub_agent.py` — plays the sandbox/agent (spec §3.5): reads served context, emits scripted
  proposals, writes object attachments to the outbox, handles shape-error / refine feedback.
- `stub_verifier.py` — plays the advisory verifier (spec §3.6): takes a shaped proposal +
  store-slice + object attachments, returns a scripted verdict.

They are deliberately dumb (scripted, deterministic, no LLM, no real evaluation). The control
plane is designed to be ignorant of how the agent and verifier work internally, so these
suffice to prove its contracts. They are retired from the happy path as the real services land
(Phase 2: verifier, Phase 3: sandbox), but may be kept as deterministic integration fixtures.
