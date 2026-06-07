"""The commit path — the single privileged route from proposal to durable state (spec §7).

Control-plane-orchestrated protocol: validate shape → resolve gate binding (no binding =
hard error) → dispatch to the advisory verifier → record decisions → advance status →
resolve supersession / refine → record provenance. The only code permitted to set status
or write decisions; the verifier advises but never writes.

TODO (Phase 1.1): implement the protocol; enforce every §5/§6 guarantee here.
"""
