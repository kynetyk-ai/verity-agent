"""The extension-point registries (spec §8).

Schema registry (artifact types + typed operation signatures, versioned); tool registry
(typed plugins); gate registry (bindings layered over gate plugins — bindings live here,
plugins live in the verifier); retrieval/planner policy. The control plane resolves
bindings and enforces "no implicit accept."

TODO (Phase 1.2): typed registry interfaces + a trivial fake domain (incl. a gateless type).
"""
