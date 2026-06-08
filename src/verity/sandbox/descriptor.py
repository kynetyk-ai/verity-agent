"""The proposal descriptor — the driver -> core boundary contract (ROADMAP Phase 3).

The agent in the sandbox never mints ids or touches the control plane. A bound operation tool writes
a small JSON **descriptor** — ``{op_name, parents, payload, metadata}`` — to a reserved file in the
outbox (:data:`RESERVED_PROPOSAL_NAME`); object attachments (a script, a data file) are ordinary
outbox files. The trusted host (:class:`~verity.sandbox.core.AgentSandbox`) then harvests it,
splits this descriptor from the attachments, and mints the typed ``Artifact`` + ``Operation``.

This keeps id/provenance minting on the trusted side: a (possibly hostile, possibly containerized)
agent only *declares* the operation, its parents, and the payload — it cannot forge ids or lineage.
Every driver (in-process, container, a future framework) honours this one contract, so the host-side
harvest/mint code is shared across all of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from verity.contracts import JSONValue
from verity.sandbox.errors import SandboxError

__all__ = ["RESERVED_PROPOSAL_NAME", "ProposalDescriptor"]

# The reserved outbox filename the propose tool writes; filtered out of the harvested object set.
RESERVED_PROPOSAL_NAME = "__proposal__.json"


@dataclass(frozen=True, slots=True)
class ProposalDescriptor:
    """What the agent *declares* about its proposal — minted into an Artifact+Operation by the host.

    ``payload`` is the domain payload stored verbatim; ``parents`` are the input artifact ids the
    agent referenced from its served context; ``metadata`` is the segregated rationale (provenance
    only, never reaches the verifier).
    """

    op_name: str
    parents: tuple[str, ...]
    payload: JSONValue
    metadata: str = ""

    def to_json(self) -> bytes:
        return json.dumps(
            {
                "op_name": self.op_name,
                "parents": list(self.parents),
                "payload": self.payload,
                "metadata": self.metadata,
            },
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> ProposalDescriptor:
        try:
            obj = json.loads(data)
        except json.JSONDecodeError as exc:
            raise SandboxError(f"proposal descriptor is not valid JSON: {exc}") from exc
        if not isinstance(obj, dict) or "op_name" not in obj:
            raise SandboxError("proposal descriptor must be an object with an 'op_name'")
        parents_raw = obj.get("parents", [])
        if not isinstance(parents_raw, list):
            raise SandboxError("proposal descriptor 'parents' must be a list")
        return cls(
            op_name=str(obj["op_name"]),
            parents=tuple(str(p) for p in parents_raw),
            payload=obj.get("payload"),
            metadata=str(obj.get("metadata", "")),
        )
