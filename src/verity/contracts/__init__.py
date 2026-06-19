"""verity.contracts — the published cross-service vocabulary (spec §3.3).

The shared kernel both halves of the system depend on, owned by neither: the boundary-crossing
value model (:mod:`.model`) and the service ports + provider registry (:mod:`.ports`). The control
plane, the verifier, and the sandbox each depend on *this*, not on one another — so a second
verifier microservice or a second harness is an added adapter against a fixed contract, and the
dependency graph never points service-to-service.

Control-plane-internal types (the store interfaces, ``Decision``, ``Provenance``, the commit
machinery, the registries) stay in :mod:`verity.control_plane`; only what crosses a wire lives here.
"""

from __future__ import annotations

from verity.contracts.errors import GateUnavailable
from verity.contracts.model import (
    Artifact,
    ArtifactStatus,
    GateDecision,
    GateVerdict,
    JSONValue,
    ObjectRef,
    Operation,
    OperationStatus,
    Payload,
    VerdictBundle,
    VerdictKind,
)
from verity.contracts.ports import (
    SANDBOX_PROVIDERS,
    VERIFIER_PROVIDERS,
    ProposalEnvelope,
    ProviderError,
    ProviderRegistry,
    SandboxPort,
    ServedContext,
    ServiceLifecycle,
    VerifierPort,
    VerifierRequest,
    VerifierSetup,
)
from verity.contracts.run_context import RunContext, SupportsRunContext

__all__ = [
    # model
    "ArtifactStatus",
    "OperationStatus",
    "VerdictKind",
    "JSONValue",
    "ObjectRef",
    "Payload",
    "Artifact",
    "Operation",
    "GateVerdict",
    "GateDecision",
    "VerdictBundle",
    # ports
    "ServiceLifecycle",
    "VerifierSetup",
    "VerifierRequest",
    "VerifierPort",
    "ProposalEnvelope",
    "ServedContext",
    "SandboxPort",
    "ProviderRegistry",
    "ProviderError",
    "SANDBOX_PROVIDERS",
    "VERIFIER_PROVIDERS",
    # run identity
    "RunContext",
    "SupportsRunContext",
    # errors
    "GateUnavailable",
]
