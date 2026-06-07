"""Stub verifier — a control-plane integration double (ROADMAP Phase 1.5; ADR 0001).

NON-PRODUCT. A scripted, in-process :class:`~verity.contracts.ports.VerifierPort`: it returns a
fixed :class:`VerdictBundle` and records every request it is handed. Since Phase 2.5 the real
verifier drives the happy path; this survives as a lightweight port-protocol double. It carries an
``identity`` distinct from the proposer (Builder/Breaker) and never evaluates anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verity.contracts import ArtifactStatus, VerdictBundle, VerifierRequest
from verity.logging import get_logger

__all__ = ["StubVerifier"]

log = get_logger("tools.harness.stub_verifier")


def _accept() -> VerdictBundle:
    return VerdictBundle(ArtifactStatus.ACCEPTED)


@dataclass
class StubVerifier:
    """A scripted, in-process :class:`VerifierPort` double (spec §3.6, ADR 0001). NON-PRODUCT."""

    identity: str = "stub-verifier"
    result: VerdictBundle = field(default_factory=_accept)
    requests: list[VerifierRequest] = field(default_factory=list)
    provisioned: bool = False

    async def dispatch(self, request: VerifierRequest) -> VerdictBundle:
        self.requests.append(request)
        log.info("stub_bundle", proposal=request.proposal.id, status=self.result.status.value)
        return self.result

    async def provision(self) -> None:
        self.provisioned = True

    async def teardown(self) -> None:
        self.provisioned = False

    async def health(self) -> bool:
        return True
