"""Stub verifier — a control-plane integration double (ROADMAP Phase 1.5).

NON-PRODUCT. Stands in for the real advisory verifier (spec §3.6): receives a shaped proposal
plus its declared store-slice and any object attachments, and returns a *scripted* verdict
(``accept`` / ``reject`` / ``refine``, with rationale and optional defects). No real evaluation.
Used to drive every commit-path branch deterministically.

It implements :class:`~verity.contracts.ports.VerifierPort`, records every request it is
handed (so tests can assert the independence contract — a rationale-free slice, §10 — and that
object attachments arrive), and resolves a verdict through an injected ``responder`` callable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from verity.contracts import GateVerdict, VerdictKind, VerifierRequest
from verity.logging import get_logger

__all__ = ["Responder", "StubVerifier", "accept_all", "by_gate"]

log = get_logger("tools.harness.stub_verifier")

# A scripted verdict function: given the request, return the verdict to advise.
Responder = Callable[[VerifierRequest], GateVerdict]


def accept_all(request: VerifierRequest) -> GateVerdict:
    """The default responder: every gate passes."""
    return GateVerdict(VerdictKind.ACCEPT, f"stub accepts {request.gate}")


def by_gate(verdicts: dict[str, GateVerdict], *, default: GateVerdict | None = None) -> Responder:
    """A responder that maps gate name → verdict, falling back to ``default`` (or accept)."""
    fallback = default if default is not None else GateVerdict(VerdictKind.ACCEPT, "stub default")

    def responder(request: VerifierRequest) -> GateVerdict:
        return verdicts.get(request.gate, fallback)

    return responder


@dataclass
class StubVerifier:
    """A scripted, in-process :class:`VerifierPort` (spec §3.6). NON-PRODUCT."""

    responder: Responder = accept_all
    requests: list[VerifierRequest] = field(default_factory=list)
    provisioned: bool = False

    async def dispatch(self, request: VerifierRequest) -> GateVerdict:
        # The verifier consumes the object attachments it is handed (it never reaches the sandbox).
        self.requests.append(request)
        verdict = self.responder(request)
        log.info(
            "stub_verdict",
            gate=request.gate,
            proposal=request.proposal.id,
            verdict=verdict.kind.value,
            objects=sorted(request.objects),
            slice_size=len(request.store_slice),
        )
        return verdict

    async def provision(self) -> None:
        self.provisioned = True

    async def teardown(self) -> None:
        self.provisioned = False

    async def health(self) -> bool:
        return True
