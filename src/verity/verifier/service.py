"""The advisory, opaque verifier service (spec §3.6, ADR 0001).

:class:`SdkVerifier` is the real in-process verifier. Unlike the per-gate-dispatch model it
replaces, it **owns its pipeline**: given a proposal it runs its own ordered checks (cheap then
hard) and returns a single :class:`VerdictBundle` — the terminal status it recommends plus the
per-check decisions. The control plane does not sequence these checks or parse the proposal; it
records the bundle and enforces the invariants (§7, ADR 0001).

The §6 staging lives **here** now: clearing the cheap checks earns a `tentative` baseline; a hard
check that cannot auto-resolve (a ``requires_human`` plugin returning ``None``) rests at
`tentative`; clearing every hard check earns `accepted`. The first reject/refine short-circuits. A
selection check may name an incumbent it beats (``GateVerdict.supersedes``), surfaced on the bundle.

The verifier carries an ``identity`` distinct from the proposer (Builder/Breaker, §7.2). A proposal
of a type this verifier has no pipeline for is a configuration error, never a silent pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verity.contracts import (
    ArtifactStatus,
    GateDecision,
    GateVerdict,
    RunContext,
    SupportsRunContext,
    VerdictBundle,
    VerdictKind,
    VerifierRequest,
)
from verity.logging import get_logger
from verity.verifier.errors import VerifierError
from verity.verifier.primitives import GatePrimitive

__all__ = ["GateStep", "SdkVerifier"]

log = get_logger("verity.verifier.service")


@dataclass(frozen=True, slots=True)
class GateStep:
    """One step in a verifier's internal pipeline: a named check + its plugin (ADR 0001, §8.3).

    ``is_hard`` splits the pipeline — clearing the cheap (``is_hard=False``) checks earns the
    `tentative` baseline, clearing the hard checks earns `accepted`. ``requires_human`` marks a
    check that must not auto-resolve (its plugin returns ``None`` → rest at `tentative`).
    """

    gate: str
    plugin: GatePrimitive
    is_hard: bool = False
    requires_human: bool = False


@dataclass
class SdkVerifier:
    """An advisory, opaque :class:`VerifierPort` over per-type pipelines (spec §3.6, ADR 0001).

    ``pipelines`` maps an artifact type to its ordered checks. ``requests`` records every dispatched
    request so a test can assert the independence contract (a rationale-free slice, §10) and that
    object attachments arrive (§3.4).
    """

    identity: str
    pipelines: dict[str, tuple[GateStep, ...]]
    requests: list[VerifierRequest] = field(default_factory=list)
    provisioned: bool = False
    # Resources this verifier provisions work on (e.g. a backend-backed code-runner) that want the
    # control plane's run identity stamped onto their workers (7.4.h). A gate's runner registers
    # here; a pure in-process gate registers nothing.
    context_sinks: tuple[SupportsRunContext, ...] = ()

    def __post_init__(self) -> None:
        # G1 — no silent accept: a gated type's pipeline must declare at least one *hard* gate.
        # Without one, `_run_pipeline` would clear the (possibly empty) cheap stage and fall
        # straight through to ACCEPTED having ruled on no hard check — exactly the implicit accept
        # the architecture forbids (§5.7). Caught at construction so it's a loud config error, not a
        # silent runtime pass.
        for atype, pipeline in self.pipelines.items():
            if not any(step.is_hard for step in pipeline):
                raise VerifierError(
                    f"pipeline for type {atype!r} declares no hard gate: a gated type with no hard "
                    f"check would accept unconditionally (no implicit accept, §5.7)"
                )

    def bind_run_context(self, ctx: RunContext) -> None:
        """Forward the control plane's run-identity cell to every resource that stamps it."""
        for sink in self.context_sinks:
            sink.bind_run_context(ctx)

    async def dispatch(self, request: VerifierRequest) -> VerdictBundle:
        self.requests.append(request)
        pipeline = self.pipelines.get(request.proposal.type)
        if pipeline is None:
            raise VerifierError(
                f"this verifier has no pipeline for type {request.proposal.type!r} "
                f"(covers: {sorted(self.pipelines)}): a covered type with no pipeline is a config "
                f"error, never a silent pass (§3.6, §5.7)"
            )
        bundle = await self._run_pipeline(pipeline, request)
        log.info(
            "bundle",
            type=request.proposal.type,
            proposal=request.proposal.id,
            status=bundle.status.value,
            decisions=[d.gate for d in bundle.decisions],
            objects=sorted(request.objects),
            slice_size=len(request.store_slice),
        )
        return bundle

    async def _run_pipeline(
        self, pipeline: tuple[GateStep, ...], request: VerifierRequest
    ) -> VerdictBundle:
        decisions: list[GateDecision] = []
        # G2 — supersession is the authoritative *hard*-stage decision's call, not last-writer-wins.
        # Only a hard ACCEPT may retire an incumbent, and at most one distinct incumbent may be
        # named; a cheap gate naming a supersedes, or two hard gates naming different incumbents, is
        # a verifier inconsistency, not a silently-applied retirement.
        superseded: set[str] = set()
        cheap = [s for s in pipeline if not s.is_hard]
        hard = [s for s in pipeline if s.is_hard]

        for stage in (cheap, hard):
            ruled = 0
            for step in stage:
                verdict = await step.plugin(request)
                if verdict is None:
                    continue  # cannot auto-resolve (requires_human); no decision recorded
                ruled += 1
                decisions.append(_decision(step, verdict))
                if verdict.kind is VerdictKind.REJECT:
                    return VerdictBundle(ArtifactStatus.REJECTED, tuple(decisions))
                if verdict.kind is VerdictKind.REFINE:
                    return VerdictBundle(ArtifactStatus.REVISED, tuple(decisions))
                if step.is_hard and verdict.supersedes is not None:
                    superseded.add(verdict.supersedes)
            if stage is hard and ruled < len(hard):
                # a hard check deferred to a human — cleared the cheap checks, not the hard ones
                return VerdictBundle(ArtifactStatus.TENTATIVE, tuple(decisions))

        if len(superseded) > 1:
            raise VerifierError(
                f"pipeline accepted {request.proposal.id!r} but its hard gates named more than one "
                f"incumbent to supersede ({sorted(superseded)}): supersession must be unambiguous"
            )
        supersedes = next(iter(superseded), None)
        return VerdictBundle(ArtifactStatus.ACCEPTED, tuple(decisions), supersedes=supersedes)

    async def provision(self) -> None:
        self.provisioned = True

    async def teardown(self) -> None:
        self.provisioned = False

    async def health(self) -> bool:
        return True


def _decision(step: GateStep, verdict: GateVerdict) -> GateDecision:
    return GateDecision(
        gate=step.gate,
        kind=verdict.kind,
        rationale=verdict.rationale,
        defects=verdict.defects,
        score=verdict.score,
    )
