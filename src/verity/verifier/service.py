"""The advisory verifier service (spec §3.6).

:class:`SdkVerifier` is the real, in-process verifier: an SDK of gate **plugins** (built from the
primitives in :mod:`.primitives`) behind the :class:`~verity.contracts.ports.VerifierPort`. It
holds a map of **gate name → plugin** and, on :meth:`dispatch`, runs the named plugin over the
shaped proposal, its declared store-slice, and any object attachments — returning the verdict the
commit path acts on. It never writes to the store and never decides whether to continue (§3.6).

The split with the control plane is exactly §8.3's: the **binding** (which gates run, in what order,
cheap-vs-hard) lives in the control plane's gate registry; the **plugins** (what each gate decides)
live here. The two meet on the gate *name* — the one piece of the binding the request carries.

A gate named in a binding but absent from this verifier's plugin map is a configuration error,
raised as :class:`~verity.verifier.errors.VerifierError` — the verifier-side "no implicit accept":
an unknown gate never silently passes (§3.6, §5.7).

In-process today; the async + lifecycle shape is the queue-fronted networked service's shape, so
fronting it with a real queue later is an added adapter, not a reshape (§3.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verity.contracts import GateVerdict, VerifierRequest
from verity.logging import get_logger
from verity.verifier.errors import VerifierError
from verity.verifier.primitives import GatePrimitive

__all__ = ["SdkVerifier"]

log = get_logger("verity.verifier.service")


@dataclass
class SdkVerifier:
    """An advisory :class:`VerifierPort` over a map of gate name → plugin (spec §3.6).

    ``plugins`` is the domain's composed gate set (built from :mod:`.primitives`). ``requests``
    records every dispatched request so a test can assert the independence contract (a
    rationale-free slice, §10) and that object attachments arrive (§3.4).
    """

    plugins: dict[str, GatePrimitive]
    requests: list[VerifierRequest] = field(default_factory=list)
    provisioned: bool = False

    async def dispatch(self, request: VerifierRequest) -> GateVerdict | None:
        self.requests.append(request)
        try:
            plugin = self.plugins[request.gate]
        except KeyError:
            raise VerifierError(
                f"no gate plugin registered for gate {request.gate!r} "
                f"(registered: {sorted(self.plugins)}): an unknown gate is a config error, "
                f"never a silent accept (§3.6, §5.7)"
            ) from None
        verdict = await plugin(request)
        log.info(
            "verdict",
            gate=request.gate,
            proposal=request.proposal.id,
            verdict=verdict.kind.value if verdict is not None else "deferred",
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
