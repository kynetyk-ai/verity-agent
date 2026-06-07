"""Service ports and the provider registry (spec §3.3–§3.6).

The control plane talks to the sandbox and the verifier only through these **ports**, and it
selects a concrete implementation **by task config** through a :class:`ProviderRegistry`. This
is the "designed-for, not built" seam: the MVP ships exactly one adapter behind each port (the
Phase 1.5 stubs), and a second harness or a second verifier microservice is an *added adapter*
registered under a new key — not a rewrite. Two properties are deliberate:

* **Async by design** (§3.4). Every port call is ``async`` so that today's in-process stub and
  tomorrow's network/queue-fronted service share one shape; a network hop is not a refactor.
* **A lifecycle surface** (``provision`` / ``teardown`` / ``health``). The control plane may, in
  the future, own the lifecycle of the agent loop *and* the verifier (start, stop, health-check
  multiple microservices). The in-process stubs satisfy these as no-ops; we do not build a
  process supervisor now.

**The rationale channel is segregated here.** A :class:`ProposalEnvelope` carries the agent's
``metadata`` (its how/why summary) for provenance/context, but a :class:`VerifierRequest`
**never** does: the verifier is handed the proposal, its object attachments, and the declared,
control-plane-cut store-slice — and nothing of the proposer's reasoning (§10, Principle 6). The
store-slice is a tuple of :class:`Artifact`, which carries no rationale field by construction,
so "no proposer rationale reaches a gate" holds at the type level, not by discipline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from verity.control_plane.commit import GateVerdict
from verity.control_plane.store import Artifact, Operation
from verity.logging import get_logger

__all__ = [
    "ServiceLifecycle",
    "VerifierRequest",
    "VerifierPort",
    "ProposalEnvelope",
    "ServedContext",
    "SandboxPort",
    "ProviderRegistry",
    "ProviderError",
    "SANDBOX_PROVIDERS",
    "VERIFIER_PROVIDERS",
]

log = get_logger("verity.control_plane.ports")


class ProviderError(RuntimeError):
    """Raised when a provider key is unknown or already registered."""


# ------------------------------------------------------------------------- lifecycle


@runtime_checkable
class ServiceLifecycle(Protocol):
    """The optional lifecycle a control plane may drive for a service (§3.4, §3.9).

    The in-process stubs implement these as no-ops; a future networked service implements real
    provisioning/health. Kept narrow so it can be a no-op today without shaping anything.
    """

    async def provision(self) -> None: ...
    async def teardown(self) -> None: ...
    async def health(self) -> bool: ...


# ------------------------------------------------------------------- verifier port (§3.6)


@dataclass(frozen=True, slots=True)
class VerifierRequest:
    """What the control plane hands the verifier — and nothing more (spec §3.6, §8.3, §10).

    ``store_slice`` is the declared, control-plane-cut slice (incumbents to beat, the
    rejected-log, …); it is a tuple of :class:`Artifact`, which has no rationale field, so the
    proposer's reasoning cannot ride along. ``objects`` are the harvested object attachments the
    gate may need to *execute* (e.g. submitted code), keyed by content hash (§3.4).
    """

    proposal: Artifact
    gate: str
    store_slice: tuple[Artifact, ...] = ()
    objects: Mapping[str, bytes] = field(default_factory=dict)


@runtime_checkable
class VerifierPort(Protocol):
    """The advisory verifier service, behind one async method plus the lifecycle (§3.6).

    ``dispatch`` is the **only** path to a gate verdict; the sandbox has none. Returns the same
    :class:`GateVerdict` the commit path consumes, so a verifier-backed
    :class:`~verity.control_plane.commit.GateRunner` is a thin adapter over this (wired in 1.4).
    """

    async def dispatch(self, request: VerifierRequest) -> GateVerdict: ...
    async def provision(self) -> None: ...
    async def teardown(self) -> None: ...
    async def health(self) -> bool: ...


# -------------------------------------------------------------------- sandbox port (§3.5)


@dataclass(frozen=True, slots=True)
class ServedContext:
    """The assembled context served to the sandbox at the start of a cycle (§9).

    A placeholder shape for the seam; context assembly (§9, Phase 1.3) fills in the
    stable-prefix / volatile-tail structure. Held here so the port signature is stable.
    """

    system_prompt: str
    manifest: str = ""
    tail: str = ""


@dataclass(frozen=True, slots=True)
class ProposalEnvelope:
    """A proposal as it arrives from the sandbox (spec §3.5, §10).

    ``metadata`` is the agent's how/why summary — stored as provenance/context, and **kept off**
    the :class:`VerifierRequest` (the rationale channel, decision recorded for this engagement).
    ``objects`` are what the agent wrote to the outbox, harvested before teardown (§3.4).
    """

    artifact: Artifact
    operation: Operation
    metadata: str = ""
    objects: Mapping[str, bytes] = field(default_factory=dict)


@runtime_checkable
class SandboxPort(Protocol):
    """The ephemeral agent-runtime + workspace, behind async calls plus the lifecycle (§3.5).

    ``serve_context`` hands the assembled context in; ``collect_proposal`` takes the proposal
    (with its outbox objects) out; ``regenerate`` rebuilds the workspace and flushes chat — one
    ephemerality event (§3.5, §9). The sandbox never reaches the verifier or the store.
    """

    async def serve_context(self, context: ServedContext) -> None: ...
    async def collect_proposal(self) -> ProposalEnvelope: ...
    async def regenerate(self) -> None: ...
    async def provision(self) -> None: ...
    async def teardown(self) -> None: ...
    async def health(self) -> bool: ...


# --------------------------------------------------------------------- provider registry


class ProviderRegistry[PortT]:
    """A config-keyed registry of port providers (decision: ports + registry + async).

    A task config names a provider by ``key`` (e.g. ``sandbox_key`` / ``verifier_key``); the
    control plane calls :meth:`create` to instantiate it. Factories are zero-arg today; a later
    phase may thread task config through them — an additive change, not a reshape.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._factories: dict[str, Callable[[], PortT]] = {}

    def register(self, key: str, factory: Callable[[], PortT]) -> None:
        if key in self._factories:
            raise ProviderError(f"{self._kind} provider already registered: {key!r}")
        self._factories[key] = factory
        log.debug("provider_registered", kind=self._kind, key=key)

    def create(self, key: str) -> PortT:
        try:
            factory = self._factories[key]
        except KeyError:
            raise ProviderError(
                f"unknown {self._kind} provider: {key!r} (registered: {sorted(self._factories)})"
            ) from None
        return factory()

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


# Module-level registries the stubs (1.5) and real services (2/3) register into.
SANDBOX_PROVIDERS: ProviderRegistry[SandboxPort] = ProviderRegistry("sandbox")
VERIFIER_PROVIDERS: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
