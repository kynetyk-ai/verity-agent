"""The independence contract — a gate sees only the store-inputs it declared (spec §8.3, §10).

Separation of identity (proposer ≠ gate, §7.2) is necessary but not sufficient: a gate that can see
the goal, the proposer's reasoning, or sibling artifacts will *rationalize* rather than judge (§10,
[PILAR]). So each gate declares the **exact** set of store-inputs it may see, and the control plane
assembles only those and withholds everything else.

This complements the type-level guarantee from :mod:`verity.contracts`: the store-slice is a tuple
of :class:`Artifact`, which carries no rationale field, so the *proposer's reasoning* cannot ride
along by construction. That closes the rationale channel; this closes the over-broad-context
channel — the gate is denied sibling state it did not ask for.

The declaration is itself a **checkable property** (§10), a sibling of the audit invariants (§5):
the resolved slice's sources are a subset of the gate's declared allowlist, asserted before
dispatch, so an over-broad gate fails before it ever runs.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from verity.contracts import Artifact, ArtifactStatus
from verity.control_plane.store import Store
from verity.logging import get_logger

__all__ = [
    "StoreInput",
    "IndependenceViolation",
    "STORE_INPUT_PROVIDERS",
    "resolve_declared_slice",
]

log = get_logger("verity.control_plane.independence")


class IndependenceViolation(RuntimeError):
    """A gate's resolved inputs exceeded — or could not be cut to — its declared allowlist (§10)."""


class StoreInput(StrEnum):
    """A declarable kind of store-slice a gate may request (spec §8.3).

    The control plane knows how to cut each of these from the store; a gate's binding names the
    subset it needs, and nothing else reaches it. ``INCUMBENTS`` are the accepted artifacts of the
    same type (the field to beat); ``REJECTED_LOG`` is the rejected artifacts of the same type (the
    trial count, §12). A gate that declares **none** sees only the artifact under test — the
    tightest independence, right for a self-contained validity check.
    """

    INCUMBENTS = "incumbents"
    REJECTED_LOG = "rejected_log"


# How the control plane cuts each declarable input from the store. Keyed by StoreInput so a declared
# input with no provider is a loud error, never a silently-empty slice.
STORE_INPUT_PROVIDERS: dict[StoreInput, Callable[[Store, Artifact], list[Artifact]]] = {
    StoreInput.INCUMBENTS: lambda store, artifact: store.query_artifacts(
        type=artifact.type, status=ArtifactStatus.ACCEPTED
    ),
    StoreInput.REJECTED_LOG: lambda store, artifact: store.rejected_log(type=artifact.type),
}


def resolve_declared_slice(
    store: Store, artifact: Artifact, declared: frozenset[StoreInput]
) -> tuple[Artifact, ...]:
    """Assemble exactly the store-inputs a gate declared, and assert nothing else leaks (spec §10).

    Returns the artifacts the gate is allowed to see — the union of its declared inputs, and an
    **empty** tuple when it declared none (only the artifact under test reaches it). Raises
    :class:`IndependenceViolation` if a declared input has no provider, or — the §10 audit — if the
    resolved slice ever carries a source outside the declared allowlist.
    """
    tagged: list[tuple[StoreInput, Artifact]] = []
    for kind in declared:
        provider = STORE_INPUT_PROVIDERS.get(kind)
        if provider is None:
            raise IndependenceViolation(
                f"gate declares store-input {kind!r}, which the control plane cannot provide"
            )
        tagged.extend((kind, art) for art in provider(store, artifact))

    # The §10 check: every resolved item's source must be within the declared allowlist. Held by
    # construction above; asserted here so a future change to slice assembly cannot leak silently.
    leaked = {kind for kind, _ in tagged if kind not in declared}
    if leaked:
        raise IndependenceViolation(
            f"resolved slice for {artifact.id!r} carries undeclared inputs {sorted(leaked)} "
            f"(declared: {sorted(declared)})"
        )
    log.debug(
        "slice_resolved", artifact=artifact.id, declared=sorted(declared), size=len(tagged)
    )
    return tuple(art for _, art in tagged)
