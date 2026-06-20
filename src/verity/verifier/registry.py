"""Verifier-type registry + plugin loader (ROADMAP 9.3, ADR 0006) — the verifier-image twin of the
control-plane's task-type loader (:mod:`verity.composition.loader`).

The verifier image selects which verifier to serve by name (``VERITY_VERIFIER``). Those names are
**discovered** from the ``verity.verifier_types`` entry-point group at boot, not hardcoded — so a
new verifier ships its own image + an entry point with no edit to this package. Two shapes, matching
the service handshake:

* a **dataless impl** builder (``() -> VerifierPort``) — ready as soon as the service starts;
* a **data-bearing setup** builder (``VerifierSetup -> VerifierPort``) — built server-side from the
  control-plane's per-task :class:`VerifierSetup` (§9.1).

An entry point resolves to a ``register_verifier(registry)`` callable. Degrade-don't-crash, an
injectable discovery, and core-first ordering match the task loader exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib.metadata import EntryPoint, entry_points

from verity.contracts.ports import VerifierPort, VerifierSetup
from verity.logging import get_logger

__all__ = ["VERIFIER_TYPES_GROUP", "VerifierRegistry", "load_verifier_plugins"]

VERIFIER_TYPES_GROUP = "verity.verifier_types"
_CORE_DIST = "verity"

log = get_logger("verity.verifier.registry")

ImplBuilder = Callable[[], VerifierPort]
SetupBuilder = Callable[[VerifierSetup], VerifierPort]
EntryPointSource = Callable[[], Iterable[EntryPoint]]


class VerifierRegistry:
    """Verifier builders by name: dataless ``impl`` and data-bearing ``setup`` shapes."""

    def __init__(self) -> None:
        self._impls: dict[str, ImplBuilder] = {}
        self._setups: dict[str, SetupBuilder] = {}

    def register_impl(self, name: str, builder: ImplBuilder) -> None:
        """Register a dataless verifier (built immediately, served as a fixed impl)."""
        self._impls[name] = builder

    def register_setup(self, name: str, builder: SetupBuilder) -> None:
        """Register a data-bearing verifier (built from the per-task ``VerifierSetup``)."""
        self._setups[name] = builder

    def has(self, name: str) -> bool:
        return name in self._impls or name in self._setups

    def names(self) -> list[str]:
        """All registered verifier names (for ``VERITY_VERIFIER`` selection + error messages)."""
        return sorted({*self._impls, *self._setups})

    def impl(self, name: str) -> ImplBuilder | None:
        return self._impls.get(name)

    def setup(self, name: str) -> SetupBuilder | None:
        return self._setups.get(name)

    def merge(
        self, other: VerifierRegistry, *, on_collision: str = "skip", source: str = ""
    ) -> list[str]:
        """Fold ``other`` into self; incumbent wins on a collision. Returns added names."""
        added: list[str] = []
        for kind, src in (("impl", other._impls), ("setup", other._setups)):
            for name, builder in src.items():
                if self.has(name):
                    if on_collision == "skip":
                        continue
                    raise ValueError(
                        f"verifier {name!r} from {source!r} collides with an existing type"
                    )
                (self._impls if kind == "impl" else self._setups)[name] = builder  # type: ignore[assignment]
                added.append(name)
        return added


def _default_entry_points() -> Iterable[EntryPoint]:
    return entry_points(group=VERIFIER_TYPES_GROUP)


def _core_first(eps: Iterable[EntryPoint]) -> list[EntryPoint]:
    def key(ep: EntryPoint) -> tuple[int, str]:
        dist = getattr(ep, "dist", None)
        dist_name = getattr(dist, "name", "") or ""
        return (0 if dist_name == _CORE_DIST else 1, ep.name)

    return sorted(eps, key=key)


def load_verifier_plugins(
    registry: VerifierRegistry,
    *,
    entry_points: EntryPointSource = _default_entry_points,
) -> VerifierRegistry:
    """Discover ``verity.verifier_types`` plugins and merge each into ``registry``; returns it.

    Each entry point must resolve to a ``register_verifier(registry)`` callable. Incumbent wins on a
    name collision. Every failure degrades — never raises — so one broken plugin cannot stop the
    verifier service booting with the rest.
    """
    for ep in _core_first(entry_points()):
        _load_one(registry, ep)
    return registry


def _load_one(registry: VerifierRegistry, ep: EntryPoint) -> None:
    try:
        register = ep.load()
    except Exception as exc:  # noqa: BLE001 — a bad plugin must never crash the service
        log.warning(
            "verifier_plugin_load_failed",
            entry_point=ep.name,
            value=getattr(ep, "value", "?"),
            error=str(exc),
        )
        return
    if not callable(register):
        log.warning(
            "verifier_plugin_not_callable", entry_point=ep.name, resolved=type(register).__name__
        )
        return
    scratch = VerifierRegistry()
    try:
        register(scratch)
    except Exception as exc:  # noqa: BLE001
        log.warning("verifier_plugin_register_failed", entry_point=ep.name, error=str(exc))
        return
    for name in sorted(n for n in scratch.names() if registry.has(n)):
        log.warning(
            "verifier_plugin_collision", entry_point=ep.name, type=name, policy="incumbent_wins"
        )
    for name in registry.merge(scratch, on_collision="skip", source=ep.name):
        log.info("verifier_plugin_registered", entry_point=ep.name, type=name)
