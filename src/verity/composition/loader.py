"""Plugin loader (ROADMAP 9.3, ADR 0004 (i) / ADR 0006): discover task-type builders at boot.

Discovers entry points in the ``verity.task_types`` group and merges each into a `TaskCatalog`, so a
new task type is added by **installing a package that advertises the entry point — no control-plane
rebuild**. The loader is generic: it imports no domain; a discovered plugin's *own* package is the
only place a domain is imported (lazily, when its entry point loads).

Degrade-don't-crash: a plugin that fails to import/resolve, is not callable, raises while
registering, or collides with an existing type is logged (structured) and **skipped** — the daemon
still boots. Discovery is **injectable** (the ``entry_points`` argument) so the unit tests pass fake
entry points with no install. The core ``verity`` distribution's entry points load **first**, so the
built-ins keep priority under the incumbent-wins collision policy.

An entry point resolves to a ``register(catalog) -> None`` callable (the same shape as
:meth:`TaskCatalog.register`), so a plugin may register one or several types and decide whether to
pass a describer. The entry-point *name* is diagnostic only; the authoritative type name is whatever
the plugin passes to ``catalog.register``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib.metadata import EntryPoint, entry_points

from verity.composition.catalog import TaskCatalog
from verity.logging import get_logger

__all__ = ["TASK_TYPES_GROUP", "load_task_plugins"]

TASK_TYPES_GROUP = "verity.task_types"
_CORE_DIST = "verity"

log = get_logger("verity.composition.loader")

#: How the loader discovers entry points — injectable so tests can pass fakes (no install needed).
EntryPointSource = Callable[[], Iterable[EntryPoint]]


def _default_entry_points() -> Iterable[EntryPoint]:
    """The stdlib discovery (Python >=3.12: the selectable ``group=`` API is stable)."""
    return entry_points(group=TASK_TYPES_GROUP)


def _core_first(eps: Iterable[EntryPoint]) -> list[EntryPoint]:
    """Deterministic order: the core ``verity`` distribution's entry points first (so the built-ins
    keep collision priority under incumbent-wins), then the rest by name."""

    def key(ep: EntryPoint) -> tuple[int, str]:
        dist = getattr(ep, "dist", None)  # EntryPoint.dist is available on Python >=3.10
        dist_name = getattr(dist, "name", "") or ""
        return (0 if dist_name == _CORE_DIST else 1, ep.name)

    return sorted(eps, key=key)


def load_task_plugins(
    catalog: TaskCatalog,
    *,
    entry_points: EntryPointSource = _default_entry_points,
) -> TaskCatalog:
    """Discover ``verity.task_types`` plugins and merge each into ``catalog``; returns ``catalog``.

    Each entry point must resolve to a ``register(catalog)`` callable. The incumbent wins on a name
    collision (the colliding plugin type is logged and skipped). Every failure degrades — never
    raises — so one broken plugin cannot stop the daemon booting with the rest.
    """
    for ep in _core_first(entry_points()):
        _load_one(catalog, ep)
    return catalog


def _load_one(catalog: TaskCatalog, ep: EntryPoint) -> None:
    try:
        register = ep.load()
    except Exception as exc:  # noqa: BLE001 — a bad plugin must never crash the daemon
        log.warning(
            "task_plugin_load_failed",
            entry_point=ep.name,
            value=getattr(ep, "value", "?"),
            error=str(exc),
        )
        return
    if not callable(register):
        log.warning(
            "task_plugin_not_callable", entry_point=ep.name, resolved=type(register).__name__
        )
        return
    # Register into a throwaway catalog first, so a colliding or partially-failing `register` can
    # never mutate the real catalog — we merge only the clean, non-colliding result.
    scratch = TaskCatalog()
    try:
        register(scratch)
    except Exception as exc:  # noqa: BLE001
        log.warning("task_plugin_register_failed", entry_point=ep.name, error=str(exc))
        return
    for name in sorted(n for n in scratch.types() if catalog.has(n)):
        log.warning(
            "task_plugin_type_collision", entry_point=ep.name, type=name, policy="incumbent_wins"
        )
    for name in catalog.merge(scratch, on_collision="skip", source=ep.name):
        log.info("task_plugin_registered", entry_point=ep.name, type=name)
