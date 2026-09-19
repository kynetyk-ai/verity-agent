"""The task catalog (ROADMAP 8.1, ADR 0004 (b)/(i)): a registry of named task-type builders.

A :class:`TaskCatalog` maps a **task-type name** (`"fe-kaggle"`, `"code"`) to an async **builder**
with a uniform signature — ``async def build(cp, *, backend, request) -> str`` — that applies it to
a generic, task-agnostic `ControlPlane` from a declarative :class:`TaskRequest`. This is the seam
the long-lived control service multiplexes over: select a type by name, parameterize it with a
request, no image rebuild.

It lives in :mod:`verity.composition` because its builders are the one layer allowed to know both a
domain and the control plane. ``register`` is also the runtime-registration seam (ADR 0004
(i)): a future plugin loader can discover builders from an entry-point group and register them at
boot with no daemon change — this ADR *places* the seam without building the loader.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from verity.composition.description import TaskTypeDescription
from verity.composition.task_request import TaskRequest
from verity.control_plane.api import ControlPlane
from verity.provisioning.backend import WorkerBackend

__all__ = ["CatalogBuilder", "Describer", "TaskCatalog", "UnknownTaskType", "default_catalog"]

# The built-in task types the core ``verity`` dist advertises as ``verity.task_types`` entry points
# (see pyproject.toml). Literals here, so this module imports no domain at load time — the builders
# are pulled lazily, only when the loader resolves their entry points (ADR 0006).
_BUILTIN_TASK_TYPES = ("code", "fe-kaggle")

#: A catalog builder: applies a task to a generic control plane and returns its in-CP task id.
CatalogBuilder = Callable[..., Awaitable[str]]
#: A describer: returns a task type's published contract (no backend needed — ADR 0004 (c)).
Describer = Callable[[], TaskTypeDescription]


class UnknownTaskType(KeyError):
    """Raised when a request names a task type the catalog does not know."""


class TaskCatalog:
    """A registry: task-type name -> async builder; the service dispatches a request through it."""

    def __init__(self) -> None:
        self._builders: dict[str, CatalogBuilder] = {}
        self._describers: dict[str, Describer] = {}

    def register(
        self, type_name: str, builder: CatalogBuilder, *, describe: Describer | None = None
    ) -> None:
        """Register a builder (+ optional describer) under ``type_name`` (the ADR 0004 (i) seam)."""
        self._builders[type_name] = builder
        if describe is not None:
            self._describers[type_name] = describe

    def merge(
        self, other: TaskCatalog, *, on_collision: str = "skip", source: str = ""
    ) -> list[str]:
        """Fold ``other``'s registrations into self; returns the names actually added.

        ``on_collision="skip"`` keeps self's existing entry (the **incumbent wins**) and skips the
        incoming one; ``"error"`` raises. The plugin loader uses this to merge discovered builders
        into the catalog without reaching into private state.
        """
        added: list[str] = []
        for name in other.types():
            if self.has(name):
                if on_collision == "skip":
                    continue
                raise ValueError(f"task type {name!r} from {source!r} collides with an existing")
            self._builders[name] = other._builders[name]
            if name in other._describers:
                self._describers[name] = other._describers[name]
            added.append(name)
        return added

    def types(self) -> list[str]:
        """The registered task-type names (what ``verity catalog`` renders)."""
        return sorted(self._builders)

    def has(self, type_name: str) -> bool:
        return type_name in self._builders

    def describe(self, type_name: str) -> TaskTypeDescription:
        """The published contract for ``type_name`` (ADR 0004 (c)); raises if it is undescribed."""
        try:
            return self._describers[type_name]()
        except KeyError as exc:
            raise UnknownTaskType(
                f"no description for task type {type_name!r}; described: {sorted(self._describers)}"
            ) from exc

    def describe_all(self) -> list[TaskTypeDescription]:
        """Every described task type's published contract, ordered by name."""
        return [self._describers[name]() for name in sorted(self._describers)]

    async def build(
        self,
        type_name: str,
        cp: ControlPlane,
        *,
        backend: WorkerBackend,
        request: TaskRequest,
    ) -> str:
        """Dispatch to the named builder, applying the task to ``cp``. Returns its in-CP task id."""
        try:
            builder = self._builders[type_name]
        except KeyError as exc:
            raise UnknownTaskType(
                f"unknown task type {type_name!r}; known types: {self.types()}"
            ) from exc
        return await builder(cp, backend=backend, request=request)


def default_catalog() -> TaskCatalog:
    """The catalog the daemon serves: every task type discovered from the ``verity.task_types``
    entry-point group (ADR 0004 (i) / 0006), including the built-in ``code`` + ``fe-kaggle`` types,
    which the core distribution advertises and which load first. A broken third-party plugin is
    logged and skipped; the daemon still boots.

    Discovery reads *installed* dist metadata (not live ``pyproject.toml``), so the built-ins must
    be present. If they are missing the install lacks its entry-point metadata — we fail **loudly**
    (run ``uv sync`` to regenerate it) rather than silently serve an empty catalog.
    """
    from verity.composition.loader import load_task_plugins  # local: keeps this module domain-free

    catalog = load_task_plugins(TaskCatalog())
    missing = sorted(set(_BUILTIN_TASK_TYPES) - set(catalog.types()))
    if missing:
        raise RuntimeError(
            f"built-in task types {missing} were not discovered from the 'verity.task_types' "
            "entry-point group — the install is missing its entry-point metadata; run `uv sync` "
            "(or reinstall) to regenerate it"
        )
    return catalog
