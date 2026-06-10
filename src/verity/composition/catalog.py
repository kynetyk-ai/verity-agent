"""The task catalog (ROADMAP 8.1, ADR 0004 (b)/(i)): a registry of named task-type builders.

A :class:`TaskCatalog` maps a **task-type name** (`"fe"`, `"code"`) to an async **builder** with a
uniform signature — ``async def build(cp, *, backend, request) -> str`` — that applies the task to a
generic, task-agnostic `ControlPlane` from a declarative :class:`TaskRequest`. This is the seam
the long-lived control service multiplexes over: select a type by name, parameterize it with a
request, no image rebuild.

It lives in :mod:`verity.composition` because its builders are the one layer allowed to know both a
domain and the control plane. ``register`` is also the runtime-registration seam (ADR 0004
(i)): a future plugin loader can discover builders from an entry-point group and register them at
boot with no daemon change — this ADR *places* the seam without building the loader.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from verity.composition.code import CODE_TASK_ID, build_code_task
from verity.composition.fe import FE_TASK_ID, build_fe_task
from verity.composition.task_request import TaskRequest
from verity.control_plane.api import ControlPlane
from verity.provisioning.backend import WorkerBackend

__all__ = ["CatalogBuilder", "TaskCatalog", "UnknownTaskType", "default_catalog"]

#: A catalog builder: applies a task to a generic control plane and returns its in-CP task id.
CatalogBuilder = Callable[..., Awaitable[str]]


class UnknownTaskType(KeyError):
    """Raised when a request names a task type the catalog does not know."""


class TaskCatalog:
    """A registry: task-type name -> async builder; the service dispatches a request through it."""

    def __init__(self) -> None:
        self._builders: dict[str, CatalogBuilder] = {}

    def register(self, type_name: str, builder: CatalogBuilder) -> None:
        """Register a builder under ``type_name`` (the ADR 0004 (i) runtime-registration seam)."""
        self._builders[type_name] = builder

    def types(self) -> list[str]:
        """The registered task-type names (what ``verity catalog`` will render in 8.2)."""
        return sorted(self._builders)

    def has(self, type_name: str) -> bool:
        return type_name in self._builders

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
    """The built-in catalog: the FE and trivial-`code` task types installed in the image."""
    catalog = TaskCatalog()
    catalog.register(FE_TASK_ID, build_fe_task)
    catalog.register(CODE_TASK_ID, build_code_task)
    return catalog
