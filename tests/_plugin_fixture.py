"""A throwaway out-of-tree task-type plugin for the loader end-to-end test (ROADMAP 9.3).

Underscore-prefixed (like ``tests/_fe_offline.py``) so pytest never collects it, and — critically —
**nothing under ``verity`` imports it**. It exists only to prove a task type the core package does
not know about appears in the catalog purely via entry-point discovery, with no edit to
``default_catalog()``. It mirrors what a real third-party plugin package would ship: a ``register``.
"""

from __future__ import annotations

from verity.composition.catalog import TaskCatalog
from verity.composition.description import OperationDescription, TaskTypeDescription

FIXTURE_TASK_ID = "fixture-echo"


async def build_fixture_task(cp, *, backend, request):  # type: ignore[no-untyped-def]
    """A trivial builder: returns the task id so the test can prove dispatch reaches here."""
    return FIXTURE_TASK_ID


def describe_fixture_task() -> TaskTypeDescription:
    return TaskTypeDescription(
        type_name=FIXTURE_TASK_ID,
        artifact_types=("Echo",),
        gated_types=(),
        operations=(OperationDescription("echo", (), "Echo"),),
        domain_instructions="echo the input",
        verifier_approach="trivially accepts",
    )


def register(catalog: TaskCatalog) -> None:
    catalog.register(FIXTURE_TASK_ID, build_fixture_task, describe=describe_fixture_task)
