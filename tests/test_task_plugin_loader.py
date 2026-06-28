"""Unit tests for the task-type plugin loader (ROADMAP 9.3, ADR 0006).

Discovery is injectable, so these pass **fake** entry points — no pip install. They pin the contract
(`register(catalog)`), the incumbent-wins collision policy, and degrade-don't-crash on every failure
mode (bad import, non-callable, register raising), plus scratch-catalog isolation.
"""

from __future__ import annotations

import asyncio

import pytest
import structlog

from verity.composition import loader as loader_module
from verity.composition.catalog import TaskCatalog, UnknownTaskType
from verity.composition.description import OperationDescription, TaskTypeDescription
from verity.composition.loader import load_task_plugins


@pytest.fixture(autouse=True)
def _hermetic_logging():
    """Isolate from any prior test that ran ``configure_logging`` (e.g. the daemon's JSON setup).

    ``capture_logs`` only intercepts a *freshly bound* logger, but ``configure_logging`` caches
    loggers (``cache_logger_on_first_use``) — so a module-level ``log`` bound under an earlier
    test's config escapes capture. Reset structlog to its (cache-free) defaults and rebind the
    module logger, so each test's ``capture_logs`` reliably sees the loader's events.
    """
    structlog.reset_defaults()
    loader_module.log = structlog.get_logger("verity.composition.loader")
    yield
    structlog.reset_defaults()


class FakeEntryPoint:
    """A stand-in for `EntryPoint`: the loader uses `.name` / `.value` / `.load()`."""

    def __init__(self, name, loader, *, value="fake_pkg:register", dist_name=""):
        self.name = name
        self.value = value
        self._loader = loader
        self.dist = type("Dist", (), {"name": dist_name})()

    def load(self):
        return self._loader()


def _source(*eps):
    return lambda: list(eps)


async def _build(cp, *, backend, request):  # a trivial async builder matching CatalogBuilder
    return "built"


def _describe():
    return TaskTypeDescription(
        type_name="plugin-task",
        artifact_types=("Echo",),
        gated_types=(),
        operations=(OperationDescription("echo", (), "Echo"),),
        domain_instructions="x",
        verifier_approach="y",
    )


def test_discovers_and_registers_a_plugin_type() -> None:
    def register(catalog):
        catalog.register("plugin-task", _build, describe=_describe)

    cat = load_task_plugins(
        TaskCatalog(), entry_points=_source(FakeEntryPoint("pt", lambda: register))
    )
    assert "plugin-task" in cat.types()
    assert "plugin-task" in {d.type_name for d in cat.describe_all()}
    assert asyncio.run(cat.build("plugin-task", cp=None, backend=None, request=None)) == "built"


def test_plugin_without_describe_registers_builder_only() -> None:
    def register(catalog):
        catalog.register("plugin-task", _build)  # no describe

    cat = load_task_plugins(
        TaskCatalog(), entry_points=_source(FakeEntryPoint("pt", lambda: register))
    )
    assert "plugin-task" in cat.types()
    assert "plugin-task" not in {d.type_name for d in cat.describe_all()}
    with pytest.raises(UnknownTaskType):
        cat.describe("plugin-task")


def test_load_failure_is_skipped_and_logged() -> None:
    def boom():
        raise ImportError("no module")

    with structlog.testing.capture_logs() as logs:
        cat = load_task_plugins(TaskCatalog(), entry_points=_source(FakeEntryPoint("bad", boom)))
    assert cat.types() == []
    assert any(e["event"] == "task_plugin_load_failed" for e in logs)


def test_non_callable_entry_point_is_skipped() -> None:
    with structlog.testing.capture_logs() as logs:
        cat = load_task_plugins(
            TaskCatalog(), entry_points=_source(FakeEntryPoint("n", lambda: 42))
        )
    assert cat.types() == []
    assert any(e["event"] == "task_plugin_not_callable" for e in logs)


def test_register_raising_leaves_real_catalog_untouched() -> None:
    def register(catalog):
        catalog.register("half", _build)  # mutates the SCRATCH catalog
        raise RuntimeError("boom mid-register")

    with structlog.testing.capture_logs() as logs:
        cat = load_task_plugins(
            TaskCatalog(), entry_points=_source(FakeEntryPoint("r", lambda: register))
        )
    assert cat.types() == []  # scratch isolation: nothing leaked into the real catalog
    assert any(e["event"] == "task_plugin_register_failed" for e in logs)


def test_collision_with_incumbent_keeps_incumbent() -> None:
    base = TaskCatalog()
    base.register("code", _build)  # the incumbent (e.g. a built-in)

    async def _other(cp, *, backend, request):
        return "shadow"

    def register(catalog):
        catalog.register("code", _other)

    with structlog.testing.capture_logs() as logs:
        cat = load_task_plugins(
            base, entry_points=_source(FakeEntryPoint("shadow", lambda: register))
        )
    assert (
        asyncio.run(cat.build("code", cp=None, backend=None, request=None)) == "built"
    )  # incumbent wins
    assert any(e["event"] == "task_plugin_type_collision" and e["type"] == "code" for e in logs)


def test_two_plugins_same_name_first_wins() -> None:
    def reg_a(catalog):
        catalog.register("dup", _build)

    async def _b(cp, *, backend, request):
        return "second"

    def reg_b(catalog):
        catalog.register("dup", _b)

    cat = load_task_plugins(
        TaskCatalog(),
        entry_points=_source(
            FakeEntryPoint("a", lambda: reg_a), FakeEntryPoint("b", lambda: reg_b)
        ),
    )
    assert (
        asyncio.run(cat.build("dup", cp=None, backend=None, request=None)) == "built"
    )  # first wins


def test_default_entry_points_smoke() -> None:
    from verity.composition.loader import _default_entry_points

    assert list(_default_entry_points()) is not None  # returns an iterable against the real env


def test_fixture_plugin_appears_without_core_import() -> None:
    """End-to-end (ROADMAP 9.3 litmus): a task type the core never imports is reachable purely via
    entry-point discovery — no edit to default_catalog(). The fixture lives under tests/, so the
    composition genericity guard confirms `verity` references it nowhere."""
    import importlib

    from tests._plugin_fixture import FIXTURE_TASK_ID
    from verity.composition import default_catalog

    fixture_ep = FakeEntryPoint(
        "fixture-echo", lambda: importlib.import_module("tests._plugin_fixture").register
    )
    cat = load_task_plugins(default_catalog(), entry_points=_source(fixture_ep))
    assert FIXTURE_TASK_ID in cat.types()
    assert FIXTURE_TASK_ID in {d.type_name for d in cat.describe_all()}
    assert (
        asyncio.run(cat.build(FIXTURE_TASK_ID, cp=None, backend=None, request=None))
        == FIXTURE_TASK_ID
    )
    # the built-ins still present alongside the discovered fixture
    assert {"code", "fe-kaggle"} <= set(cat.types())
