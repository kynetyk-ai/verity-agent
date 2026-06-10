"""The control plane is generic and task-agnostic (ROADMAP 7.4, decoupled). Offline.

Two proofs that the control plane is not shaped around any task:

1. **Mechanical (the convincer):** an AST import-guard asserting `verity.control_plane` imports
   *nothing* from `verity.domains`. If anyone couples the control plane to FE (or any task), the
   build fails here. This is what makes "the CP stays generic" enforceable rather than a promise.
2. **By demonstration:** the *same* generic `ControlPlane(store)` construction accepts two unrelated
   tasks — FE and the trivial `code` task — each applied through the CP's API (`configure_*_task`),
   with no task knowledge baked into the control plane.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
from dataclasses import dataclass, field

import verity.control_plane
from verity.composition import configure_code_task, configure_fe_task
from verity.control_plane.api import ControlPlane
from verity.control_plane.store import SqliteStore
from verity.provisioning import FakeBackend


def _imported_modules(package: object) -> set[str]:
    pkg_dir = pathlib.Path(package.__file__).parent  # type: ignore[attr-defined]
    found: set[str] = set()
    for path in pkg_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module)
    return found


def test_control_plane_imports_no_domain() -> None:
    imports = _imported_modules(verity.control_plane)
    leaks = sorted(
        m for m in imports if m == "verity.domains" or m.startswith("verity.domains.")
    )
    assert not leaks, f"the control plane must stay task-agnostic; it imports: {leaks}"


@dataclass
class _Split:
    agent_train_csv: bytes = b"id,a,class\n1,2,X\n3,4,Y\n"
    reserved_test_csv: bytes = b"id,a\n5,6\n"
    reserved_labels: dict[str, str] = field(default_factory=lambda: {"5": "X"})


def test_one_generic_control_plane_runs_two_different_tasks() -> None:
    # The identical construction — a bare ControlPlane(store) — accepts either task via its API.
    fe_cp = ControlPlane(SqliteStore())
    assert asyncio.run(configure_fe_task(fe_cp, backend=FakeBackend(), split=_Split())) == "fe"

    code_cp = ControlPlane(SqliteStore())
    assert asyncio.run(configure_code_task(code_cp, backend=FakeBackend())) == "code"
