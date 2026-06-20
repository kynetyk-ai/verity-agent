"""The control plane is generic and task-agnostic (ROADMAP 7.4, decoupled). Offline.

Two proofs that the control plane is not shaped around any task:

1. **Mechanical (the convincer):** an AST import-guard asserting `verity.control_plane` imports
   *nothing* from `verity.domains`. If anyone couples the control plane to FE (or any task), the
   build fails here. This is what makes "the CP stays generic" enforceable rather than a promise.
2. **By demonstration:** the *same* generic `ControlPlane(store)` construction accepts two unrelated
   tasks — the trivial `code` task and the FE-Kaggle task — each applied through the CP's API, with
   no task knowledge baked into the control plane.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib

import verity.composition
import verity.control_plane
from tests._fe_offline import loopback_fe_kaggle_factory, role_files_from_raw
from verity.composition import configure_code_task
from verity.composition.fe_kaggle import configure_fe_kaggle_task
from verity.control_plane.api import ControlPlane
from verity.control_plane.store import SqliteStore
from verity.provisioning import FakeBackend
from verity.verifier.kaggle import FakeKaggleScorer

_RAW = b"id,a,class\n0,0,X\n1,2,X\n2,4,X\n3,6,Y\n4,8,Y\n5,10,Y\n"
_REAL_TEST = b"id\n100\n101\n"


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


def test_cp_image_imports_no_fe_gates_or_kaggle() -> None:
    """§9.1 litmus: importing the daemon's whole composition surface must NOT pull in the FE gate
    module or the third-party ``kaggle`` lib — those live in the verifier image now, so a new
    verifier/task type needs no control-plane rebuild. Run in a *subprocess* for a clean import
    graph (this test process has already imported the gates via the offline loopback helper)."""
    import subprocess
    import sys

    probe = (
        "import sys;"
        "import verity.service.daemon;"  # the `verity serve` entrypoint — the whole CP image graph
        "import verity.composition.catalog;"
        "import verity.composition.fe_kaggle;"
        "gates=[m for m in sys.modules if 'domains.feature_engineering_kaggle' in m];"
        "kg=[m for m in sys.modules if m=='kaggle' or m.startswith('kaggle.')];"
        "assert not gates, ('FE gate module in CP image: '+repr(gates));"
        "assert not kg, ('third-party kaggle in CP image: '+repr(kg));"
        "print('ok')"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


def test_cp_daemon_imports_without_the_kaggle_extra() -> None:
    """The operational proof that the control-plane image can drop ``--extra kaggle`` (§9.1): block
    the third-party ``kaggle`` package, then import the whole daemon graph. It must succeed — the CP
    pulls ``verity.verifier`` (for the code task's runner), whose kaggle client lazy-imports the lib
    only when it actually submits. Subprocess for a clean graph + a controlled import block."""
    import subprocess
    import sys

    probe = (
        "import builtins;"
        "_real=builtins.__import__\n"
        "def _blocked(name,*a,**k):\n"
        "    if name=='kaggle' or name.startswith('kaggle.'):\n"
        "        raise ImportError('simulated CP image: no kaggle extra')\n"
        "    return _real(name,*a,**k)\n"
        "builtins.__import__=_blocked\n"
        "import verity.service.daemon;"
        "import verity.composition.catalog;"
        "import verity.verifier;"
        "print('ok')"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


def test_cp_image_carries_no_data_prep() -> None:
    """ADR 0005 / §9.2 litmus: the control-plane composition surface holds **no** dataset split.

    Data prep is the user's responsibility (``tools/prepare_fe_data.py``); the CP routes opaque,
    role-keyed blobs. So (a) the old in-CP split module is gone, and (b) no module under
    ``verity.composition`` imports a split helper (``subsample``/``stratified_split``) or the
    user-side prep package. This is what makes "the CP interprets no dataset semantics" enforceable.
    """
    import importlib.util

    assert importlib.util.find_spec("verity.composition.dataset") is None, (
        "verity.composition.dataset must not exist — the split moved user-side (ADR 0005)"
    )
    leaks: list[str] = []
    pkg_dir = pathlib.Path(verity.composition.__file__).parent  # type: ignore[arg-type]
    banned_names = {"subsample", "stratified_split"}
    banned_modules = {"tools.harness.dataset", "verity.composition.dataset"}
    for path in pkg_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in banned_modules:
                leaks.append(f"{path.name}: from {node.module}")
            if isinstance(node, ast.ImportFrom) and any(
                a.name in banned_names for a in node.names
            ):
                leaks.append(f"{path.name}: imports {[a.name for a in node.names]}")
    assert not leaks, f"the control plane must carry no data prep; found: {leaks}"


def test_one_generic_control_plane_runs_two_different_tasks() -> None:
    # The identical construction — a bare ControlPlane(store) — accepts either task via its API.
    code_cp = ControlPlane(SqliteStore())
    assert asyncio.run(configure_code_task(code_cp, backend=FakeBackend())) == "code"

    kaggle_cp = ControlPlane(SqliteStore())
    backend = FakeBackend()
    files = role_files_from_raw(_RAW, _REAL_TEST, per_class=100, reserved_fraction=0.5)
    task_id = asyncio.run(
        configure_fe_kaggle_task(
            kaggle_cp, backend=backend,
            make_verifier=loopback_fe_kaggle_factory(backend, FakeKaggleScorer()),
            agent_files=files["agent"], verifier_files=files["verifier"],
        )
    )
    assert task_id == "fe-kaggle"
