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

import verity.control_plane
from tests._fe_offline import loopback_fe_kaggle_factory
from verity.composition import configure_code_task
from verity.composition.dataset import stratified_split, subsample
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


def test_one_generic_control_plane_runs_two_different_tasks() -> None:
    # The identical construction — a bare ControlPlane(store) — accepts either task via its API.
    code_cp = ControlPlane(SqliteStore())
    assert asyncio.run(configure_code_task(code_cp, backend=FakeBackend())) == "code"

    kaggle_cp = ControlPlane(SqliteStore())
    backend = FakeBackend()
    sub = subsample(_RAW, per_class=100, target="class")
    split = stratified_split(sub, target="class", id_column="id", reserved_fraction=0.5)
    task_id = asyncio.run(
        configure_fe_kaggle_task(
            kaggle_cp, backend=backend,
            make_verifier=loopback_fe_kaggle_factory(backend, FakeKaggleScorer()),
            split=split, full_train_csv=sub, real_test_csv=_REAL_TEST,
        )
    )
    assert task_id == "fe-kaggle"
