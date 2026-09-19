"""The genericity invariant as a property (ROADMAP 7.4 / ADR 0003). Offline.

The whole design rests on the control plane being agnostic to how workers run, and on
`provisioning/` being neutral infra. This asserts the import boundary directly: `provisioning/` may
not import any
service or domain layer, and `control_plane/` may not import `provisioning/` (provisioning lives
*behind* the ports). A regression here is an architecture regression, caught mechanically.
"""

from __future__ import annotations

import ast
import pathlib

import verity.control_plane
import verity.provisioning


def _imported_modules(package: object) -> set[str]:
    """Every module name imported by any .py file under ``package``."""
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


def _references(modules: set[str], prefix: str) -> list[str]:
    return sorted(m for m in modules if m == prefix or m.startswith(prefix + "."))


def test_provisioning_imports_no_service_or_domain_layer() -> None:
    imports = _imported_modules(verity.provisioning)
    forbidden = ("verity.sandbox", "verity.verifier", "verity.control_plane", "verity.domains")
    leaks = [m for prefix in forbidden for m in _references(imports, prefix)]
    assert not leaks, f"provisioning/ must stay neutral; it imports: {leaks}"


def test_control_plane_does_not_import_provisioning() -> None:
    imports = _imported_modules(verity.control_plane)
    leaks = _references(imports, "verity.provisioning")
    assert not leaks, f"the control plane must stay provisioning-agnostic; it imports: {leaks}"
