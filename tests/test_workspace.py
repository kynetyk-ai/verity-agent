"""Workspace contract + layout tests (spec §3.4–§3.5) — ROADMAP Phase 1.2."""

from __future__ import annotations

from pathlib import Path

import pytest

from verity.control_plane.workspace import (
    WORKSPACE_CONTRACT,
    Access,
    DefaultLayout,
    WorkspaceError,
)


def test_contract_has_the_six_invariant_roles() -> None:
    names = WORKSPACE_CONTRACT.role_names()
    assert names == ("data", "context", "tools", "spec", "scratch", "outbox")
    assert {r.name for r in WORKSPACE_CONTRACT.readable_roles()} == {
        "data",
        "context",
        "tools",
        "spec",
    }
    assert {r.name for r in WORKSPACE_CONTRACT.writable_roles()} == {"scratch", "outbox"}
    assert WORKSPACE_CONTRACT.role("outbox").access is Access.WRITABLE_EPHEMERAL


def test_unknown_role_raises() -> None:
    with pytest.raises(WorkspaceError):
        WORKSPACE_CONTRACT.role("nope")


def test_provision_creates_all_roles_and_places_read_only_contents(tmp_path: Path) -> None:
    layout = DefaultLayout()
    ws = layout.provision(
        tmp_path,
        contents={"data": {"train.csv": b"a,b\n1,2\n"}, "spec": {"schema.json": b"{}"}},
    )
    for role in ("data", "context", "tools", "spec", "scratch", "outbox"):
        assert ws.path_for(role).is_dir()
    assert (ws.path_for("data") / "train.csv").read_bytes() == b"a,b\n1,2\n"
    assert (ws.path_for("spec") / "schema.json").read_bytes() == b"{}"
    # writable-ephemeral roles start empty
    assert list(ws.outbox().iterdir()) == []


def test_provision_refuses_to_prepopulate_writable_roles(tmp_path: Path) -> None:
    layout = DefaultLayout()
    with pytest.raises(WorkspaceError):
        layout.provision(tmp_path, contents={"outbox": {"x": b"y"}})


def test_provision_rejects_unknown_role(tmp_path: Path) -> None:
    layout = DefaultLayout()
    with pytest.raises(WorkspaceError):
        layout.provision(tmp_path, contents={"ghost": {"x": b"y"}})


def test_harvest_reads_the_outbox(tmp_path: Path) -> None:
    layout = DefaultLayout()
    ws = layout.provision(tmp_path, contents={})
    (ws.outbox() / "submission.py").write_bytes(b"print('hi')")
    (ws.outbox() / "features.json").write_bytes(b"[]")
    harvested = layout.harvest(ws)
    assert harvested == {"features.json": b"[]", "submission.py": b"print('hi')"}
