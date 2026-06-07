"""The workspace contract and its pluggable layout (spec §3.4–§3.5).

Two things are kept deliberately separate:

* **The workspace *contract*** is **invariant** and **versioned** like the schema (§3.4): the
  fixed set of logical roles — read-only ``data`` / ``context`` / ``tools`` / ``spec`` and
  writable-ephemeral ``scratch`` / ``outbox`` — their access modes, and the rule that the outbox
  is the deterministic harvest source. Its *structure* never varies; only the per-task *contents*
  do. Two consequences are load-bearing: the read-only / writable split makes gold-data isolation
  physical (§3.5), and the outbox makes harvest "read the outbox," not "find the path the agent
  named" (§3.4).
* **The workspace *layout*** is **pluggable**: how those logical roles materialize on disk —
  concrete directory names, plus any harness-specific affordances — for a given agent framework.
  The MVP ships exactly one default layout using the plain §3.4 role names; a second harness's
  layout is an added :class:`WorkspaceLayout` implementation, not a rewrite. The control plane
  provisions *and harvests* through the layout, so "where is the outbox" is the layout's concern
  while harvest-before-teardown stays a control-plane ordering invariant (§3.4).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from verity.logging import get_logger

__all__ = [
    "Access",
    "WorkspaceRole",
    "WorkspaceContract",
    "WORKSPACE_CONTRACT",
    "WORKSPACE_CONTRACT_VERSION",
    "ProvisionedWorkspace",
    "WorkspaceLayout",
    "DefaultLayout",
    "WorkspaceError",
]

log = get_logger("verity.control_plane.workspace")


class WorkspaceError(RuntimeError):
    """Raised on a workspace-contract violation (unknown role, pre-populating a writable role)."""


class Access(StrEnum):
    """A role's access mode. The read-only/writable split is the gold-data boundary (§3.5)."""

    READ_ONLY = "read-only"
    WRITABLE_EPHEMERAL = "writable-ephemeral"


@dataclass(frozen=True, slots=True)
class WorkspaceRole:
    """One logical role in the invariant workspace skeleton (§3.4)."""

    name: str
    access: Access
    purpose: str


# The invariant contract — a **versioned kernel artifact** (§3.4). Bump the version on any change
# to "how the agent is oriented," so it is deliberate and recorded rather than ambient.
WORKSPACE_CONTRACT_VERSION = 1

_ROLES: tuple[WorkspaceRole, ...] = (
    WorkspaceRole("data", Access.READ_ONLY, "the task's data sources (read-only, §3.4)"),
    WorkspaceRole("context", Access.READ_ONLY, "skills, SOPs, gold-standard examples"),
    WorkspaceRole("tools", Access.READ_ONLY, "the task's registered tools (§8.2)"),
    WorkspaceRole("spec", Access.READ_ONLY, "the proposal-shape spec / output schema (§3.4)"),
    WorkspaceRole("scratch", Access.WRITABLE_EPHEMERAL, "the agent's working space"),
    WorkspaceRole(
        "outbox", Access.WRITABLE_EPHEMERAL, "proposal + object attachments; harvest source"
    ),
)

OUTBOX_ROLE = "outbox"


@dataclass(frozen=True, slots=True)
class WorkspaceContract:
    """The invariant role skeleton the control plane provisions into every sandbox (§3.4)."""

    version: int
    roles: tuple[WorkspaceRole, ...]

    def role(self, name: str) -> WorkspaceRole:
        for role in self.roles:
            if role.name == name:
                return role
        raise WorkspaceError(f"unknown workspace role: {name!r}")

    def role_names(self) -> tuple[str, ...]:
        return tuple(role.name for role in self.roles)

    def readable_roles(self) -> tuple[WorkspaceRole, ...]:
        return tuple(r for r in self.roles if r.access is Access.READ_ONLY)

    def writable_roles(self) -> tuple[WorkspaceRole, ...]:
        return tuple(r for r in self.roles if r.access is Access.WRITABLE_EPHEMERAL)


WORKSPACE_CONTRACT = WorkspaceContract(version=WORKSPACE_CONTRACT_VERSION, roles=_ROLES)


@dataclass(frozen=True, slots=True)
class ProvisionedWorkspace:
    """A workspace materialized on disk: the role → absolute-path map plus the harvest source."""

    root: Path
    paths: Mapping[str, Path]

    def path_for(self, role: str) -> Path:
        if role not in self.paths:
            raise WorkspaceError(f"role {role!r} is not provisioned")
        return self.paths[role]

    def outbox(self) -> Path:
        return self.path_for(OUTBOX_ROLE)


class WorkspaceLayout(Protocol):
    """Maps the invariant contract onto concrete on-disk paths for a given harness (§3.4).

    An implementation decides directory names and any harness-specific affordances, then
    provisions the skeleton and exposes the outbox as the harvest source.
    """

    def path_for(self, role: str) -> str: ...
    def provision(
        self, root: Path, contents: Mapping[str, Mapping[str, bytes]]
    ) -> ProvisionedWorkspace: ...
    def harvest(self, workspace: ProvisionedWorkspace) -> dict[str, bytes]: ...


@dataclass(frozen=True, slots=True)
class DefaultLayout:
    """The default layout: each logical role maps to a directory of the same name (§3.4).

    The plain spec-role layout — the safe default before a target harness is chosen (the harness
    pick and its tuned layout are a Phase 3 adapter). ``contents`` places per-task files into the
    **read-only** roles; the writable-ephemeral roles are created empty, ready to be discarded on
    regeneration.
    """

    contract: WorkspaceContract = WORKSPACE_CONTRACT

    def path_for(self, role: str) -> str:
        # validates the role exists in the contract
        return self.contract.role(role).name

    def provision(
        self, root: Path, contents: Mapping[str, Mapping[str, bytes]]
    ) -> ProvisionedWorkspace:
        writable = {r.name for r in self.contract.writable_roles()}
        for role_name in contents:
            if role_name not in self.contract.role_names():
                raise WorkspaceError(f"cannot place contents into unknown role {role_name!r}")
            if role_name in writable:
                raise WorkspaceError(
                    f"refusing to pre-populate writable-ephemeral role {role_name!r}: only "
                    f"read-only roles carry per-task contents (§3.4)"
                )
        paths: dict[str, Path] = {}
        for role in self.contract.roles:
            role_dir = root / self.path_for(role.name)
            role_dir.mkdir(parents=True, exist_ok=True)
            for filename, data in contents.get(role.name, {}).items():
                (role_dir / filename).write_bytes(data)
            paths[role.name] = role_dir
        log.info(
            "workspace_provisioned",
            root=str(root),
            contract_version=self.contract.version,
            roles=list(paths),
        )
        return ProvisionedWorkspace(root=root, paths=paths)

    def harvest(self, workspace: ProvisionedWorkspace) -> dict[str, bytes]:
        """Read every file in the outbox — the deterministic harvest source (§3.4, §3.5)."""
        outbox = workspace.outbox()
        harvested = {
            path.name: path.read_bytes()
            for path in sorted(outbox.iterdir())
            if path.is_file()
        }
        log.info("outbox_harvested", count=len(harvested), outbox=str(outbox))
        return harvested
