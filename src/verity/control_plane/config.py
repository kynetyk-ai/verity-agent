"""Per-task configuration and the composed system prompt (spec §3.4).

The configuration surface for a task (configured, not coded): task instructions, the
proposal-shape validator, data sources, contextual info, and the §8 registrations — plus the
provider keys that select the sandbox and verifier adapters (:mod:`verity.contracts`). The invariant
**workspace contract** itself lives in :mod:`.workspace`; this module composes the agent's
system prompt against it.

The **composed system prompt** (§3.4) is assembled *mechanically* — pure string assembly, no
judgment, consistent with the control plane's unintelligence — from three layers:

1. **Kernel orientation** (invariant across all domains) — work independently, the workspace
   layout, where to write proposals/objects, self-verify-then-submit, and follow any correction
   instructions. Rendered from the :class:`~verity.control_plane.workspace.WorkspaceContract`
   so it cannot drift from the actual roles.
2. **Domain instructions** (per-domain; part of the domain definition, §8).
3. **Task instructions** (per-task; the user writes intent, not paths).

The orientation + layout layers are stable, so they belong in the prompt-cache prefix (§9).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

from verity.contracts import Artifact, JSONValue
from verity.control_plane.commit import ShapeValidator
from verity.control_plane.registries import (
    GatedTypeRegistry,
    ObjectProvisioningPolicy,
    RetrievalPolicy,
    SchemaRegistry,
)
from verity.control_plane.workspace import WORKSPACE_CONTRACT, WorkspaceContract

__all__ = [
    "TaskConfig",
    "HarvestedChild",
    "Harvester",
    "ObjectNamer",
    "render_workspace_layout",
    "compose_system_prompt",
    "orientation_digest",
    "KERNEL_ORIENTATION_TEMPLATE",
]


@dataclass(frozen=True, slots=True)
class HarvestedChild:
    """A child artifact the harness mints from a parent's payload (spec §4.1, §12 ``Feature``).

    The domain's :data:`Harvester` returns these; the control plane mints the typed artifact + the
    ``op_name`` operation (parent → child), carrying the parent's object sidecar so the child's gate
    can see it. Ids and lineage are minted on the trusted side — the domain only declares content.
    """

    artifact_type: str
    op_name: str
    payload: JSONValue


# A domain hook: derive the child artifacts to harvest from a (just-accepted) parent artifact (§12).
Harvester = Callable[[Artifact], list[HarvestedChild]]

# A domain hook: the object names a proposal of this artifact declares (the only files harvested
# from its outbox; everything else is dropped). Empty set = the proposal carries no objects.
# Required on every task — context discipline: an undeclared output never enters the store (§3.4).
ObjectNamer = Callable[[Artifact], frozenset[str]]


@dataclass(frozen=True, slots=True)
class TaskConfig:
    """A task's full configuration surface (spec §3.4).

    ``sandbox_key`` / ``verifier_key`` name the providers to resolve from the port registries
    (:mod:`verity.contracts`), so swapping the harness or verifier is a config change.
    ``shape_validator``
    realizes the proposal-shape spec the commit path checks at §7.0. ``data_sources`` are host
    paths the control plane reads into the read-only ``data/`` role (in-process) or the driver
    bind-mounts (container). ``object_provisioning`` selects durable objects to materialize into the
    workspace each cycle as references the agent builds on (defaults to ``NONE``).
    """

    task_id: str
    instructions: str
    domain_instructions: str
    schema: SchemaRegistry
    gated_types: GatedTypeRegistry
    retrieval: RetrievalPolicy
    shape_validator: ShapeValidator
    object_namer: ObjectNamer
    sandbox_key: str
    verifier_key: str
    # The tenant this task runs under (Phase 7 multi-tenancy seam). A single ``"default"`` tenant
    # today; the API is keyed by it from the outset so tenant scoping slots in without a reshape.
    tenant_id: str = "default"
    data_sources: tuple[str, ...] = ()
    context_files: tuple[str, ...] = ()
    object_provisioning: ObjectProvisioningPolicy = field(default_factory=ObjectProvisioningPolicy)
    harvester: Harvester | None = None
    contract: WorkspaceContract = field(default=WORKSPACE_CONTRACT)

    def system_prompt(self) -> str:
        """The composed 3-layer system prompt for this task (§3.4)."""
        return compose_system_prompt(
            contract=self.contract,
            domain_instructions=self.domain_instructions,
            task_instructions=self.instructions,
        )


KERNEL_ORIENTATION_TEMPLATE = """\
Work independently and diligently to complete the task provided:
- Plan the steps that must be taken to achieve the goal
- Execute the steps to complete a best-work proposal
- **Do not** ask clarifying questions, request user input, or wait for permission. Execute
  independently.

Your workspace has a fixed layout (read-only inputs; writable-ephemeral working areas):
{layout}

- The required output schema for the proposal lives under spec/.
- Prepare to submit a proposal by moving any object attachments (a script, a data file) to outbox/;
  do not rely on any other path.
- Self-verify the shape of the proposal and if possible, test the proposal prior to submission.
- Submit the proposal using the tool provided and within the time allotted.
- If you receive instructions to correct a previously submitted proposal, follow them
  precisely.
"""


def render_workspace_layout(contract: WorkspaceContract) -> str:
    """Render the workspace roles as the prompt's layout block (kept in sync with §3.4)."""
    width = max(len(role.name) for role in contract.roles)
    return "\n".join(
        f"  {role.name + '/':<{width + 1}}  ({role.access})  {role.purpose}"
        for role in contract.roles
    )


def _render_orientation(contract: WorkspaceContract) -> str:
    """The invariant kernel-orientation block (layer 1), rendered from the contract."""
    return KERNEL_ORIENTATION_TEMPLATE.format(layout=render_workspace_layout(contract))


def compose_system_prompt(
    *, contract: WorkspaceContract, domain_instructions: str, task_instructions: str
) -> str:
    """Mechanically assemble the 3-layer system prompt (§3.4). Deterministic string assembly."""
    return "\n\n".join(
        [
            "# Kernel orientation (invariant)\n" + _render_orientation(contract),
            "# Domain instructions\n" + domain_instructions.strip(),
            "# Task instructions\n" + task_instructions.strip(),
        ]
    )


def orientation_digest(contract: WorkspaceContract) -> str:
    """A content hash of the rendered kernel orientation — the #12 stamp's tamper-evidence.

    Covers both the orientation template and the contract's role skeleton, so any change to *how the
    agent is oriented* changes the digest even if the version number was not bumped (ROADMAP 5.1).
    """
    return hashlib.sha256(_render_orientation(contract).encode()).hexdigest()
