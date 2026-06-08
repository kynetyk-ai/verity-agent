"""Per-task configuration and the composed system prompt (spec §3.4).

The configuration surface for a task (configured, not coded): task instructions, the
proposal-shape validator, data sources, contextual info, and the §8 registrations — plus the
provider keys that select the sandbox and verifier adapters (:mod:`verity.contracts`). The invariant
**workspace contract** itself lives in :mod:`.workspace`; this module composes the agent's
system prompt against it.

The **composed system prompt** (§3.4) is assembled *mechanically* — pure string assembly, no
judgment, consistent with the control plane's unintelligence — from three layers:

1. **Kernel orientation** (invariant across all domains) — the loop, the propose-only contract,
   the workspace layout, where to write proposals/objects, and how shape-errors and refine
   feedback come back. Rendered from the :class:`~verity.control_plane.workspace.WorkspaceContract`
   so it cannot drift from the actual roles.
2. **Domain instructions** (per-domain; part of the domain definition, §8).
3. **Task instructions** (per-task; the user writes intent, not paths).

The orientation + layout layers are stable, so they belong in the prompt-cache prefix (§9).
"""

from __future__ import annotations

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
    "render_workspace_layout",
    "compose_system_prompt",
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
    sandbox_key: str
    verifier_key: str
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
You operate a single repeating loop: read -> propose -> gate -> commit.
You PROPOSE artifacts; you never write durable state, and you never contact the verifier.
A proposal is a new artifact plus the operation that produced it, with a short note on how and
why you produced it.

Your workspace has a fixed layout (read-only inputs; writable-ephemeral working areas):
{layout}

Write your proposal and any object attachments (a script, a data file) to outbox/. The harness
harvests the outbox; do not rely on any other path. The output schema lives under spec/.

Two corrections may come back:
- A shape-error means the proposal is malformed (wrong parts/types). Fix the formatting and
  resubmit; nothing is recorded.
- Refine feedback means the proposal is mostly sound but carries named defects. Produce a
  tracked revision that addresses exactly those defects."""


def render_workspace_layout(contract: WorkspaceContract) -> str:
    """Render the workspace roles as the prompt's layout block (kept in sync with §3.4)."""
    width = max(len(role.name) for role in contract.roles)
    return "\n".join(
        f"  {role.name + '/':<{width + 1}}  ({role.access})  {role.purpose}"
        for role in contract.roles
    )


def compose_system_prompt(
    *, contract: WorkspaceContract, domain_instructions: str, task_instructions: str
) -> str:
    """Mechanically assemble the 3-layer system prompt (§3.4). Deterministic string assembly."""
    orientation = KERNEL_ORIENTATION_TEMPLATE.format(
        layout=render_workspace_layout(contract)
    )
    return "\n\n".join(
        [
            "# Kernel orientation (invariant)\n" + orientation,
            "# Domain instructions\n" + domain_instructions.strip(),
            "# Task instructions\n" + task_instructions.strip(),
        ]
    )
