"""Stub agent / workspace — a control-plane integration double (ROADMAP Phase 1.5).

NON-PRODUCT. Stands in for the real sandbox (spec §3.5): reads the served context, emits
*scripted* proposals, writes object attachments to the outbox, and responds to shape-error and
refine feedback — all deterministically, with no LLM. Used to exercise the control plane's
propose → gate → commit cycle and the object harvest path.

It implements :class:`~verity.contracts.ports.SandboxPort` over a **real**
:class:`~verity.control_plane.workspace.DefaultLayout` workspace on disk: each scripted step's
object attachments are written to the ``outbox`` and carried out by reference in the envelope
(§3.5), and :meth:`regenerate` discards the writable workspace, so ephemerality is physical. It
holds **no** reference to the store or the verifier and exposes no write path to either — the
privileged-mutator boundary is structural (§3.3, §13.9).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path

from verity.contracts import ProposalEnvelope, ServedContext
from verity.control_plane.workspace import (
    DefaultLayout,
    ProvisionedWorkspace,
    WorkspaceLayout,
)
from verity.logging import get_logger

__all__ = ["StubAgent"]

log = get_logger("tools.harness.stub_agent")


@dataclass
class StubAgent:
    """A scripted, in-process :class:`SandboxPort` over a real workspace (spec §3.5). NON-PRODUCT.

    ``steps`` is the scripted sequence of proposals; the agent emits them in order. ``served``
    records every context it was handed (including the feedback channel), so a test can confirm
    the agent received a shape-error or refine feedback (§3.5).
    """

    root: Path
    steps: list[ProposalEnvelope]
    layout: WorkspaceLayout = field(default_factory=DefaultLayout)
    served: list[ServedContext] = field(default_factory=list)
    regenerations: int = 0
    _cursor: int = field(default=0, init=False)
    _workspace: ProvisionedWorkspace | None = field(default=None, init=False)

    async def provision(self) -> None:
        self._workspace = self.layout.provision(self.root, contents={})

    async def serve_context(self, context: ServedContext) -> None:
        self.served.append(context)

    async def collect_proposal(self) -> ProposalEnvelope:
        step = self.steps[self._cursor]
        self._cursor += 1
        workspace = self._require_workspace()
        # Write the step's object attachments to the outbox, then carry them out by reference
        # in the envelope (§3.5) — the control plane content-addresses them at intake.
        for name, data in step.objects.items():
            (workspace.outbox() / name).write_bytes(data)
        harvested = self.layout.harvest(workspace)
        log.info("stub_proposal", artifact=step.artifact.id, objects=sorted(harvested))
        return replace(step, objects=harvested)

    async def regenerate(self) -> None:
        # Discard the writable workspace — re-provisioning empties scratch/ and outbox/ (§3.5).
        if self.root.exists():
            shutil.rmtree(self.root)
        self._workspace = self.layout.provision(self.root, contents={})
        self.regenerations += 1

    async def teardown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    async def health(self) -> bool:
        return True

    @property
    def last_feedback(self) -> str:
        return self.served[-1].feedback if self.served else ""

    @property
    def workspace(self) -> ProvisionedWorkspace:
        return self._require_workspace()

    def _require_workspace(self) -> ProvisionedWorkspace:
        if self._workspace is None:
            raise RuntimeError("workspace not provisioned; call provision() first")
        return self._workspace
