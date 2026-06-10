"""The framework-neutral sandbox core (ROADMAP Phase 3).

:class:`AgentSandbox` is the real :class:`~verity.contracts.SandboxPort` that replaces the stub.
It owns the workspace lifecycle and the trusted harvest/mint half of a cycle; the framework-specific
half (running the agent loop) is delegated to an injected ``SandboxDriver`` (see :mod:`.driver`).

One cycle:

* ``serve_context`` stashes the assembled :class:`~verity.contracts.ServedContext`;
* ``collect_proposal`` builds the user message, runs the driver (which leaves a descriptor +
  attachments in the outbox), then **harvests**, **splits** the descriptor from the attachments, and
  **mints** the typed ``Artifact`` + ``Operation`` from an injected clock / id-source — the agent
  never mints ids or sees the store;
* ``regenerate`` discards the workspace and re-provisions it empty (the ephemerality event, §3.5).

The core is **domain-agnostic**: it binds whatever operations the schema declares and serves the
context the control plane assembled — nothing task-specific is hardcoded here.
"""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    Operation,
    OperationStatus,
    ProposalEnvelope,
    RunContext,
    ServedContext,
    SupportsRunContext,
)
from verity.control_plane.registries import SchemaRegistry
from verity.control_plane.workspace import (
    DefaultLayout,
    ProvisionedWorkspace,
    WorkspaceLayout,
)
from verity.logging import get_logger
from verity.sandbox.descriptor import (
    RESERVED_PROPOSAL_NAME,
    RESERVED_TELEMETRY_NAME,
    ProposalDescriptor,
)
from verity.sandbox.driver import SandboxDriver
from verity.sandbox.errors import SandboxError

__all__ = ["AgentSandbox"]

log = get_logger("verity.sandbox.core")


def _default_id() -> str:
    return f"art-{uuid.uuid4().hex[:12]}"


def _default_clock() -> str:
    return ""


def _parse_telemetry(raw: bytes | None) -> dict[str, object] | None:
    """Parse the advisory ``__telemetry__.json`` (5.3b); degrade to ``None`` on any decode error.

    Telemetry is observability, not the proposal — a malformed or non-object telemetry file must
    never fail an otherwise-valid cycle (ROADMAP 5.1), so it is dropped with a warning, not raised.
    """
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.warning("telemetry_unparseable", error=str(exc))
        return None
    if not isinstance(parsed, dict):
        log.warning("telemetry_not_an_object", got=type(parsed).__name__)
        return None
    return parsed


@dataclass
class AgentSandbox:
    """A real :class:`SandboxPort` over a :class:`WorkspaceLayout`, driven by a ``SandboxDriver``.

    ``static_contents`` populates the read-only roles at provision (shape ``spec/``, ``context/``
    skills); ``proposer_identity`` is the artifact ``created_by``, distinct from the verifier so
    proposer != gate holds (§7.2). ``clock`` / ``id_source`` are injected for deterministic tests.
    """

    root: Path
    driver: SandboxDriver
    schema: SchemaRegistry
    proposer_identity: str
    static_contents: Mapping[str, Mapping[str, bytes]] = field(default_factory=dict)
    layout: WorkspaceLayout = field(default_factory=DefaultLayout)
    clock: Callable[[], str] = _default_clock
    id_source: Callable[[], str] = _default_id

    _served: ServedContext | None = field(default=None, init=False)
    _workspace: ProvisionedWorkspace | None = field(default=None, init=False)

    # -- lifecycle ----------------------------------------------------------------

    def bind_run_context(self, ctx: RunContext) -> None:
        """Forward the control plane's run-identity cell to the driver, if it stamps it (a
        backend-backed driver labels its workers; an in-process driver ignores this)."""
        if isinstance(self.driver, SupportsRunContext):
            self.driver.bind_run_context(ctx)

    async def provision(self) -> None:
        self._workspace = self.layout.provision(self.root, dict(self.static_contents))

    async def regenerate(self) -> None:
        # Discard the writable workspace and re-provision empty — one ephemerality event (§3.5).
        if self.root.exists():
            shutil.rmtree(self.root)
        self._workspace = self.layout.provision(self.root, dict(self.static_contents))

    async def teardown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self._workspace = None

    async def health(self) -> bool:
        return True

    # -- cycle --------------------------------------------------------------------

    async def serve_context(self, context: ServedContext) -> None:
        self._served = context
        if context.workspace_objects:
            self._materialize_objects(context.workspace_objects)

    def _materialize_objects(
        self, objects: Mapping[str, Mapping[str, bytes]]
    ) -> None:
        """Write the control plane's durable refs into the (freshly provisioned) workspace (§9).

        Re-runs every cycle on top of the regenerated workspace, so agent edits never persist; the
        target role is the control plane's choice (a writable role) — read-only is reserved for gold
        ``static_contents``. Names may be nested paths within the role.
        """
        workspace = self._require_workspace()
        for role, files in objects.items():
            role_dir = workspace.path_for(role)
            for name, data in files.items():
                dest = role_dir / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
        log.info(
            "workspace_objects_materialized",
            roles={role: sorted(files) for role, files in objects.items()},
        )

    async def collect_proposal(self) -> ProposalEnvelope:
        served = self._served
        if served is None:
            raise SandboxError("collect_proposal called before serve_context")
        workspace = self._require_workspace()

        await self.driver.run(
            system_prompt=served.system_prompt,
            user_message=self._user_message(served),
            operations=self.schema.operations(),
            workspace=workspace,
        )

        try:
            harvested = self.layout.harvest(workspace)
        except OSError as exc:
            # An outbox I/O error is a failed cycle, not a crash: type it so the control plane's
            # degrade-don't-crash path catches it like any other sandbox failure (ROADMAP 5.1).
            raise SandboxError(f"could not harvest the outbox: {exc}") from exc
        descriptor_bytes = harvested.pop(RESERVED_PROPOSAL_NAME, None)
        telemetry_bytes = harvested.pop(RESERVED_TELEMETRY_NAME, None)
        if descriptor_bytes is None:
            raise SandboxError(
                "the agent produced no proposal "
                f"(no {RESERVED_PROPOSAL_NAME} in the outbox); outbox had: {sorted(harvested)}"
            )
        descriptor = ProposalDescriptor.from_json(descriptor_bytes)
        # Telemetry is advisory (5.3b): a corrupt __telemetry__.json must never sink an otherwise
        # good proposal, so parse defensively and degrade to None on any decode error.
        agent_telemetry = _parse_telemetry(telemetry_bytes)
        envelope = self._mint(descriptor, objects=harvested, agent_telemetry=agent_telemetry)
        log.info(
            "proposal_collected",
            op=descriptor.op_name,
            artifact=envelope.artifact.id,
            type=envelope.artifact.type,
            objects=sorted(harvested),
        )
        return envelope

    # -- internals ----------------------------------------------------------------

    def _mint(
        self,
        descriptor: ProposalDescriptor,
        *,
        objects: Mapping[str, bytes],
        agent_telemetry: Mapping[str, object] | None = None,
    ) -> ProposalEnvelope:
        """Mint the typed artifact + provenance edge from the agent's declaration (host-trusted)."""
        signature = self.schema.operation(descriptor.op_name)
        if signature is None:
            raise SandboxError(f"unknown operation in proposal: {descriptor.op_name!r}")
        type_def = self.schema.type(signature.output)
        artifact_id = self.id_source()
        timestamp = self.clock()
        artifact = Artifact(
            id=artifact_id,
            type=signature.output,
            payload=descriptor.payload,
            status=ArtifactStatus.PROPOSED,
            created_by=self.proposer_identity,
            created_at=timestamp,
            is_root=type_def.is_root if type_def is not None else False,
        )
        operation = Operation(
            op_id=f"op-{artifact_id}",
            op_name=signature.name,
            parents=descriptor.parents,
            output_id=artifact_id,
            status=OperationStatus.SUCCESS,
            created_at=timestamp,
        )
        return ProposalEnvelope(
            artifact=artifact,
            operation=operation,
            metadata=descriptor.metadata,
            objects=dict(objects),
            agent_telemetry=agent_telemetry,
        )

    def _user_message(self, served: ServedContext) -> str:
        parts: list[str] = []
        if served.tail.strip():
            parts.append(served.tail.strip())
        if served.feedback.strip():
            parts.append("# Correction from your previous attempt\n" + served.feedback.strip())
        parts.append(
            "Do the work, then call exactly one operation tool to submit your proposal. "
            "Reference the input artifact id(s) from the context above as `parents`."
        )
        return "\n\n".join(parts)

    def _require_workspace(self) -> ProvisionedWorkspace:
        if self._workspace is None:
            raise SandboxError("workspace not provisioned; call provision() first")
        return self._workspace
