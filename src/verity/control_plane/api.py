"""The control-plane API surface — async (spec §3.4).

The integration layer that ties the kernel together and is the **only** service that mutates
durable state. It is async by design (§3.4): external calls are coroutines so today's in-process
stubs and tomorrow's networked services share one shape. The proven §7 commit path stays
synchronous and runs in a worker thread while the API awaits the advisory verifier; the store is
the sole, serialized mutator, so that crossing is safe.

What it does (the §3.4 API contract):

* **Configure-by-task** — register a task's configuration, stamp its schema version, and resolve
  + provision its sandbox and verifier from the provider registries (:mod:`.ports`).
* **Proposal intake** — validate shape (a malformed proposal is corrected and **records nothing**,
  not even a `proposed` row); **harvest the outbox objects before teardown** and content-address
  them into the object store; capture the agent's rationale on a **separate channel** (never sent
  to the verifier); then `propose` and run the commit path.
* **Verifier dispatch** — the only path to the verifier. Builds the declared, rationale-free
  store-slice and calls :meth:`VerifierPort.dispatch`; the sandbox has no path to it.
* **Cycle control** — serve the assembled context (§9) and apply the orchestration policy
  (run-again / stop, with the refine cycle cap).
* **Extraction** — accepted artifacts, provenance, and the rejected / superseded / revised logs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from verity.control_plane.commit import (
    CommitResult,
    GateSpec,
    GateVerdict,
    ShapeError,
    run_commit,
)
from verity.control_plane.config import TaskConfig
from verity.control_plane.context import AssembledContext, ContextAssembler
from verity.control_plane.ports import (
    SANDBOX_PROVIDERS,
    VERIFIER_PROVIDERS,
    ProposalEnvelope,
    ProviderRegistry,
    SandboxPort,
    ServedContext,
    VerifierPort,
    VerifierRequest,
)
from verity.control_plane.store import (
    Artifact,
    ArtifactStatus,
    Clock,
    ObjectRef,
    Provenance,
    SqliteStore,
    default_clock,
)
from verity.logging import get_logger

__all__ = [
    "IntakeResult",
    "OrchestrationPolicy",
    "TaskState",
    "ControlPlane",
    "ControlPlaneError",
    "UnknownTask",
]

log = get_logger("verity.control_plane.api")


class ControlPlaneError(RuntimeError):
    """A misuse of the control-plane API."""


class UnknownTask(ControlPlaneError):
    """Raised when an operation references a task that was never configured."""


@dataclass(frozen=True, slots=True)
class IntakeResult:
    """The outcome of submitting one proposal (spec §3.4 intake).

    ``entered_protocol`` is False exactly when the proposal was malformed: then ``shape_error``
    carries the correction and **nothing was recorded** (no object harvested, no `proposed` row,
    no decision — §7.0). Otherwise ``commit`` is the §7 result and ``harvested`` maps each outbox
    object's name to its content-addressed reference.
    """

    entered_protocol: bool
    shape_error: ShapeError | None = None
    commit: CommitResult | None = None
    harvested: dict[str, ObjectRef] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OrchestrationPolicy:
    """The declarative run-again / stop rule and the refine cycle cap (spec §3.4, §6, §7).

    Continuation is the control plane's call, not the verifier's. ``max_cycles`` bounds the run;
    ``refine_cap`` bounds how many times a single lineage may be sent back to refine before it
    terminates in ``rejected`` (so an un-satisfiable artifact cannot loop forever, §12 seam).
    ``stop_on_accept`` ends the run as soon as an artifact is accepted (useful for tests/demos).
    """

    max_cycles: int = 20
    refine_cap: int = 3
    stop_on_accept: bool = False

    def should_continue(self, *, cycles_run: int) -> bool:
        return cycles_run < self.max_cycles


@dataclass(slots=True)
class TaskState:
    """The live state for one configured task: its config and its resolved services."""

    config: TaskConfig
    sandbox: SandboxPort
    verifier: VerifierPort
    rationale: dict[str, str] = field(default_factory=dict)
    refine_counts: dict[str, int] = field(default_factory=dict)


class ControlPlane:
    """The sole-mutator service (spec §3.4). Owns the store, drives the commit path and the loop."""

    def __init__(
        self,
        store: SqliteStore,
        *,
        assembler: ContextAssembler | None = None,
        policy: OrchestrationPolicy | None = None,
        sandbox_providers: ProviderRegistry[SandboxPort] = SANDBOX_PROVIDERS,
        verifier_providers: ProviderRegistry[VerifierPort] = VERIFIER_PROVIDERS,
        clock: Clock = default_clock,
    ) -> None:
        self._store = store
        self._assembler = assembler if assembler is not None else ContextAssembler()
        self._policy = policy if policy is not None else OrchestrationPolicy()
        self._sandbox_providers = sandbox_providers
        self._verifier_providers = verifier_providers
        self._clock = clock
        self._tasks: dict[str, TaskState] = {}

    # -- configure-by-task --------------------------------------------------------

    async def configure(self, config: TaskConfig) -> None:
        """Register a task: stamp its schema version, resolve + provision its services (§3.4)."""
        current = self._store.current_schema_version()
        version = (current.version + 1) if current is not None else 1
        self._store.register_schema_version(
            config.schema.snapshot(version=version, created_at=self._clock())
        )
        sandbox = self._sandbox_providers.create(config.sandbox_key)
        verifier = self._verifier_providers.create(config.verifier_key)
        await sandbox.provision()
        await verifier.provision()
        self._tasks[config.task_id] = TaskState(config=config, sandbox=sandbox, verifier=verifier)
        log.info("task_configured", task_id=config.task_id, schema_version=version)

    async def teardown(self, task_id: str) -> None:
        task = self._task(task_id)
        await task.sandbox.teardown()
        await task.verifier.teardown()

    def _task(self, task_id: str) -> TaskState:
        if task_id not in self._tasks:
            raise UnknownTask(f"task not configured: {task_id!r}")
        return self._tasks[task_id]

    # -- cycle control: serve context (§9) ----------------------------------------

    async def serve_context(self, task_id: str, *, goal: str, scratch: str = "") -> ServedContext:
        task = self._task(task_id)
        assembled: AssembledContext = self._assembler.assemble(
            self._store,
            system_prompt=task.config.system_prompt(),
            goal=goal,
            scratch=scratch,
        )
        served = ServedContext(
            system_prompt=assembled.stable_prefix,
            tail=assembled.volatile_tail,
        )
        await task.sandbox.serve_context(served)
        return served

    # -- proposal intake (§3.4, §7) -----------------------------------------------

    async def submit_proposal(
        self, task_id: str, envelope: ProposalEnvelope, *, supersedes: str | None = None
    ) -> IntakeResult:
        """Validate shape, harvest objects before teardown, then propose and commit (§3.4, §7)."""
        task = self._task(task_id)

        # §7.0 — a malformed proposal is corrected and records NOTHING (no harvest, no row).
        shape_error = task.config.shape_validator(envelope.artifact)
        if shape_error is not None:
            log.info("intake_shape_error", task_id=task_id, artifact_id=envelope.artifact.id)
            return IntakeResult(entered_protocol=False, shape_error=shape_error)

        # Harvest the outbox objects BEFORE the sandbox is regenerated, content-addressing each
        # into the object store (§3.4). Harvest-before-teardown is the hard ordering constraint.
        harvested = {name: self._store.put_object(data) for name, data in envelope.objects.items()}

        # The agent's rationale is provenance/context — stored here, NEVER sent to a gate (§10).
        if envelope.metadata:
            task.rationale[envelope.artifact.id] = envelope.metadata

        self._store.propose(envelope.artifact, envelope.operation)
        result = await self._run_commit(task, envelope, supersedes=supersedes)
        log.info("proposal_committed", task_id=task_id, outcome=result.outcome.value)
        return IntakeResult(entered_protocol=True, commit=result, harvested=harvested)

    async def _run_commit(
        self, task: TaskState, envelope: ProposalEnvelope, *, supersedes: str | None
    ) -> CommitResult:
        """Run the sync §7 commit path in a worker thread, bridging to the async verifier."""
        loop = asyncio.get_running_loop()
        objects = dict(envelope.objects)

        def run_gate(gate: GateSpec, artifact: Artifact) -> GateVerdict | None:
            request = VerifierRequest(
                proposal=artifact,
                gate=gate.name,
                store_slice=self._build_slice(artifact),
                objects=objects,
            )
            future = asyncio.run_coroutine_threadsafe(task.verifier.dispatch(request), loop)
            return future.result()

        return await asyncio.to_thread(
            run_commit,
            envelope.artifact.id,
            store=self._store,
            sink=self._store,
            resolve_binding=task.config.gates.resolve,
            validate_shape=task.config.shape_validator,
            run_gate=run_gate,
            supersedes=supersedes,
            clock=self._clock,
        )

    def _build_slice(self, artifact: Artifact) -> tuple[Artifact, ...]:
        """The declared, control-plane-cut store-slice handed to a gate (spec §8.3, §10, §12).

        Default slice: the accepted incumbents of the same type (to beat) plus the rejected-log
        (the trial count, §12). It is a tuple of :class:`Artifact`, which carries no rationale, so
        the proposer's reasoning cannot ride along — "no proposer rationale reaches a gate" holds
        by construction. A per-gate declared allowlist can narrow this further later (§8.3).
        """
        accepted = self._store.query_artifacts(type=artifact.type, status=ArtifactStatus.ACCEPTED)
        rejected = self._store.rejected_log(type=artifact.type)
        return (*accepted, *rejected)

    # -- cycle control: the orchestration loop (§3.4) -----------------------------

    async def run_cycle(self, task_id: str, *, goal: str) -> IntakeResult:
        """One read → propose → gate → commit cycle, then regenerate the sandbox (§3.5, §10)."""
        task = self._task(task_id)
        await self.serve_context(task_id, goal=goal)
        envelope = await task.sandbox.collect_proposal()
        result = await self.submit_proposal(task_id, envelope)
        # Harvest already happened in submit_proposal; now the ephemeral workspace is discarded.
        await task.sandbox.regenerate()
        return result

    async def run(self, task_id: str, *, goal: str) -> list[IntakeResult]:
        """Drive cycles until the orchestration policy stops the run (§3.4)."""
        results: list[IntakeResult] = []
        cycles = 0
        while self._policy.should_continue(cycles_run=cycles):
            result = await self.run_cycle(task_id, goal=goal)
            results.append(result)
            cycles += 1
            if (
                self._policy.stop_on_accept
                and result.commit is not None
                and result.commit.status is ArtifactStatus.ACCEPTED
            ):
                break
        log.info("run_complete", task_id=task_id, cycles=cycles)
        return results

    # -- extraction (§3.4) --------------------------------------------------------

    def accepted_artifacts(self, *, type: str | None = None) -> list[Artifact]:
        return self._store.query_artifacts(type=type, status=ArtifactStatus.ACCEPTED)

    def provenance(self, artifact_id: str) -> Provenance:
        """The full lineage + decisions behind an artifact — "why do we believe X" (§4.2)."""
        return self._store.get_provenance(artifact_id)

    def rejected_log(self, *, type: str | None = None) -> list[Artifact]:
        return self._store.rejected_log(type=type)

    def superseded_log(self, *, type: str | None = None) -> list[Artifact]:
        return self._store.superseded_log(type=type)

    def revised_log(self, *, type: str | None = None) -> list[Artifact]:
        return self._store.query_artifacts(type=type, status=ArtifactStatus.REVISED)

    def rationale_for(self, artifact_id: str) -> str | None:
        """The agent's how/why summary, from the provenance/context channel (never a gate input)."""
        for task in self._tasks.values():
            if artifact_id in task.rationale:
                return task.rationale[artifact_id]
        return None
