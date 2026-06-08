"""The control-plane API surface — async (spec §3.4).

The integration layer that ties the kernel together and is the **only** service that mutates
durable state. It is async by design (§3.4): external calls are coroutines so today's in-process
stubs and tomorrow's networked services share one shape. The proven §7 commit path stays
synchronous and runs in a worker thread while the API awaits the advisory verifier; the store is
the sole, serialized mutator, so that crossing is safe.

What it does (the §3.4 API contract):

* **Configure-by-task** — register a task's configuration, stamp its schema version, and resolve
  + provision its sandbox and verifier from the provider registries (:mod:`verity.contracts`).
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
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

from verity.contracts import (
    SANDBOX_PROVIDERS,
    VERIFIER_PROVIDERS,
    Operation,
    OperationStatus,
    ProposalEnvelope,
    ProviderRegistry,
    SandboxPort,
    ServedContext,
    VerdictBundle,
    VerdictKind,
    VerifierPort,
    VerifierRequest,
)
from verity.control_plane.commit import (
    CommitOutcome,
    CommitResult,
    ShapeError,
    run_commit,
)
from verity.control_plane.config import TaskConfig
from verity.control_plane.context import AssembledContext, ContextAssembler
from verity.control_plane.independence import resolve_declared_slice
from verity.control_plane.run_report import (
    CycleInput,
    RunReport,
    Timings,
    build_run_report,
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
from verity.sandbox.errors import SandboxError

__all__ = [
    "IntakeResult",
    "OrchestrationPolicy",
    "TaskState",
    "ControlPlane",
    "ControlPlaneError",
    "UnknownTask",
    "OrchestrationError",
    "RunReport",
]

log = get_logger("verity.control_plane.api")


class ControlPlaneError(RuntimeError):
    """A misuse of the control-plane API."""


class UnknownTask(ControlPlaneError):
    """Raised when an operation references a task that was never configured."""


class OrchestrationError(ControlPlaneError):
    """Raised when a run aborts on too many consecutive sandbox failures (§3.4)."""


@dataclass(frozen=True, slots=True)
class IntakeResult:
    """The outcome of submitting one proposal (spec §3.4 intake).

    ``entered_protocol`` is False exactly when the proposal was malformed *or* the sandbox produced
    no usable proposal this cycle. On a malformed proposal ``shape_error`` carries the correction;
    on a sandbox failure ``sandbox_error`` carries the reason. In both cases **nothing is recorded**
    (no object harvested, no `proposed` row, no decision — §7.0). Otherwise ``commit`` is the §7
    result and ``harvested`` maps each outbox object's name to its content-addressed reference.
    """

    entered_protocol: bool
    shape_error: ShapeError | None = None
    commit: CommitResult | None = None
    harvested: dict[str, ObjectRef] = field(default_factory=dict)
    sandbox_error: str | None = None


@dataclass(frozen=True, slots=True)
class OrchestrationPolicy:
    """The declarative run-again / stop rule and the refine cycle cap (spec §3.4, §6, §7).

    Continuation is the control plane's call, not the verifier's. ``max_cycles`` bounds the run;
    ``refine_cap`` bounds how many times a single lineage may be sent back to refine before it
    terminates in ``rejected`` (so an un-satisfiable artifact cannot loop forever, §12 seam).
    ``stop_on_accept`` ends the run as soon as an artifact is accepted (useful for tests/demos).
    ``max_consecutive_sandbox_failures`` bounds how many cycles in a row may fail to produce a
    proposal (a timed-out / crashed / runaway sandbox) before the run aborts, so a wholly-broken
    sandbox cannot spin to ``max_cycles``; the counter resets on any cycle that does propose.
    """

    max_cycles: int = 20
    refine_cap: int = 3
    stop_on_accept: bool = False
    max_consecutive_sandbox_failures: int = 3

    def should_continue(self, *, cycles_run: int) -> bool:
        return cycles_run < self.max_cycles


@dataclass(slots=True)
class TaskState:
    """The live state for one configured task: its config and its resolved services."""

    config: TaskConfig
    sandbox: SandboxPort
    verifier: VerifierPort
    rationale: dict[str, str] = field(default_factory=dict)
    history: list[CycleInput] = field(default_factory=list)  # per-cycle facts for the RunReport


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
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._assembler = assembler if assembler is not None else ContextAssembler()
        self._policy = policy if policy is not None else OrchestrationPolicy()
        self._sandbox_providers = sandbox_providers
        self._verifier_providers = verifier_providers
        self._clock = clock
        self._timer = timer  # monotonic seconds, injected so RunReport timings are testable
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

    async def serve_context(
        self, task_id: str, *, goal: str, scratch: str = "", feedback: str = ""
    ) -> ServedContext:
        task = self._task(task_id)
        assembled: AssembledContext = self._assembler.assemble(
            self._store,
            system_prompt=task.config.system_prompt(),
            goal=goal,
            scratch=scratch,
            retrieval=task.config.retrieval,  # the task's policy drives the tail (§8.4)
        )
        served = ServedContext(
            system_prompt=assembled.stable_prefix,
            tail=assembled.volatile_tail,
            feedback=feedback,
            # Durable refs the agent builds on, selected by status/recency only (§9, no verifier
            # knowledge); re-materialized every cycle into a writable role.
            workspace_objects=task.config.object_provisioning.materialize(self._store),
        )
        await task.sandbox.serve_context(served)
        return served

    # -- proposal intake (§3.4, §7) -----------------------------------------------

    async def submit_proposal(
        self, task_id: str, envelope: ProposalEnvelope
    ) -> IntakeResult:
        """Shape-check, harvest objects before teardown, then propose and commit (§3.4, §7)."""
        task = self._task(task_id)

        # §7.0 — a malformed proposal is corrected and records NOTHING (no harvest, no row). The
        # shape check is a presence/type filter only; if it fails the verifier is never triggered.
        shape_error = task.config.shape_validator(envelope.artifact)
        if shape_error is not None:
            log.info("intake_shape_error", task_id=task_id, artifact_id=envelope.artifact.id)
            return IntakeResult(entered_protocol=False, shape_error=shape_error)

        # Harvest the outbox objects BEFORE the sandbox is regenerated, content-addressing each
        # into the object store (§3.4). Harvest-before-teardown is the hard ordering constraint.
        harvested = {name: self._store.put_object(data) for name, data in envelope.objects.items()}

        # Record the object refs as a control-plane-owned **sidecar** on the artifact (§4.1, ADR
        # 0001) — the domain payload is stored verbatim; the control plane never reaches into it.
        artifact = (
            replace(envelope.artifact, objects=tuple(harvested.items()))
            if harvested
            else envelope.artifact
        )

        # The agent's rationale is provenance/context — stored here, NEVER sent to a gate (§10).
        if envelope.metadata:
            task.rationale[artifact.id] = envelope.metadata

        self._store.propose(artifact, envelope.operation)
        self._link_revision_if_any(envelope)

        # §3.4 refine_cap: if this lineage has already been refined to the cap, a further refine
        # terminates it in 'rejected' rather than looping (computed from the store, not loop state).
        refines = self._count_lineage_refines(envelope.operation)
        refine_exhausted = refines >= self._policy.refine_cap
        result = await self._run_commit(
            task, artifact.id, envelope.objects, refine_exhausted=refine_exhausted
        )
        # On acceptance, harvest the domain's child artifacts (e.g. one Feature per declared
        # feature, §12): each is an inspectable node, committed through its own declared gate.
        if result.outcome is CommitOutcome.ACCEPTED and task.config.harvester is not None:
            await self._harvest_children(task, artifact)
        log.info("proposal_committed", task_id=task_id, outcome=result.outcome.value)
        return IntakeResult(entered_protocol=True, commit=result, harvested=harvested)

    async def _harvest_children(self, task: TaskState, parent: Artifact) -> None:
        """Mint the domain's harvested children from an accepted parent and gate each (§4.1, §12).

        Ids and lineage are minted here on the trusted side (the domain only declares content), via
        a ``harvest`` operation parent → child. Each child carries the parent's objects so its
        declared gate (e.g. ``Feature`` grounding) can check it against the submitted code.
        """
        children = task.config.harvester(parent) if task.config.harvester else []
        if not children:
            return
        parent_bytes = {
            name: self._store.get_object(ref.content_hash) for name, ref in parent.objects
        }
        for index, child in enumerate(children):
            child_id = f"{parent.id}::{child.artifact_type.lower()}::{index}"
            timestamp = self._clock()
            artifact = Artifact(
                id=child_id,
                type=child.artifact_type,
                payload=child.payload,
                status=ArtifactStatus.PROPOSED,
                created_by=parent.created_by,  # the proposer, distinct from the gate (§7.2)
                created_at=timestamp,
                objects=parent.objects,  # the child records the same code provenance
            )
            operation = Operation(
                op_id=f"op-{child_id}",
                op_name=child.op_name,
                parents=(parent.id,),
                output_id=child_id,
                status=OperationStatus.SUCCESS,
                created_at=timestamp,
            )
            self._store.propose(artifact, operation)
            result = await self._run_commit(task, child_id, parent_bytes)
            log.info(
                "child_harvested",
                parent=parent.id, child=child_id,
                type=child.artifact_type, outcome=result.outcome.value,
            )

    async def _run_commit(
        self,
        task: TaskState,
        artifact_id: str,
        objects: Mapping[str, bytes],
        *,
        refine_exhausted: bool = False,
    ) -> CommitResult:
        """Run the sync §7 commit path in a worker thread, bridging the one opaque verifier handoff.

        The ``dispatch`` closure cuts the proposal's declared store-slice (§10) and hands the whole
        proposal to the verifier in a single call; the commit path records the bundle it returns.
        """
        loop = asyncio.get_running_loop()
        bound_objects = dict(objects)

        def dispatch(artifact: Artifact) -> VerdictBundle:
            declared = task.config.gated_types.resolve(artifact.type) or frozenset()
            request = VerifierRequest(
                proposal=artifact,
                store_slice=resolve_declared_slice(self._store, artifact, declared),
                objects=bound_objects,
            )
            future = asyncio.run_coroutine_threadsafe(task.verifier.dispatch(request), loop)
            return future.result()

        return await asyncio.to_thread(
            run_commit,
            artifact_id,
            store=self._store,
            sink=self._store,
            resolve_coverage=task.config.gated_types.resolve,
            dispatch=dispatch,
            verifier_identity=task.verifier.identity,
            clock=self._clock,
            refine_exhausted=refine_exhausted,
        )

    def _count_lineage_refines(self, operation: Operation) -> int:
        """How many ancestors of this (revising) proposal are ``revised`` — the refine depth (§3.4).

        Walks the ``revises`` chain back from the flagged parent, counting ``revised`` ancestors, so
        the cap binds a *lineage* (which spans multiple artifact ids) rather than a single id.
        """
        if operation.op_name != "revises" or not operation.parents:
            return 0
        count = 0
        current: str | None = operation.parents[0]
        seen: set[str] = set()
        while current is not None and current not in seen:
            seen.add(current)
            artifact = self._store.get_artifact(current)
            if artifact is None or artifact.status is not ArtifactStatus.REVISED:
                break
            count += 1
            revises = [op for op in self._store.operations_into(current) if op.op_name == "revises"]
            current = revises[0].parents[0] if revises and revises[0].parents else None
        return count

    def _link_revision_if_any(self, envelope: ProposalEnvelope) -> None:
        """A ``revises`` operation links the flagged artifact to its revision (spec §4.1, §7.5b).

        On a ``refine`` verdict the flagged artifact is already ``revised``; when the agent
        re-proposes via a ``revises`` op, this sets its ``revised_by`` so the lineage stays intact.
        """
        op = envelope.operation
        if op.op_name != "revises" or not op.parents:
            return
        flagged = self._store.get_artifact(op.parents[0])
        if flagged is not None and flagged.status is ArtifactStatus.REVISED:
            self._store.link_revision(flagged.id, envelope.artifact.id)

    # -- cycle control: the orchestration loop (§3.4) -----------------------------

    async def run_cycle(self, task_id: str, *, goal: str, feedback: str = "") -> IntakeResult:
        """One read → propose → gate → commit cycle, then regenerate the sandbox (§3.5, §10).

        A sandbox that fails to yield a usable proposal this cycle (a timed-out / crashed / runaway
        agent, or one that proposed nothing — all surfaced as :class:`SandboxError`) does **not**
        abort the run: it is recorded as a failed cycle and the workspace is still regenerated, so a
        single bad cycle cannot destroy a multi-round session (§3.4; issue #21).
        """
        task = self._task(task_id)
        started = self._timer()
        await self.serve_context(task_id, goal=goal, feedback=feedback)
        sandbox_started = self._timer()
        try:
            envelope = await task.sandbox.collect_proposal()
        except SandboxError as exc:
            sandbox_ms = (self._timer() - sandbox_started) * 1000
            log.warning("sandbox_cycle_failed", task_id=task_id, error=str(exc))
            await task.sandbox.regenerate()  # discard the broken workspace; next cycle starts fresh
            result = IntakeResult(entered_protocol=False, sandbox_error=str(exc))
            self._record_cycle(task, goal, result, Timings(
                cycle_ms=(self._timer() - started) * 1000, sandbox_ms=sandbox_ms))
            return result
        sandbox_ms = (self._timer() - sandbox_started) * 1000
        commit_started = self._timer()
        result = await self.submit_proposal(task_id, envelope)
        commit_ms = (self._timer() - commit_started) * 1000
        # Harvest already happened in submit_proposal; now the ephemeral workspace is discarded.
        await task.sandbox.regenerate()
        self._record_cycle(
            task, goal, result,
            Timings(cycle_ms=(self._timer() - started) * 1000,
                    sandbox_ms=sandbox_ms, commit_ms=commit_ms),
            agent_telemetry=envelope.agent_telemetry,
        )
        return result

    def _record_cycle(
        self, task: TaskState, goal: str, result: IntakeResult, timings: Timings,
        *, agent_telemetry: Mapping[str, object] | None = None,
    ) -> None:
        """Append this cycle's raw facts to the task's history (the RunReport's input, 5.3a/b)."""
        task.history.append(CycleInput(
            goal=goal,
            entered_protocol=result.entered_protocol,
            shape_error=result.shape_error.message if result.shape_error is not None else None,
            sandbox_error=result.sandbox_error,
            commit=result.commit,
            timings=timings,
            agent_telemetry=agent_telemetry,
        ))

    async def run(self, task_id: str, *, goal: str) -> list[IntakeResult]:
        """Drive cycles until the orchestration policy stops the run (§3.4).

        The correction from each cycle (a shape-error or refine defects) is fed back into the next
        cycle's served context, so the agent can fix or revise (§3.5).
        """
        results: list[IntakeResult] = []
        feedback = ""
        cycles = 0
        consecutive_failures = 0
        while self._policy.should_continue(cycles_run=cycles):
            result = await self.run_cycle(task_id, goal=goal, feedback=feedback)
            results.append(result)
            cycles += 1
            if result.sandbox_error is not None:
                # A failed cycle: bound runaway failure, else feed the reason back and continue.
                consecutive_failures += 1
                cap = self._policy.max_consecutive_sandbox_failures
                if cap and consecutive_failures >= cap:
                    log.error("run_aborted_sandbox_failures", task_id=task_id, failures=cap)
                    raise OrchestrationError(
                        f"aborting run after {consecutive_failures} consecutive sandbox failures "
                        f"(last: {result.sandbox_error})"
                    )
            else:
                consecutive_failures = 0
                if (
                    self._policy.stop_on_accept
                    and result.commit is not None
                    and result.commit.status is ArtifactStatus.ACCEPTED
                ):
                    break
            feedback = _feedback_from(result)
        log.info("run_complete", task_id=task_id, cycles=cycles)
        return results

    # -- extraction (§3.4) --------------------------------------------------------

    def run_report(self, task_id: str) -> RunReport:
        """A generic, machine-readable projection of a task's run so far (ROADMAP 5.3a).

        Built from the recorded per-cycle facts + the store's lifecycle queries; it assumes nothing
        about the verifier or domain (decisions are projected verbatim, ``score`` nullable). The
        control plane *emits* it; parsing it is the receiving service's job.
        """
        task = self._task(task_id)
        policy = {
            "max_cycles": self._policy.max_cycles,
            "refine_cap": self._policy.refine_cap,
            "stop_on_accept": self._policy.stop_on_accept,
            "max_consecutive_sandbox_failures": self._policy.max_consecutive_sandbox_failures,
        }
        return build_run_report(
            task.history, self._store, task_id=task_id, policy=policy,
            generated_at=self._clock(), rationale=task.rationale,
        )

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


def _feedback_from(result: IntakeResult) -> str:
    """Render the cycle's correction for the next served context (§3.5, widened — issue #21).

    Beyond §3.5's shape-error and refine, this also threads back a **sandbox failure** reason (so a
    timed-out / runaway agent is told to produce one complete proposal) and a **rejection** reason
    (so the agent learns *why* a hard gate rejected it — previously a reject threaded nothing).
    Feeding the gate's reason back to the proposer is independent of the gate's own inputs (§10).
    """
    if result.sandbox_error is not None:
        return (
            f"sandbox-error: your previous attempt did not produce a valid submission "
            f"({result.sandbox_error}); produce exactly one complete proposal within the budget"
        )
    if not result.entered_protocol and result.shape_error is not None:
        return f"shape-error: {result.shape_error.message}"
    commit = result.commit
    if commit is not None and commit.outcome is CommitOutcome.REVISED and commit.defects:
        return "refine: " + ", ".join(commit.defects)
    if commit is not None and commit.outcome is CommitOutcome.REJECTED:
        return f"rejected: {_rejection_reason(commit)}"
    return ""


def _rejection_reason(commit: CommitResult) -> str:
    """The rejecting gate's rationale, for next-cycle feedback (§7, issue #21)."""
    for decision in commit.decisions:
        if decision.verdict is VerdictKind.REJECT:
            return decision.rationale
    return "the submission did not clear a gate"
