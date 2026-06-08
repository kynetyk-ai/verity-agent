"""The `RunReport` — a generic, machine-readable projection of what a run did (ROADMAP 5.3a).

A **faithful JSON projection of the kernel's own vocabulary** — artifact lifecycle states (§6),
verdict kinds, the per-check decisions, and provenance. It assumes **nothing** about a verifier or a
domain: a ``score`` is one *optional* field a decision may carry, never elevated to a first-class
concept, and there is no built-in "score trajectory" / "best score" (those would assume numeric
verdicts). The control plane **emits** this structured record; **parsing/interpreting it is the
receiving service's job** (the eval harness extracts scores itself from ``decisions[].score``).

This module imports only kernel value types + the store read surface — never the domains or the
verifier — so the report stays domain/verifier-agnostic by construction.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from verity.contracts import ArtifactStatus
from verity.control_plane.commit import CommitResult
from verity.control_plane.store import Store

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "Timings",
    "CycleInput",
    "DecisionReport",
    "ProposalReport",
    "CommitReport",
    "CycleReport",
    "RunSummary",
    "RunReport",
    "build_run_report",
]

# Bump when the report's JSON shape changes, so a receiving service can parse safely.
REPORT_SCHEMA_VERSION = 1

JsonDict = dict[str, Any]


def _round_ms(value: float | None) -> float | None:
    """Round a latency to 3 decimals for clean JSON (``None`` passes through)."""
    return round(value, 3) if value is not None else None


# -------------------------------------------------------------- raw per-cycle input (from the CP)


@dataclass(frozen=True, slots=True)
class Timings:
    """Best-effort per-cycle wall-clock latencies in milliseconds (``None`` when not measured)."""

    cycle_ms: float | None = None
    sandbox_ms: float | None = None
    commit_ms: float | None = None

    def to_dict(self) -> JsonDict:
        return {
            "cycle_ms": _round_ms(self.cycle_ms), "sandbox_ms": _round_ms(self.sandbox_ms),
            "commit_ms": _round_ms(self.commit_ms),
        }


@dataclass(frozen=True, slots=True)
class CycleInput:
    """The raw facts the control plane records per cycle — the builder's input (not the report).

    Kernel-typed only (a :class:`CommitResult` or ``None``), so this module never depends on the
    control-plane API. ``shape_error`` / ``sandbox_error`` are the §7.0 / §21 failure messages.
    """

    goal: str
    entered_protocol: bool
    shape_error: str | None = None
    sandbox_error: str | None = None
    commit: CommitResult | None = None
    timings: Timings = field(default_factory=Timings)
    agent_telemetry: Mapping[str, Any] | None = None  # tokens / steps / model (5.3b), or None


# ----------------------------------------------------------------- the report (machine-readable)


@dataclass(frozen=True, slots=True)
class DecisionReport:
    """One gate's recorded ruling, verbatim. ``score`` is nullable — never assumed present."""

    gate: str
    verdict: str
    rationale: str
    defects: list[str]
    score: float | None

    def to_dict(self) -> JsonDict:
        return {
            "gate": self.gate, "verdict": self.verdict, "rationale": self.rationale,
            "defects": self.defects, "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class ProposalReport:
    """What the agent proposed this cycle (object *names* only, never bytes)."""

    artifact_id: str
    type: str
    operation: str | None
    parents: list[str]
    objects: list[str]

    def to_dict(self) -> JsonDict:
        return {
            "artifact_id": self.artifact_id, "type": self.type, "operation": self.operation,
            "parents": self.parents, "objects": self.objects,
        }


@dataclass(frozen=True, slots=True)
class CommitReport:
    """The verifier's verdict as recorded, projected verbatim (no interpretation)."""

    outcome: str
    status: str
    supersedes: str | None
    defects: list[str]
    decisions: list[DecisionReport]

    def to_dict(self) -> JsonDict:
        return {
            "outcome": self.outcome, "status": self.status, "supersedes": self.supersedes,
            "defects": self.defects, "decisions": [d.to_dict() for d in self.decisions],
        }


@dataclass(frozen=True, slots=True)
class CycleReport:
    """One cycle, faithful to what happened — including a skipped (sandbox-failed) cycle."""

    index: int
    goal: str
    entered_protocol: bool
    shape_error: str | None
    sandbox_error: str | None
    proposal: ProposalReport | None
    commit: CommitReport | None
    rationale: str | None
    timings: Timings
    agent_telemetry: JsonDict | None

    def to_dict(self) -> JsonDict:
        return {
            "index": self.index, "goal": self.goal, "entered_protocol": self.entered_protocol,
            "shape_error": self.shape_error, "sandbox_error": self.sandbox_error,
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "commit": self.commit.to_dict() if self.commit else None,
            "rationale": self.rationale, "timings": self.timings.to_dict(),
            "agent_telemetry": self.agent_telemetry,
        }


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Run-level aggregates — lifecycle/provenance facts only, no score math."""

    cycles_run: int
    outcomes: dict[str, int]
    accepted: list[JsonDict]
    superseded: list[JsonDict]
    trial_count: int
    refine_count: int
    object_count: int
    wall_time_ms: float | None

    def to_dict(self) -> JsonDict:
        return {
            "cycles_run": self.cycles_run, "outcomes": self.outcomes, "accepted": self.accepted,
            "superseded": self.superseded, "trial_count": self.trial_count,
            "refine_count": self.refine_count, "object_count": self.object_count,
            "wall_time_ms": self.wall_time_ms,
        }


@dataclass(frozen=True, slots=True)
class RunReport:
    """A versioned, machine-readable record of one run (ROADMAP 5.3a)."""

    report_schema_version: int
    task_id: str
    generated_at: str
    policy: JsonDict
    cycles: list[CycleReport]
    summary: RunSummary
    agent_telemetry: JsonDict | None  # run-total tokens / steps / tool-calls (5.3b), or None

    def to_dict(self) -> JsonDict:
        return {
            "report_schema_version": self.report_schema_version, "task_id": self.task_id,
            "generated_at": self.generated_at, "policy": self.policy,
            "cycles": [c.to_dict() for c in self.cycles], "summary": self.summary.to_dict(),
            "agent_telemetry": self.agent_telemetry,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def render_text(self) -> str:
        """A compact human summary — a convenience, **not** the surface other services consume."""
        s = self.summary
        lines = [
            f"RunReport task={self.task_id} cycles={s.cycles_run}",
            "  outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(s.outcomes.items())),
            f"  accepted={len(s.accepted)} superseded={len(s.superseded)} "
            f"trials={s.trial_count} refines={s.refine_count} objects={s.object_count}",
        ]
        for c in self.cycles:
            outcome = c.sandbox_error and "sandbox-failed" or (
                c.shape_error and "shape-error" or (c.commit.outcome if c.commit else "—")
            )
            who = c.proposal.artifact_id if c.proposal else "(no proposal)"
            lines.append(f"  [{c.index}] {outcome}: {who}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------- the builder


def build_run_report(
    cycles: Sequence[CycleInput],
    store: Store,
    *,
    task_id: str,
    policy: JsonDict,
    generated_at: str,
    rationale: Mapping[str, str],
) -> RunReport:
    """Project the recorded cycles + the store into a generic :class:`RunReport`.

    Reads only the kernel's lifecycle queries (``query_artifacts`` / ``rejected_log`` /
    ``superseded_log`` / ``operations_into``) and projects decisions verbatim — no domain knowledge.
    """
    cycle_reports = [
        CycleReport(
            index=i,
            goal=ci.goal,
            entered_protocol=ci.entered_protocol,
            shape_error=ci.shape_error,
            sandbox_error=ci.sandbox_error,
            proposal=_proposal(store, ci.commit),
            commit=_commit(store, ci.commit),
            rationale=_rationale(rationale, ci.commit),
            timings=ci.timings,
            agent_telemetry=dict(ci.agent_telemetry) if ci.agent_telemetry is not None else None,
        )
        for i, ci in enumerate(cycles)
    ]
    return RunReport(
        report_schema_version=REPORT_SCHEMA_VERSION,
        task_id=task_id,
        generated_at=generated_at,
        policy=policy,
        cycles=cycle_reports,
        summary=_summary(store, cycles),
        agent_telemetry=_run_telemetry(cycles),
    )


def _run_telemetry(cycles: Sequence[CycleInput]) -> JsonDict | None:
    """Sum the per-cycle agent telemetry into a run total (``None`` if no cycle reported any)."""
    present = [ci.agent_telemetry for ci in cycles if ci.agent_telemetry is not None]
    if not present:
        return None
    totals: JsonDict = {
        key: sum(int(t.get(key, 0) or 0) for t in present)
        for key in ("input_tokens", "output_tokens", "total_tokens", "model_steps", "tool_calls")
    }
    models = [t.get("model") for t in present if t.get("model")]
    totals["model"] = models[-1] if models else None
    return totals


def _proposal(store: Store, commit: CommitResult | None) -> ProposalReport | None:
    if commit is None:
        return None
    artifact = store.get_artifact(commit.artifact_id)
    if artifact is None:
        return None
    ops = [o for o in store.operations_into(artifact.id) if o.output_id == artifact.id]
    op = ops[0] if ops else None
    return ProposalReport(
        artifact_id=artifact.id,
        type=artifact.type,
        operation=op.op_name if op is not None else None,
        parents=list(op.parents) if op is not None else [],
        objects=[name for name, _ in artifact.objects],
    )


def _commit(store: Store, commit: CommitResult | None) -> CommitReport | None:
    if commit is None:
        return None
    superseded = [a.id for a in store.superseded_log() if a.superseded_by == commit.artifact_id]
    return CommitReport(
        outcome=commit.outcome.value,
        status=commit.status.value,
        supersedes=superseded[0] if superseded else None,
        defects=list(commit.defects or ()),
        decisions=[
            DecisionReport(
                gate=d.gate, verdict=d.verdict.value, rationale=d.rationale,
                defects=list(d.defects or ()), score=d.score,
            )
            for d in commit.decisions
        ],
    )


def _rationale(rationale: Mapping[str, str], commit: CommitResult | None) -> str | None:
    if commit is None:
        return None
    return rationale.get(commit.artifact_id)


def _outcome_label(ci: CycleInput) -> str:
    if ci.sandbox_error is not None:
        return "sandbox_failed"
    if not ci.entered_protocol:
        return "shape_error"
    if ci.commit is not None:
        return ci.commit.outcome.value
    return "unknown"


def _summary(store: Store, cycles: Sequence[CycleInput]) -> RunSummary:
    outcomes: dict[str, int] = {}
    for ci in cycles:
        label = _outcome_label(ci)
        outcomes[label] = outcomes.get(label, 0) + 1
    object_hashes = {
        ref.content_hash for a in store.query_artifacts() for _, ref in a.objects
    }
    cycle_times = [ci.timings.cycle_ms for ci in cycles if ci.timings.cycle_ms is not None]
    return RunSummary(
        cycles_run=len(cycles),
        outcomes=outcomes,
        accepted=[
            {"id": a.id, "type": a.type}
            for a in store.query_artifacts(status=ArtifactStatus.ACCEPTED)
        ],
        superseded=[
            {"id": a.id, "superseded_by": a.superseded_by} for a in store.superseded_log()
        ],
        trial_count=len(store.rejected_log()),
        refine_count=len(store.query_artifacts(status=ArtifactStatus.REVISED)),
        object_count=len(object_hashes),
        wall_time_ms=sum(cycle_times) if cycle_times else None,
    )
