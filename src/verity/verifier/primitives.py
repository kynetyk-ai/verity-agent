"""The gate-primitive SDK — the rungs of the reliability ladder as composable plugins (spec §11).

A **gate plugin** is the irreducible judging work (§8.3); physically it lives here in the verifier,
not the control plane. Each constructor below builds a plugin that realizes one rung of the
reliability ladder (§11) and encodes that rung's *discipline*, leaving only the domain-specific
hook to the caller:

* :func:`deterministic_check` — **rung 2**: a pass/fail predicate. The cheap, always-run validity
  gate; clearing it earns ``tentative`` (§6, [PILAR]). It can ``refine`` when the predicate
  localizes a defect rather than failing outright.
* :func:`numeric_scorer` — **rung 1**: the most trustworthy gate. Encodes "earns its place iff it
  improves the score **net of cost**, beating the incumbent" — the mechanical comparison is here;
  the domain supplies the score, the cost, and how to read the incumbent's score (§11, §12).
* :func:`llm_judge` — **rung 4**: a model verdict, **labeled weak** so a judged accept is never
  mistaken for a measured one (§11). The model dependency is isolated behind
  :class:`~verity.verifier.model_client.ModelClient`.
* :func:`human_in_the_loop` — **rung 5**: returns *no* verdict, so a ``requires_human`` gate rests
  the artifact at ``tentative`` rather than auto-resolving (§8.3, §7.4).
* :func:`model_tester` — **seam**: the train-and-score rung lands with the feature-engineering
  domain (§12, Phase 4); the contract is fixed here, the implementation is deferred.

The :func:`auto_code_runner` rung (executes object attachments in isolation) lands in Phase 2.2,
composed over a container ``CodeRunner``.

A primitive is a coroutine ``(VerifierRequest) -> GateVerdict | None`` — async so a deterministic
check and a networked judge share one shape. Returning ``None`` means "cannot auto-resolve" (§7.4).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from verity.control_plane.commit import GateVerdict
from verity.control_plane.ports import VerifierRequest
from verity.control_plane.store import Artifact, VerdictKind
from verity.logging import get_logger
from verity.verifier.code_runner import CodeRunner, RunRequest, RunResult
from verity.verifier.model_client import ModelClient

__all__ = [
    "GatePrimitive",
    "CheckOutcome",
    "DeterministicPredicate",
    "ScoreFn",
    "CostFn",
    "IncumbentScoreFn",
    "deterministic_check",
    "numeric_scorer",
    "llm_judge",
    "auto_code_runner",
    "human_in_the_loop",
    "model_tester",
    "InputsFn",
    "VerdictFromRun",
    "WEAK_JUDGE_LABEL",
]

log = get_logger("verity.verifier.primitives")

# A gate plugin: judge a request, returning a verdict or ``None`` (cannot auto-resolve, §7.4).
GatePrimitive = Callable[[VerifierRequest], Awaitable[GateVerdict | None]]


# ----------------------------------------------------------------- rung 2: deterministic check


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    """The result of a deterministic predicate (spec §11 rung 2).

    ``ok`` passes the gate. When it fails, ``defects`` decides the verdict: a non-empty list
    localizes the failure into a ``refine`` (fix this part, §7.5b); an empty/``None`` list is a
    flat ``reject``.
    """

    ok: bool
    rationale: str
    defects: tuple[str, ...] | None = None


DeterministicPredicate = Callable[[VerifierRequest], CheckOutcome]


def deterministic_check(predicate: DeterministicPredicate) -> GatePrimitive:
    """Build a rung-2 gate from a pass/fail ``predicate`` (spec §11)."""

    async def primitive(request: VerifierRequest) -> GateVerdict | None:
        outcome = predicate(request)
        if outcome.ok:
            return GateVerdict(VerdictKind.ACCEPT, outcome.rationale)
        if outcome.defects:
            return GateVerdict(VerdictKind.REFINE, outcome.rationale, defects=outcome.defects)
        return GateVerdict(VerdictKind.REJECT, outcome.rationale)

    return primitive


# ----------------------------------------------------------------- rung 1: numeric scorer

ScoreFn = Callable[[VerifierRequest], float]
CostFn = Callable[[Artifact], float]
IncumbentScoreFn = Callable[[tuple[Artifact, ...]], float | None]


def numeric_scorer(
    score: ScoreFn,
    *,
    cost: CostFn | None = None,
    incumbent: IncumbentScoreFn | None = None,
    margin: float = 0.0,
) -> GatePrimitive:
    """Build the rung-1 selection gate: accept iff the artifact improves the score net of cost.

    ``score`` is the artifact's raw score; ``cost`` its complexity penalty (default 0); the net is
    ``score - cost``. ``incumbent`` reads the best net score to beat from the declared store-slice
    (``None`` ⇒ no incumbent, so the baseline is 0). The artifact is accepted iff
    ``net > baseline + margin`` — the spec's "improve the model net of added complexity" (§11, §12).
    """

    async def primitive(request: VerifierRequest) -> GateVerdict | None:
        raw = score(request)
        penalty = cost(request.proposal) if cost is not None else 0.0
        net = raw - penalty
        prior = incumbent(request.store_slice) if incumbent is not None else None
        baseline = prior if prior is not None else 0.0
        if net > baseline + margin:
            return GateVerdict(
                VerdictKind.ACCEPT,
                f"net {net:.4f} beats baseline {baseline:.4f} (margin {margin:.4f})",
                score=net,
            )
        return GateVerdict(
            VerdictKind.REJECT,
            f"net {net:.4f} does not beat baseline {baseline:.4f} (margin {margin:.4f})",
            score=net,
        )

    return primitive


# ----------------------------------------------------------------- rung 4: llm-judge (labeled weak)

# Every llm-judge verdict is prefixed with this so a judged accept is never read as a measured one.
WEAK_JUDGE_LABEL = "[weak: llm-judge — a model verdict, not a measurement]"


def llm_judge(client: ModelClient, *, criteria: str) -> GatePrimitive:
    """Build a rung-4 gate from a :class:`ModelClient`; its verdict is **labeled weak** (§11)."""

    async def primitive(request: VerifierRequest) -> GateVerdict | None:
        reply = await client.judge(
            criteria=criteria, artifact=request.proposal, context=request.store_slice
        )
        return GateVerdict(
            reply.verdict,
            f"{WEAK_JUDGE_LABEL} {reply.rationale}",
            defects=reply.defects,
            score=reply.score,
        )

    return primitive


# --------------------------------------------------- rung 1/2: auto-code-runner (executed)

# Derive the read-only datasets a run needs from the request (e.g. the agent's data + a reserved
# verification set, §12); and turn a finished run into a verdict (the domain's scoring lives here).
InputsFn = Callable[[VerifierRequest], Mapping[str, bytes]]
VerdictFromRun = Callable[[RunResult], GateVerdict]


def _ran_clean_verdict(result: RunResult) -> GateVerdict:
    """The default interpretation: a clean exit passes; a crash or timeout fails (validity rung)."""
    if result.timed_out:
        return GateVerdict(VerdictKind.REJECT, "submission timed out before completing")
    if result.exit_code != 0:
        stderr = result.stderr.strip()
        tail = stderr.splitlines()[-1] if stderr else "no stderr"
        return GateVerdict(VerdictKind.REJECT, f"submission exited {result.exit_code}: {tail}")
    return GateVerdict(VerdictKind.ACCEPT, "submission ran clean")


def auto_code_runner(
    runner: CodeRunner,
    *,
    entrypoint: str = "submission.py",
    inputs: InputsFn | None = None,
    interpret: VerdictFromRun | None = None,
    output_name: str = "result.json",
    timeout_s: float = 30.0,
) -> GatePrimitive:
    """Build the rung that *executes* the submitted code in isolation (spec §3.6, §11, §12).

    Pulls the ``entrypoint`` object attachment off the request, runs it via the container ``runner``
    over the datasets ``inputs`` selects, and maps the run to a verdict with ``interpret`` (default:
    ran-clean → accept, crash/timeout → reject — the validity rung). A domain scores by passing an
    ``interpret`` that reads ``result.output``; the model-tester rung (§12) composes over this.
    A missing attachment is a reject, not a crash — the agent simply didn't submit runnable code.
    """
    decide = interpret if interpret is not None else _ran_clean_verdict

    async def primitive(request: VerifierRequest) -> GateVerdict | None:
        code = request.objects.get(entrypoint)
        if code is None:
            return GateVerdict(
                VerdictKind.REJECT, f"no {entrypoint!r} object attachment to execute"
            )
        run_inputs = inputs(request) if inputs is not None else {}
        result = await runner.run(
            RunRequest(
                code=code,
                entrypoint=entrypoint,
                inputs=run_inputs,
                output_name=output_name,
                timeout_s=timeout_s,
            )
        )
        return decide(result)

    return primitive


# ----------------------------------------------------------------- rung 5: human-in-the-loop


def human_in_the_loop() -> GatePrimitive:
    """Build a rung-5 gate that never auto-resolves: ``None`` ⇒ rest at ``tentative`` (§8.3)."""

    async def primitive(request: VerifierRequest) -> GateVerdict | None:
        log.info("human_gate_deferred", gate=request.gate, proposal=request.proposal.id)
        return None

    return primitive


# ----------------------------------------------------------------- seam: model-tester (Phase 4)


def model_tester() -> GatePrimitive:
    """Seam for the train-and-score rung — implemented with the §12 domain (Phase 4).

    The contract (a :data:`GatePrimitive`) is fixed now; calling it before Phase 4 is a clear,
    loud error rather than a silent default.
    """

    async def primitive(request: VerifierRequest) -> GateVerdict | None:
        raise NotImplementedError(
            "the model-tester primitive lands with the feature-engineering domain (§12, Phase 4)"
        )

    return primitive
