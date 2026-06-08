"""Gate-primitive SDK + verifier-service tests (spec §3.6, §8.3, §11; ADR 0001).

The primitives are unchanged (each returns one ``GateVerdict``); the :class:`SdkVerifier` now owns a
per-type pipeline and returns a :class:`VerdictBundle` — the staging that used to live in the commit
path lives here.
"""

from __future__ import annotations

import asyncio

import pytest

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    VerdictKind,
    VerifierPort,
    VerifierRequest,
)
from verity.verifier import (
    CheckOutcome,
    FakeModelClient,
    GateStep,
    JudgeReply,
    SdkVerifier,
    VerifierError,
    deterministic_check,
    human_in_the_loop,
    llm_judge,
    model_tester,
    numeric_scorer,
)
from verity.verifier.model_client import _parse_reply
from verity.verifier.primitives import WEAK_JUDGE_LABEL


def _artifact(
    artifact_id: str = "a1", *, artifact_type: str = "Feature", payload=None, created_by: str = "a"
) -> Artifact:
    return Artifact(
        id=artifact_id,
        type=artifact_type,
        payload=payload if payload is not None else {"v": 1},
        status=ArtifactStatus.PROPOSED,
        created_by=created_by,
        created_at="t1",
    )


def _request(*, proposal: Artifact | None = None, store_slice=(), objects=None) -> VerifierRequest:
    return VerifierRequest(
        proposal=proposal if proposal is not None else _artifact(),
        store_slice=tuple(store_slice),
        objects=objects or {},
    )


def _run(primitive, request):
    return asyncio.run(primitive(request))


# ----------------------------------------------------------------- rung 2: deterministic check


def test_deterministic_check_passes_to_accept() -> None:
    primitive = deterministic_check(lambda _r: CheckOutcome(ok=True, rationale="well-formed"))
    verdict = _run(primitive, _request())
    assert verdict is not None and verdict.kind is VerdictKind.ACCEPT


def test_deterministic_check_fails_to_reject() -> None:
    primitive = deterministic_check(lambda _r: CheckOutcome(ok=False, rationale="malformed"))
    verdict = _run(primitive, _request())
    assert verdict is not None and verdict.kind is VerdictKind.REJECT


def test_deterministic_check_localized_defect_is_refine() -> None:
    primitive = deterministic_check(
        lambda _r: CheckOutcome(ok=False, rationale="one bad part", defects=("col_b",))
    )
    verdict = _run(primitive, _request())
    assert verdict is not None
    assert verdict.kind is VerdictKind.REFINE
    assert verdict.defects == ("col_b",)


# ----------------------------------------------------------------- rung 1: numeric scorer


def test_numeric_scorer_accepts_when_it_beats_the_baseline() -> None:
    primitive = numeric_scorer(lambda _r: 0.9)
    verdict = _run(primitive, _request())
    assert verdict is not None and verdict.kind is VerdictKind.ACCEPT
    assert verdict.score == pytest.approx(0.9)


def test_numeric_scorer_rejects_when_it_does_not_beat_the_baseline() -> None:
    primitive = numeric_scorer(lambda _r: 0.0)
    verdict = _run(primitive, _request())
    assert verdict is not None and verdict.kind is VerdictKind.REJECT


def test_numeric_scorer_nets_out_cost() -> None:
    primitive = numeric_scorer(lambda _r: 0.5, cost=lambda _a: 0.6)
    verdict = _run(primitive, _request())
    assert verdict is not None and verdict.kind is VerdictKind.REJECT
    assert verdict.score == pytest.approx(-0.1)


def test_numeric_scorer_must_beat_incumbent_by_margin() -> None:
    incumbents = (_artifact("inc"),)
    primitive = numeric_scorer(lambda _r: 0.80, incumbent=lambda _slice: 0.78, margin=0.05)
    verdict = _run(primitive, _request(store_slice=incumbents))
    assert verdict is not None and verdict.kind is VerdictKind.REJECT

    primitive_ok = numeric_scorer(lambda _r: 0.90, incumbent=lambda _slice: 0.78, margin=0.05)
    verdict_ok = _run(primitive_ok, _request(store_slice=incumbents))
    assert verdict_ok is not None and verdict_ok.kind is VerdictKind.ACCEPT


# ----------------------------------------------------------------- rung 4: llm-judge (labeled weak)


def test_llm_judge_wraps_reply_and_labels_it_weak() -> None:
    client = FakeModelClient(reply=lambda _c, _a, _x: JudgeReply(VerdictKind.ACCEPT, "reads ok"))
    verdict = _run(llm_judge(client, criteria="is it insightful?"), _request())
    assert verdict is not None and verdict.kind is VerdictKind.ACCEPT
    assert verdict.rationale.startswith(WEAK_JUDGE_LABEL)  # never an unlabeled judged accept (§11)
    assert client.calls == [("is it insightful?", "a1")]


def test_llm_judge_carries_refine_defects_through() -> None:
    client = FakeModelClient(
        reply=lambda _c, _a, _ctx: JudgeReply(VerdictKind.REFINE, "almost", defects=("tone",))
    )
    verdict = _run(llm_judge(client, criteria="x"), _request())
    assert verdict is not None
    assert verdict.kind is VerdictKind.REFINE and verdict.defects == ("tone",)


def test_llm_judge_sees_only_artifacts_in_context() -> None:
    seen: list[object] = []

    def spy(_criteria: str, _artifact: Artifact, context: tuple[Artifact, ...]) -> JudgeReply:
        seen.extend(context)
        return JudgeReply(VerdictKind.ACCEPT, "ok")

    incumbents = (_artifact("inc"),)
    _run(llm_judge(FakeModelClient(reply=spy), criteria="x"), _request(store_slice=incumbents))
    assert all(isinstance(item, Artifact) for item in seen)


# ----------------------------------------------------------------- rung 5 + seam


def test_human_in_the_loop_defers_with_no_verdict() -> None:
    assert _run(human_in_the_loop(), _request()) is None  # ⇒ rest at tentative (§7.4, §8.3)


def test_model_tester_is_a_loud_seam_until_phase_4() -> None:
    with pytest.raises(NotImplementedError):
        _run(model_tester(), _request())


# ----------------------------------------------------------------- the model-client parser


def test_parse_reply_reads_a_structured_verdict() -> None:
    reply = _parse_reply('{"verdict": "refine", "rationale": "fix col", "defects": ["col_b"]}')
    assert reply.verdict is VerdictKind.REFINE
    assert reply.defects == ("col_b",)


def test_parse_reply_raises_on_unparseable_reply() -> None:
    with pytest.raises(VerifierError):
        _parse_reply("the artifact looks fine to me")


# ----------------------------------------------------------------- the SdkVerifier (bundles)


def _verifier() -> SdkVerifier:
    return SdkVerifier(
        identity="test-verifier",
        pipelines={
            "Feature": (
                GateStep(
                    "well-formed",
                    deterministic_check(lambda _r: CheckOutcome(ok=True, rationale="ok")),
                    is_hard=False,
                ),
                GateStep("worth-keeping", numeric_scorer(lambda _r: 0.9), is_hard=True),
            ),
            "Human": (
                GateStep("needs-human", human_in_the_loop(), is_hard=True, requires_human=True),
            ),
        },
    )


def test_sdk_verifier_satisfies_the_port_protocol() -> None:
    assert isinstance(_verifier(), VerifierPort)


def test_sdk_verifier_runs_its_pipeline_and_returns_an_accepted_bundle() -> None:
    verifier = _verifier()
    bundle = asyncio.run(verifier.dispatch(_request()))
    assert bundle.status is ArtifactStatus.ACCEPTED
    assert [d.gate for d in bundle.decisions] == ["well-formed", "worth-keeping"]
    assert [r.proposal.type for r in verifier.requests] == ["Feature"]


def test_sdk_verifier_rests_at_tentative_on_a_human_deferral() -> None:
    verifier = _verifier()
    bundle = asyncio.run(verifier.dispatch(_request(proposal=_artifact(artifact_type="Human"))))
    # the hard human check returned no verdict → cleared cheap (none), not hard → tentative
    assert bundle.status is ArtifactStatus.TENTATIVE


def test_sdk_verifier_short_circuits_on_reject() -> None:
    verifier = SdkVerifier(
        identity="v",
        pipelines={
            "Feature": (
                GateStep(
                    "no",
                    deterministic_check(lambda _r: CheckOutcome(ok=False, rationale="bad")),
                ),
                GateStep("never", numeric_scorer(lambda _r: 1.0), is_hard=True),
            )
        },
    )
    bundle = asyncio.run(verifier.dispatch(_request()))
    assert bundle.status is ArtifactStatus.REJECTED
    assert [d.gate for d in bundle.decisions] == ["no"]  # the hard check never ran


def test_sdk_verifier_uncovered_type_is_a_loud_error_not_a_silent_pass() -> None:
    verifier = _verifier()
    with pytest.raises(VerifierError):
        asyncio.run(verifier.dispatch(_request(proposal=_artifact(artifact_type="Unknown"))))


def test_sdk_verifier_lifecycle_is_a_noop_surface() -> None:
    verifier = _verifier()
    asyncio.run(verifier.provision())
    assert verifier.provisioned is True
    assert asyncio.run(verifier.health()) is True
    asyncio.run(verifier.teardown())
    assert verifier.provisioned is False
