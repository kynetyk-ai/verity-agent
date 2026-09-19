"""The holdout scorer's ``accept_policy`` axis — the ablation knob (experimental-design §6.2).

Drives the real ``SdkVerifier`` built by ``build_feature_engineering_verifier`` over a
``FakeCodeRunner`` whose output we script, asserting that the two policies share one scoring run but
differ only on the accept verdict:

* ``improve_over_best_prior`` (the §12 ``selection`` behaviour) rejects a worse-than-prior runnable
  submission; the existing ``test_feature_engineering_gates.py`` covers it in depth, so here we only
  pin the contrast against ``always``.
* ``always`` accepts a worse-than-prior runnable submission, records its score, and still rejects a
  non-runnable / non-scorable one (the fair, non-hobbled construct, §7).
"""

from __future__ import annotations

import asyncio

import pytest

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    VerdictKind,
    VerifierRequest,
)
from verity.domains.feature_engineering import (
    ACCEPT_ALWAYS,
    ACCEPT_IMPROVE_OVER_BEST_PRIOR,
    ENTRYPOINT,
    REQUIREMENTS,
    SUBMISSION,
    build_feature_engineering_verifier,
)
from verity.verifier import FakeCodeRunner, RunRequest, RunResult

# Reserved hold-out the gate holds (never exposed): three classes of two rows each.
RESERVED = {"1": "STAR", "2": "STAR", "3": "GALAXY", "4": "GALAXY", "5": "QSO", "6": "QSO"}
PERFECT = dict(RESERVED)  # balanced accuracy 1.0
ONE_THIRD = dict.fromkeys(RESERVED, "STAR")  # only STAR right -> 0.333


def _preds_csv(preds: dict[str, str]) -> bytes:
    rows = "\n".join(f"{rid},{label}" for rid, label in preds.items())
    return f"id,class\n{rows}\n".encode()


def _runner(by_marker: dict[bytes, dict[str, str]]) -> FakeCodeRunner:
    def script(req: RunRequest) -> RunResult:
        for marker, preds in by_marker.items():
            if marker in req.code:
                return RunResult(exit_code=0, stdout="", stderr="", output=_preds_csv(preds))
        return RunResult(exit_code=0, stdout="", stderr="", output=None)

    return FakeCodeRunner(script=script)


def _request(
    code: bytes, *, proposal_id: str = "p1", store_slice: tuple[Artifact, ...] = (),
    with_code: bool = True,
) -> VerifierRequest:
    proposal = Artifact(
        proposal_id, SUBMISSION,
        {"entrypoint": ENTRYPOINT, "requirements": REQUIREMENTS},
        ArtifactStatus.PROPOSED, "agent", "t1",
    )
    objects = {ENTRYPOINT: code, REQUIREMENTS: b"pandas==2.2.2"} if with_code else {}
    return VerifierRequest(proposal=proposal, store_slice=store_slice, objects=objects)


def _verifier(runner: FakeCodeRunner, *, accept_policy: str):
    return build_feature_engineering_verifier(
        runner,
        agent_train_csv=b"id,class\n0,STAR\n",
        reserved_test_csv=b"id\n1\n2\n3\n4\n5\n6\n",
        reserved_labels=RESERVED,
        accept_policy=accept_policy,
    )


def _accepted(art_id: str) -> Artifact:
    return Artifact(art_id, SUBMISSION, {}, ArtifactStatus.ACCEPTED, "agent", "t1")


def _dispatch(verifier, request: VerifierRequest):
    return asyncio.run(verifier.dispatch(request))


def test_unknown_accept_policy_is_a_loud_error() -> None:
    with pytest.raises(ValueError, match="unknown accept_policy"):
        _verifier(_runner({}), accept_policy="hand-wavy")


# --------------------------------------------------------------- always: accept regardless of score


def test_always_accepts_a_worse_than_prior_submission_and_records_its_score() -> None:
    # one instance scores the perfect incumbent then judges a strictly-worse challenger.
    verifier = _verifier(
        _runner({b"best": PERFECT, b"weak": ONE_THIRD}), accept_policy=ACCEPT_ALWAYS
    )
    assert _dispatch(verifier, _request(b"best", proposal_id="inc")).status is (
        ArtifactStatus.ACCEPTED
    )
    bundle = _dispatch(
        verifier, _request(b"weak", proposal_id="chal", store_slice=(_accepted("inc"),))
    )
    # under always-accept the weaker submission is accepted anyway, with its real score recorded.
    assert bundle.status is ArtifactStatus.ACCEPTED
    accept = bundle.decisions[-1]
    assert accept.gate == "score-and-accept" and accept.kind is VerdictKind.ACCEPT
    assert accept.score == pytest.approx(1 / 3)


def test_always_still_rejects_a_non_runnable_submission() -> None:
    # no scorable predictions -> nothing to record -> reject under both policies (fair construct).
    runner = FakeCodeRunner(script=lambda *_: RunResult(1, "", "boom\nTraceback line"))
    bundle = _dispatch(_verifier(runner, accept_policy=ACCEPT_ALWAYS), _request(b"crashy"))
    assert bundle.status is ArtifactStatus.REJECTED


def test_always_supersedes_nothing_so_priors_are_retained() -> None:
    verifier = _verifier(_runner({b"a": PERFECT, b"b": ONE_THIRD}), accept_policy=ACCEPT_ALWAYS)
    _dispatch(verifier, _request(b"a", proposal_id="inc"))
    bundle = _dispatch(
        verifier, _request(b"b", proposal_id="chal", store_slice=(_accepted("inc"),))
    )
    assert bundle.supersedes is None  # accumulate-context conditions keep every accepted prior


# ------------------------------------------- improve_over_best_prior: the contrasting verdict


def test_improve_over_best_prior_rejects_the_same_worse_submission() -> None:
    verifier = _verifier(
        _runner({b"best": PERFECT, b"weak": ONE_THIRD}),
        accept_policy=ACCEPT_IMPROVE_OVER_BEST_PRIOR,
    )
    assert _dispatch(verifier, _request(b"best", proposal_id="inc")).status is (
        ArtifactStatus.ACCEPTED
    )
    bundle = _dispatch(
        verifier, _request(b"weak", proposal_id="chal", store_slice=(_accepted("inc"),))
    )
    # same population as the always test above — here the worse submission is rejected.
    assert bundle.status is ArtifactStatus.REJECTED
    assert bundle.decisions[-1].gate == "selection"
    assert bundle.decisions[-1].score == pytest.approx(1 / 3)  # score still recorded on a reject
