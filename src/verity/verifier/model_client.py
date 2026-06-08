"""The model-client seam for the LLM-judge primitive (spec §3.6, §11 rung 4).

The LLM-judge is the **weakest gate that may ship as a default, and never unlabeled** (§11). Its
one dependency on a live model is isolated behind this narrow :class:`ModelClient` port, so:

* the test suite runs **offline and deterministically** against :class:`FakeModelClient`, and
* a real model backend (:class:`AnthropicModelClient`) is an *added adapter*, not a rewrite —
  the same seam reasoning as the sandbox/verifier ports (:mod:`verity.contracts.ports`).

A client is handed only the artifact under test and the declared store-slice — never the
proposer's rationale (the slice is :class:`Artifact` only, by construction, §10).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from verity.contracts import Artifact, VerdictKind
from verity.logging import get_logger
from verity.verifier.errors import VerifierError

if TYPE_CHECKING:  # the real backend is imported lazily so the package never hard-requires it
    from anthropic import AsyncAnthropic

__all__ = [
    "JudgeReply",
    "ModelClient",
    "FakeModelClient",
    "AnthropicModelClient",
    "ReplyFn",
]

log = get_logger("verity.verifier.model_client")


@dataclass(frozen=True, slots=True)
class JudgeReply:
    """A judge's structured reply (spec §3.6). ``defects`` localizes a ``refine`` (§7.5b)."""

    verdict: VerdictKind
    rationale: str
    defects: tuple[str, ...] | None = None
    score: float | None = None


@runtime_checkable
class ModelClient(Protocol):
    """The narrow port the LLM-judge primitive depends on (spec §3.6).

    One method: judge an artifact against ``criteria``, given only the declared store-slice as
    ``context``. Async so the in-process fake and a networked model backend share one shape.
    """

    async def judge(
        self, *, criteria: str, artifact: Artifact, context: tuple[Artifact, ...]
    ) -> JudgeReply: ...


# A deterministic reply rule for the fake: (criteria, artifact, context) -> reply.
ReplyFn = Callable[[str, Artifact, tuple[Artifact, ...]], JudgeReply]


def _accept(_criteria: str, _artifact: Artifact, _context: tuple[Artifact, ...]) -> JudgeReply:
    return JudgeReply(VerdictKind.ACCEPT, "fake-judge default-accepts")


@dataclass
class FakeModelClient:
    """A deterministic, in-process :class:`ModelClient` for the suite (NON-PRODUCT).

    ``reply`` is a pure rule from ``(criteria, artifact, context)`` to a :class:`JudgeReply`, so a
    test scripts any verdict without a live model; ``calls`` records each invocation for assertions.
    """

    reply: ReplyFn = _accept
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def judge(
        self, *, criteria: str, artifact: Artifact, context: tuple[Artifact, ...]
    ) -> JudgeReply:
        self.calls.append((criteria, artifact.id))
        return self.reply(criteria, artifact, context)


_SYSTEM = (
    "You are an independent gate in a verification harness. You judge ONE artifact against the "
    "given criteria, using only the artifact and the provided context. You did not produce the "
    "artifact and must not assume it is good. Reply with a single JSON object and nothing else: "
    '{"verdict": "accept"|"reject"|"refine", "rationale": str, "defects": [str, ...] | null}. '
    'Use "refine" only when the artifact is mostly sound but has a localized, nameable defect.'
)


@dataclass
class AnthropicModelClient:
    """A live :class:`ModelClient` backed by the Anthropic API (spec §3.6, §11 rung 4).

    Integration-only — **not exercised by the unit suite** (which uses :class:`FakeModelClient`).
    ``anthropic`` is imported lazily, so importing this module never requires the SDK or a key.
    A non-JSON reply is a verifier *infrastructure* failure, not a judgment, so it raises rather
    than laundering an unparseable response into an accept/reject.
    """

    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    _client: AsyncAnthropic | None = field(default=None, init=False, repr=False)

    def _ensure_client(self) -> AsyncAnthropic:
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic()
        return self._client

    async def judge(
        self, *, criteria: str, artifact: Artifact, context: tuple[Artifact, ...]
    ) -> JudgeReply:
        client = self._ensure_client()
        prompt = (
            f"Criteria:\n{criteria}\n\n"
            f"Artifact (type {artifact.type}, id {artifact.id}):\n{artifact.payload!r}\n\n"
            f"Context ({len(context)} prior artifacts):\n"
            + "\n".join(f"- {a.id} [{a.status}] {a.payload!r}" for a in context)
        )
        message = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return _parse_reply(text)


def _parse_reply(text: str) -> JudgeReply:
    try:
        data = json.loads(text)
        verdict = VerdictKind(data["verdict"])
        rationale = str(data["rationale"])
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise VerifierError(f"unparseable judge reply: {text!r}") from exc
    raw_defects = data.get("defects")
    defects = tuple(str(d) for d in raw_defects) if raw_defects else None
    return JudgeReply(verdict, rationale, defects=defects)
