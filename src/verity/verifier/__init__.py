"""The verifier service — the advisory gate host (spec §3.6, §8.3, §11).

An SDK of composable gate **primitives** (:mod:`.primitives`) realizing the rungs of the
reliability ladder, plus the :class:`~verity.verifier.service.SdkVerifier` that hosts them behind
the :class:`~verity.contracts.ports.VerifierPort`. The LLM-judge's model dependency is isolated
behind the :class:`~verity.verifier.model_client.ModelClient` seam, so the suite runs offline.
"""

from __future__ import annotations

from verity.verifier.code_runner import (
    CodeRunner,
    ContainerCodeRunner,
    FakeCodeRunner,
    RunRequest,
    RunResult,
    docker_available,
)
from verity.verifier.errors import VerifierError
from verity.verifier.model_client import (
    AnthropicModelClient,
    FakeModelClient,
    JudgeReply,
    ModelClient,
)
from verity.verifier.primitives import (
    CheckOutcome,
    GatePrimitive,
    auto_code_runner,
    deterministic_check,
    human_in_the_loop,
    llm_judge,
    model_tester,
    numeric_scorer,
)
from verity.verifier.service import SdkVerifier

__all__ = [
    "VerifierError",
    "ModelClient",
    "FakeModelClient",
    "AnthropicModelClient",
    "JudgeReply",
    "GatePrimitive",
    "CheckOutcome",
    "deterministic_check",
    "numeric_scorer",
    "llm_judge",
    "auto_code_runner",
    "human_in_the_loop",
    "model_tester",
    "CodeRunner",
    "FakeCodeRunner",
    "ContainerCodeRunner",
    "RunRequest",
    "RunResult",
    "docker_available",
    "SdkVerifier",
]
