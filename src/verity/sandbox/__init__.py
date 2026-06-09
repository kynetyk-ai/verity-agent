"""The sandbox service — the real agent runtime behind :class:`~verity.contracts.SandboxPort`.

This package exports only the **framework-neutral** core, so importing :mod:`verity.sandbox` never
requires the optional Deep Agents extra. The driver and its factory are explicit submodule
imports (:mod:`verity.sandbox.deepagents_driver`, :mod:`verity.sandbox.registration`) used where the
``sandbox`` extra is installed.
"""

from __future__ import annotations

from verity.sandbox.core import AgentSandbox
from verity.sandbox.descriptor import RESERVED_PROPOSAL_NAME, ProposalDescriptor
from verity.sandbox.driver import SandboxDriver
from verity.sandbox.errors import SandboxError
from verity.sandbox.model_spec import ModelSpec, coerce_model, resolve_model

__all__ = [
    "AgentSandbox",
    "SandboxDriver",
    "ProposalDescriptor",
    "RESERVED_PROPOSAL_NAME",
    "SandboxError",
    "ModelSpec",
    "resolve_model",
    "coerce_model",
]
