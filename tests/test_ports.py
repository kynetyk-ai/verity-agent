"""Service-port and provider-registry tests (spec §3.3–§3.6) — ROADMAP Phase 1.2."""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    ProviderError,
    ProviderRegistry,
    VerdictBundle,
    VerifierPort,
    VerifierRequest,
)


class _FakeVerifier:
    """A minimal in-process VerifierPort: lifecycle no-ops, dispatch returns a fixed bundle."""

    def __init__(self) -> None:
        self.identity = "fake-verifier"
        self.provisioned = False

    async def dispatch(self, request: VerifierRequest) -> VerdictBundle:
        return VerdictBundle(ArtifactStatus.ACCEPTED)

    async def provision(self) -> None:
        self.provisioned = True

    async def teardown(self) -> None:
        self.provisioned = False

    async def health(self) -> bool:
        return True


def _artifact(artifact_id: str) -> Artifact:
    return Artifact(
        id=artifact_id,
        type="Note",
        payload={"text": "x"},
        status=ArtifactStatus.PROPOSED,
        created_by="agent",
        created_at="t1",
    )


def test_fake_verifier_satisfies_the_port_protocol() -> None:
    assert isinstance(_FakeVerifier(), VerifierPort)


def test_provider_registry_resolves_by_key() -> None:
    registry: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    registry.register("stub", _FakeVerifier)
    assert registry.keys() == ("stub",)
    port = registry.create("stub")
    assert isinstance(port, VerifierPort)


def test_provider_registry_unknown_key_raises() -> None:
    registry: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    with pytest.raises(ProviderError):
        registry.create("missing")


def test_provider_registry_rejects_duplicate_registration() -> None:
    registry: ProviderRegistry[VerifierPort] = ProviderRegistry("verifier")
    registry.register("stub", _FakeVerifier)
    with pytest.raises(ProviderError):
        registry.register("stub", _FakeVerifier)


def test_dispatch_excludes_rationale_by_construction() -> None:
    # A VerifierRequest carries only the proposal, the declared slice, and objects — no field
    # (and no gate name) through which the proposer's rationale could reach the verifier (§10).
    request = VerifierRequest(proposal=_artifact("n1"))
    field_names = {f.name for f in dataclasses.fields(request)}
    assert field_names == {"proposal", "store_slice", "objects"}
    assert "rationale" not in field_names


def test_async_dispatch_and_lifecycle_round_trip() -> None:
    verifier = _FakeVerifier()

    async def scenario() -> VerdictBundle:
        await verifier.provision()
        assert await verifier.health() is True
        bundle = await verifier.dispatch(VerifierRequest(proposal=_artifact("n1")))
        await verifier.teardown()
        return bundle

    bundle = asyncio.run(scenario())
    assert bundle.status is ArtifactStatus.ACCEPTED
    assert verifier.provisioned is False
