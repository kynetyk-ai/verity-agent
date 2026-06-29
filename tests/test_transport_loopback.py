"""The transport seam over the loopback transport (ROADMAP Phase 7.2). Offline — no socket.

The loopback transport wires a client straight to a server's ``handle``, so this proves the whole
seam — serialize → dispatch → serialize → deserialize — without a network: every port method
round-trips, object bytes survive intact, the verifier's identity is learned over the wire, and the
boundary errors (`GateUnavailable` / `SandboxError` / `VerifierError`) cross as themselves. The last
test drives a real `ControlPlane` whose verifier is remote and asserts degrade-don't-crash survives
the hop with no control-plane change.
"""

from __future__ import annotations

import asyncio

import pytest

from verity.contracts import (
    Artifact,
    ArtifactStatus,
    GateDecision,
    GateUnavailable,
    Operation,
    OperationStatus,
    ProposalEnvelope,
    ServedContext,
    VerdictBundle,
    VerdictKind,
    VerifierRequest,
    VerifierSetup,
)
from verity.sandbox.errors import SandboxError
from verity.transport import (
    LoopbackTransport,
    RemoteSandbox,
    RemoteVerifier,
    SandboxServer,
    TransportError,
    VerifierServer,
)
from verity.transport.base import Transport
from verity.verifier.errors import VerifierError

_ACCEPT = VerdictBundle(
    ArtifactStatus.ACCEPTED, (GateDecision("worth-keeping", VerdictKind.ACCEPT, "ok", score=0.9),)
)
_ARTIFACT = Artifact("a1", "Note", {"text": "hi"}, ArtifactStatus.PROPOSED, "agent", "t1")
_OP = Operation("op1", "author", ("src",), "a1", OperationStatus.SUCCESS, "t1")


# ----------------------------------------------------------------------------- in-test impls


class _RecordingVerifier:
    """A VerifierPort double: returns a fixed bundle, records every request, or raises."""

    def __init__(self, *, bundle: VerdictBundle = _ACCEPT, raises: Exception | None = None) -> None:
        self.identity = "fake-verifier"
        self.bundle = bundle
        self.raises = raises
        self.requests: list[VerifierRequest] = []
        self.provisioned = False

    async def dispatch(self, request: VerifierRequest) -> VerdictBundle:
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        return self.bundle

    async def provision(self) -> None:
        self.provisioned = True

    async def teardown(self) -> None:
        self.provisioned = False

    async def health(self) -> bool:
        return True


class _ScriptedSandbox:
    """A SandboxPort double: emits a fixed envelope (or raises), records serve/regenerate."""

    def __init__(
        self, *, envelope: ProposalEnvelope | None = None, raises: Exception | None = None
    ) -> None:
        self.envelope = envelope
        self.raises = raises
        self.served: list[ServedContext] = []
        self.regenerated = 0

    async def serve_context(self, context: ServedContext) -> None:
        self.served.append(context)

    async def collect_proposal(self) -> ProposalEnvelope:
        if self.raises is not None:
            raise self.raises
        assert self.envelope is not None
        return self.envelope

    async def regenerate(self) -> None:
        self.regenerated += 1

    async def provision(self) -> None: ...
    async def teardown(self) -> None: ...
    async def health(self) -> bool:
        return True


def _remote_verifier(impl: object) -> RemoteVerifier:
    return RemoteVerifier(LoopbackTransport(VerifierServer(impl).handle))  # type: ignore[arg-type]


def _remote_sandbox(impl: object) -> RemoteSandbox:
    return RemoteSandbox(LoopbackTransport(SandboxServer(impl).handle))  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- verifier round-trips


def test_remote_verifier_dispatch_roundtrips_with_object_bytes() -> None:
    impl = _RecordingVerifier()
    remote = _remote_verifier(impl)
    request = VerifierRequest(
        proposal=_ARTIFACT,
        store_slice=(_ARTIFACT,),
        objects={"submission.py": b"print('hi')", "data.bin": bytes(range(256))},
    )

    bundle = asyncio.run(remote.dispatch(request))

    assert bundle == _ACCEPT  # the verdict survived the hop
    assert impl.requests[0] == request  # the request arrived byte-identical (object bytes included)


def test_remote_verifier_provision_resolves_identity_over_the_wire() -> None:
    impl = _RecordingVerifier()
    remote = _remote_verifier(impl)
    assert remote.identity == ""  # unknown until provisioned (it is learned async)

    asyncio.run(remote.provision())

    assert remote.identity == "fake-verifier" and impl.provisioned is True
    assert asyncio.run(remote.health()) is True


# --------------------------------------------- data-bearing verifier: setup→build→dispatch (§9.1)


def test_setup_ships_data_and_builds_the_verifier_before_dispatch() -> None:
    built: list[VerifierSetup] = []

    def builder(setup: VerifierSetup) -> _RecordingVerifier:
        built.append(setup)
        return _RecordingVerifier()

    server = VerifierServer(builder=builder)
    setup = VerifierSetup(
        objects={"train.csv": b"a,b\n1,2\n", "labels.bin": bytes(range(256))},
        params={"competition": "demo", "reserved_labels": {"7": "x"}},
    )
    remote = RemoteVerifier(LoopbackTransport(server.handle), setup_payload=setup)

    asyncio.run(remote.provision())  # ships `setup` first, then `provision`
    bundle = asyncio.run(remote.dispatch(VerifierRequest(proposal=_ARTIFACT)))

    assert bundle == _ACCEPT
    assert remote.identity == "fake-verifier"  # learned over the wire, post-build
    assert len(built) == 1  # the server built exactly one impl from the payload
    assert built[0].objects["train.csv"] == b"a,b\n1,2\n"  # data crossed intact (object bytes)
    assert built[0].objects["labels.bin"] == bytes(range(256))
    assert built[0].params == {"competition": "demo", "reserved_labels": {"7": "x"}}


def test_dispatch_before_setup_degrades_to_gate_unavailable() -> None:
    # A builder-backed server has no impl until `setup`; dispatching first must fail informatively
    # (not an AttributeError). A reachable-but-misbehaving verifier is a *recoverable* boundary
    # failure (GateUnavailable) so the run degrades this cycle rather than aborting (S2) — the
    # informative message is preserved, and a persistent failure is bounded by the failure cap.
    server = VerifierServer(builder=lambda _setup: _RecordingVerifier())
    remote = RemoteVerifier(LoopbackTransport(server.handle))
    with pytest.raises(GateUnavailable, match="not set up yet"):
        asyncio.run(remote.dispatch(VerifierRequest(proposal=_ARTIFACT)))


def test_builder_server_rejects_being_constructed_empty() -> None:
    with pytest.raises(ValueError, match="impl or a builder"):
        VerifierServer()


# ----------------------------------------------------------------------------- sandbox round-trips


def test_remote_sandbox_serve_and_collect_roundtrip_with_objects() -> None:
    envelope = ProposalEnvelope(
        artifact=_ARTIFACT, operation=_OP, metadata="how/why",
        objects={"out.txt": b"result"},
        agent_telemetry={"model_steps": 3},
    )
    impl = _ScriptedSandbox(envelope=envelope)
    remote = _remote_sandbox(impl)
    context = ServedContext(
        system_prompt="SYS", tail="goal",
        workspace_objects={"scratch": {"prior.py": b"print(1)"}},
    )

    asyncio.run(remote.serve_context(context))
    collected = asyncio.run(remote.collect_proposal())
    asyncio.run(remote.regenerate())

    assert impl.served[0] == context  # workspace_objects bytes survived
    assert collected == envelope  # artifact + metadata + objects + telemetry all intact
    assert impl.regenerated == 1


# ----------------------------------------------------------------------------- error classification


def test_gate_unavailable_crosses_the_wire_as_itself() -> None:
    remote = _remote_verifier(_RecordingVerifier(raises=GateUnavailable("backend unreachable")))
    with pytest.raises(GateUnavailable, match="backend unreachable"):
        asyncio.run(remote.dispatch(VerifierRequest(proposal=_ARTIFACT)))


def test_verifier_error_stays_fatal_across_the_wire() -> None:
    remote = _remote_verifier(_RecordingVerifier(raises=VerifierError("no plugin for gate")))
    with pytest.raises(VerifierError, match="no plugin"):
        asyncio.run(remote.dispatch(VerifierRequest(proposal=_ARTIFACT)))


def test_sandbox_error_crosses_the_wire_as_itself() -> None:
    remote = _remote_sandbox(_ScriptedSandbox(raises=SandboxError("no descriptor in outbox")))
    with pytest.raises(SandboxError, match="no descriptor"):
        asyncio.run(remote.collect_proposal())


def test_unknown_error_degrades_to_gate_unavailable() -> None:
    # An error kind the envelope doesn't know about must not vanish. At the transport layer it
    # surfaces as TransportError; the RemoteVerifier then maps that to GateUnavailable so a
    # reachable-but-misbehaving verifier degrades the cycle instead of aborting the run (S2). The
    # original message is preserved end to end.
    remote = _remote_verifier(_RecordingVerifier(raises=ValueError("something odd")))
    with pytest.raises(GateUnavailable, match="something odd"):
        asyncio.run(remote.dispatch(VerifierRequest(proposal=_ARTIFACT)))


def test_unknown_method_is_a_transport_error() -> None:
    server = VerifierServer(_RecordingVerifier())
    transport: Transport = LoopbackTransport(server.handle)

    async def go() -> bytes:
        return await transport.request("nonsense", b"")

    from verity.transport.envelope import parse_result

    with pytest.raises(TransportError, match="unknown verifier method"):
        parse_result(asyncio.run(go()))


# --------------------------------------------------------- CP degrade-don't-crash over the wire


def test_control_plane_degrades_over_a_remote_verifier() -> None:
    # The load-bearing property: a remote verifier that raises GateUnavailable is recorded as a
    # failed cycle and the run continues — NO ControlPlane change (it sees only the async port).
    from tests.test_gate_resilience import _build, _FlakyVerifier, _note
    from verity.control_plane.api import OrchestrationPolicy

    remote = _remote_verifier(_FlakyVerifier())
    cp, _sandbox, store = _build(
        items=[_note("n1", fail_gate=True), _note("n2")],
        policy=OrchestrationPolicy(max_cycles=5, stop_on_accept=True),
        verifier=remote,
    )
    results = asyncio.run(cp.run("t1", goal="author a note"))

    assert results[0].gate_error is not None  # the remote GateUnavailable survived the hop
    assert results[0].sandbox_error is None
    assert results[1].commit is not None and results[1].commit.status is ArtifactStatus.ACCEPTED
    assert store.get_artifact("n2") is not None  # cycle 2 committed despite cycle 1's failure
