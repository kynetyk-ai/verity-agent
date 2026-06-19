"""Shared offline-FE test substrate (ROADMAP 8.1): a config-driven `FakeBackend` worker script.

Not a test module (no ``test_`` prefix). It provides a `FeWorkers` callable that plays *both* FE
worker roles over a `WorkerBackend` — a ``sandbox`` worker emits a scripted proposal descriptor, a
``code-runner`` worker "runs" the submitted script and returns predictions keyed off a marker — so a
single `FakeBackend` drives the whole read→propose→gate→commit loop with no Docker and no model.
This mirrors the capstone in ``tests/test_fe_via_config_acceptance.py`` but is reusable across the
8.1 tests (byte-provenance, per-task isolation, the control service).
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.domains.feature_engineering import ENTRYPOINT, PREDICTIONS_OUTPUT, REQUIREMENTS
from verity.provisioning.backend import CompletedWorker, WorkerSpec
from verity.sandbox import ProposalDescriptor
from verity.sandbox.container_io import CONTAINER_OUTBOX
from verity.sandbox.descriptor import RESERVED_PROPOSAL_NAME
from verity.verifier.kaggle import KaggleScorer


def preds_csv(preds: dict[str, str]) -> bytes:
    return ("id,class\n" + "\n".join(f"{k},{v}" for k, v in preds.items()) + "\n").encode()


def entrypoint_bytes(marker: bytes, names: list[str]) -> bytes:
    """A submission script carrying a marker (the fake runner keys predictions off it). ``names``
    embed as a comment for readability only — the domain no longer gates on declared features."""
    return marker + b"\n# defines: " + " ".join(names).encode()


@dataclass
class FeWorkers:
    """Dispatches on the worker's ``role`` label: ``sandbox`` proposes, ``code-runner`` scores.

    ``steps`` is a list of ``(op_name, parents, marker, feature_names)`` proposals, emitted one per
    sandbox cycle; ``by_marker`` maps a submission's marker to the predictions the runner returns
    for it (so a script that predicts the reserved labels exactly scores 1.0 → accepted).
    """

    steps: list[tuple[str, tuple[str, ...], bytes, list[str]]]
    by_marker: dict[bytes, dict[str, str]]
    cursor: int = 0

    def __call__(self, spec: WorkerSpec) -> CompletedWorker:
        role = spec.labels.role
        if role == "sandbox":
            return self._propose()
        if role == "code-runner":
            return self._run_code(spec)
        return CompletedWorker(exit_code=1, stdout="", stderr=f"unexpected worker role {role!r}")

    def _propose(self) -> CompletedWorker:
        if self.cursor >= len(self.steps):  # a surplus cycle proposes nothing
            return CompletedWorker(exit_code=0, stdout="", stderr="")
        op_name, parents, marker, names = self.steps[self.cursor]
        self.cursor += 1
        payload = {"entrypoint": ENTRYPOINT, "requirements": REQUIREMENTS}
        descriptor = ProposalDescriptor(op_name, parents, payload, metadata="i reasoned thus")
        return CompletedWorker(
            exit_code=0, stdout="", stderr="",
            outputs={
                f"{CONTAINER_OUTBOX}/{ENTRYPOINT}": entrypoint_bytes(marker, names),
                f"{CONTAINER_OUTBOX}/{REQUIREMENTS}": b"pandas==2.2.2",
                f"{CONTAINER_OUTBOX}/{RESERVED_PROPOSAL_NAME}": descriptor.to_json(),
            },
        )

    def _run_code(self, spec: WorkerSpec) -> CompletedWorker:
        code = spec.readonly_inputs.get(f"/work/{ENTRYPOINT}", b"")
        for marker, preds in self.by_marker.items():
            if marker in code:
                return CompletedWorker(
                    exit_code=0, stdout="", stderr="",
                    outputs={f"/out/{PREDICTIONS_OUTPUT}": preds_csv(preds)},
                )
        return CompletedWorker(exit_code=0, stdout="", stderr="")  # no scorable output


def loopback_fe_kaggle_factory(backend, scorer):  # type: ignore[no-untyped-def]
    """An offline ``make_verifier`` for `configure_fe_kaggle_task` (§9.1).

    Builds the **real** FE-Kaggle gate stack in-process over the given `FakeBackend` + scorer and
    wraps it in a loopback `RemoteVerifier`, so the offline suite drives the same
    setup→build→dispatch seam the production HTTP path does — with no Docker, model, or network. The
    builder receives the per-task `VerifierSetup` the control plane shipped, exactly as the verifier
    image's builder would over the wire.
    """
    from verity.contracts.ports import VerifierSetup
    from verity.domains.feature_engineering_kaggle import (
        build_feature_engineering_kaggle_verifier,
    )
    from verity.transport import LoopbackTransport, RemoteVerifier, VerifierServer
    from verity.verifier import BackendCodeRunner

    def _builder(setup: VerifierSetup):  # type: ignore[no-untyped-def]
        runner = BackendCodeRunner(backend=backend, config="fe-kaggle")
        return build_feature_engineering_kaggle_verifier(
            runner, scorer=scorer,
            agent_train_csv=setup.objects["agent_train"],
            reserved_test_csv=setup.objects["reserved_test"],
            reserved_labels=setup.params["reserved_labels"],
            full_train_csv=setup.objects["full_train"],
            real_test_csv=setup.objects["real_test"],
            timeout_s=float(setup.params.get("timeout_s", 900.0)),
            wait_deadline_s=float(setup.params.get("wait_deadline_s", 86_400.0)),
            poll_interval_s=float(setup.params.get("poll_interval_s", 60.0)),
            submit_message=str(setup.params.get("submit_message", "verity")),
        )

    async def make(setup: VerifierSetup):  # type: ignore[no-untyped-def]
        return RemoteVerifier(
            LoopbackTransport(VerifierServer(builder=_builder).handle), setup_payload=setup
        )

    return make


def offline_catalog(scorer: KaggleScorer | None = None):  # type: ignore[no-untyped-def]
    """A `TaskCatalog` for offline service/daemon tests: the real `fe-kaggle` task wired to a
    `FakeKaggleScorer` (so the loop runs with no Kaggle creds/network), plus the `code` task.

    The production catalog builder (`build_fe_kaggle_task`) constructs a `RealKaggleScorer`; this
    mirrors its data prep but injects the fake — the only difference needed to drive `fe-kaggle`
    end to end over a `FakeBackend`. ``scorer`` overrides the default `FakeKaggleScorer`.
    """
    from verity.composition.catalog import TaskCatalog
    from verity.composition.code import CODE_TASK_ID, build_code_task, describe_code_task
    from verity.composition.dataset import stratified_split, subsample
    from verity.composition.fe import provisioning_config_from
    from verity.composition.fe_kaggle import (
        FE_KAGGLE_TASK_ID,
        configure_fe_kaggle_task,
        describe_fe_kaggle_task,
    )
    from verity.verifier.kaggle import FakeKaggleScorer

    active_scorer: KaggleScorer = scorer if scorer is not None else FakeKaggleScorer()

    async def _build_offline_fe_kaggle(cp, *, backend, request):  # type: ignore[no-untyped-def]
        raw = cp.store.get_object(request.data.data_ref)
        real_test = cp.store.get_object(request.data.test_ref)
        sub = subsample(raw, per_class=request.data.per_class, target=request.data.target)
        split = stratified_split(
            sub, target=request.data.target, id_column=request.data.id_column,
            reserved_fraction=request.data.reserved_fraction,
        )
        return await configure_fe_kaggle_task(
            cp, backend=backend,
            make_verifier=loopback_fe_kaggle_factory(backend, active_scorer),
            split=split, full_train_csv=sub, real_test_csv=real_test,
            provisioning=provisioning_config_from(request.sandbox),
            poll_interval_s=0.0, wait_deadline_s=5.0,
        )

    catalog = TaskCatalog()
    catalog.register(FE_KAGGLE_TASK_ID, _build_offline_fe_kaggle, describe=describe_fe_kaggle_task)
    catalog.register(CODE_TASK_ID, build_code_task, describe=describe_code_task)
    return catalog
