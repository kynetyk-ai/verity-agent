"""An in-memory `WorkerBackend` test double (ROADMAP 7.4 / ADR 0003) — NON-PRODUCT.

The implementor side of the contract, shipped with it (the analog of ``FakeCodeRunner``): it runs no
Docker and no subprocess. It **records every `WorkerSpec` it was launched with** and returns a
**scripted** result, so a unit test can assert *"given this cycle, the worker got these labels /
read-only inputs / network"* — and the later backend-backed driver and code-runner reuse it to prove
the spec they build, without a daemon. ``list``/``reap`` track live workers by their labels.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from verity.provisioning.backend import (
    CompletedWorker,
    WorkerHandle,
    WorkerSpec,
    WorkerStatus,
)

__all__ = ["FakeBackend", "WorkerScript"]

# A fake's scripted behaviour: the spec it was launched with -> the result.
WorkerScript = Callable[[WorkerSpec], CompletedWorker]


def _clean(_spec: WorkerSpec) -> CompletedWorker:
    """The default: a clean exit producing no outputs (a test scripts outputs when it cares)."""
    return CompletedWorker(exit_code=0, stdout="", stderr="")


class FakeBackend:
    """A deterministic in-memory `WorkerBackend`. ``script`` maps a spec to its result; ``launched``
    records every spec for assertions."""

    def __init__(self, script: WorkerScript | None = None) -> None:
        self.script: WorkerScript = script or _clean
        self.launched: list[WorkerSpec] = []
        self._live: dict[str, WorkerSpec] = {}
        self._counter = 0

    async def run_to_completion(self, spec: WorkerSpec) -> CompletedWorker:
        handle = await self.launch(spec)
        try:
            return await self.wait(handle, timeout_s=spec.timeout_s)
        finally:
            await self.destroy(handle)

    async def launch(self, spec: WorkerSpec) -> WorkerHandle:
        self.launched.append(spec)
        self._counter += 1
        handle = WorkerHandle(id=f"fake-{self._counter}", labels=spec.labels)
        self._live[handle.id] = spec
        return handle

    async def wait(self, handle: WorkerHandle, *, timeout_s: float) -> CompletedWorker:
        spec = self._live.get(handle.id)
        if spec is None:
            return CompletedWorker(exit_code=-1, stdout="", stderr="no such worker")
        return self.script(spec)

    async def status(self, handle: WorkerHandle) -> WorkerStatus:
        return WorkerStatus.RUNNING if handle.id in self._live else WorkerStatus.GONE

    async def logs(self, handle: WorkerHandle) -> bytes:
        return b""

    async def destroy(self, handle: WorkerHandle) -> None:
        self._live.pop(handle.id, None)

    async def list(self, selector: Mapping[str, str]) -> list[WorkerHandle]:
        return [
            WorkerHandle(id=wid, labels=spec.labels)
            for wid, spec in self._live.items()
            if spec.labels.matches(selector)
        ]

    async def reap(self, selector: Mapping[str, str]) -> int:
        matching = [wid for wid, spec in self._live.items() if spec.labels.matches(selector)]
        for wid in matching:
            del self._live[wid]
        return len(matching)
