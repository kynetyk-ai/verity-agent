"""The thin client over the daemon's Unix socket (ROADMAP 8.2, ADR 0004 (d)).

An httpx `AsyncClient` bound to the daemon's Unix-domain socket (``uds=``) — one coroutine per verb,
returning parsed JSON and raising :class:`ControlServiceError` on a non-2xx reply. It holds no
`ControlPlane`: a `docker exec … verity …` spawns a *fresh* process, so it must reach the standing
daemon's in-memory state (configured tasks, the run lock, in-flight runs) over the socket, not build
its own. The `verity` CLI is a thin layer over this.

Imports httpx — the ``service`` extra; the package ``__init__`` does not import this module.
"""

from __future__ import annotations

from typing import Any

import httpx

__all__ = ["Client", "ControlServiceError"]


class ControlServiceError(RuntimeError):
    """A non-2xx reply from the daemon — carries the HTTP status and the server's detail."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


class Client:
    """A coroutine-per-verb client over the daemon's Unix socket."""

    def __init__(
        self, socket_path: str, *, base_url: str = "http://verity", timeout_s: float = 120.0
    ) -> None:
        self._socket_path = socket_path
        self._base_url = base_url
        self._timeout_s = timeout_s

    def _open(self) -> httpx.AsyncClient:
        # A fresh client per call (the CLI is one-shot); the transport binds to the daemon's socket.
        return httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=self._socket_path),
            base_url=self._base_url,
            timeout=self._timeout_s,
        )

    async def _call(self, method: str, path: str, *, json: Any | None = None) -> Any:
        try:
            async with self._open() as client:
                response = await client.request(method, path, json=json)
        except httpx.HTTPError as exc:  # daemon down / socket missing — a clean, typed failure
            raise ControlServiceError(
                503, f"control-plane daemon unreachable at {self._socket_path} ({exc})"
            ) from exc
        if response.status_code >= 400:
            detail = response.text
            try:
                detail = response.json().get("detail", detail)
            except (ValueError, AttributeError):
                pass
            raise ControlServiceError(response.status_code, detail)
        return response.json()

    async def catalog(self, type_name: str | None = None) -> Any:
        return await self._call("GET", f"/catalog/{type_name}" if type_name else "/catalog")

    async def ingest(self, name: str) -> Any:
        return await self._call("POST", "/objects", json={"name": name})

    async def create_task(self, request: dict[str, Any], *, data: str | None = None) -> Any:
        return await self._call("POST", "/tasks", json={"request": request, "data": data})

    async def list_tasks(self) -> Any:
        return await self._call("GET", "/tasks")

    async def run(self, task_id: str, *, goal: str | None = None) -> Any:
        return await self._call("POST", f"/tasks/{task_id}/runs", json={"goal": goal})

    async def status(self, run_id: str) -> Any:
        return await self._call("GET", f"/runs/{run_id}")

    async def results(self, run_id: str) -> Any:
        return await self._call("GET", f"/runs/{run_id}/results")

    async def export(self, run_id: str) -> Any:
        return await self._call("POST", f"/runs/{run_id}/export")
