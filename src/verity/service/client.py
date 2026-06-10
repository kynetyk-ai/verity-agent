"""The thin client over the daemon (ROADMAP 8.2 Unix socket, 8.4 network TCP; ADR 0004 (d)/(e)).

An httpx `AsyncClient` bound to **either** the daemon's Unix-domain socket (``uds=``, the local
`docker exec` path) **or** a network base URL (``http://host:port``, the 8.4 external API), with an
optional **bearer token** for the authenticated TCP surface. One coroutine per verb, returning
parsed JSON (or raw bytes for a download) and raising :class:`ControlServiceError` on a non-2xx
reply. It holds no `ControlPlane`: a fresh client process reaches the standing daemon's in-memory
state (configured tasks, the run lock, in-flight runs) over the wire, not by building its own. The
`verity` CLI is a thin layer over this.

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
    """A coroutine-per-verb client over the daemon's Unix socket or a network URL."""

    def __init__(
        self,
        socket_path: str | None = None,
        *,
        base_url: str | None = None,
        token: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        if (socket_path is None) == (base_url is None):
            raise ValueError("provide exactly one of socket_path or base_url")
        self._socket_path = socket_path
        # httpx needs a base_url even for the uds transport; the host part is cosmetic there.
        self._base_url = base_url or "http://verity"
        self._token = token
        self._timeout_s = timeout_s

    @property
    def _target(self) -> str:
        return self._socket_path or self._base_url

    def _open(self) -> httpx.AsyncClient:
        # A fresh client per call (the CLI is one-shot). UDS binds a transport; TCP uses default.
        transport = (
            httpx.AsyncHTTPTransport(uds=self._socket_path)
            if self._socket_path is not None
            else None
        )
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else None
        return httpx.AsyncClient(
            transport=transport, base_url=self._base_url, headers=headers, timeout=self._timeout_s
        )

    async def _request(
        self, method: str, path: str, *, json: Any | None = None, content: bytes | None = None
    ) -> httpx.Response:
        kwargs: dict[str, Any] = {}
        if json is not None:
            kwargs["json"] = json
        if content is not None:
            kwargs["content"] = content
            kwargs["headers"] = {"Content-Type": "application/octet-stream"}
        try:
            async with self._open() as client:
                response = await client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:  # daemon down / socket missing — a clean, typed failure
            raise ControlServiceError(
                503, f"control-plane daemon unreachable at {self._target} ({exc})"
            ) from exc
        if response.status_code >= 400:
            detail = response.text
            try:
                detail = response.json().get("detail", detail)
            except (ValueError, AttributeError):
                pass
            raise ControlServiceError(response.status_code, detail)
        return response

    async def _call(self, method: str, path: str, *, json: Any | None = None) -> Any:
        return (await self._request(method, path, json=json)).json()

    async def catalog(self, type_name: str | None = None) -> Any:
        return await self._call("GET", f"/catalog/{type_name}" if type_name else "/catalog")

    async def ingest(self, name: str) -> Any:
        return await self._call("POST", "/objects", json={"name": name})

    async def ingest_bytes(self, data: bytes) -> Any:
        """Upload raw object bytes over the wire (no shared exchange volume) → a data handle."""
        return (await self._request("POST", "/objects", content=data)).json()

    async def download(self, run_id: str, object_path: str) -> bytes:
        """Fetch a single durable artifact object's bytes over the wire."""
        return (await self._request("GET", f"/artifacts/{run_id}/{object_path}")).content

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
