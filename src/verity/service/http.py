"""The HTTP control surface over the `ControlService` core (ROADMAP 8.2, ADR 0004 (d)/(e)).

A FastAPI app whose REST routes mirror the CLI verbs — and are exactly the external surface 8.4
rebinds to a TCP port + auth. In 8.2 the daemon serves this app over a **Unix-domain socket**
(uvicorn ``uds=``), so it is local/`docker exec`-only and needs no auth (the v1 trust boundary). The
app holds no `ControlPlane` of its own; it only calls the injected `ControlService`. File I/O moves
through an **exchange directory** (not the socket): ``ingest`` reads ``<exchange_in>/<name>``,
``export`` writes durable artifact bytes under ``<exchange_out>/<run_id>/`` — never into a worker.

Imports FastAPI — installed via the ``service`` extra; the package ``__init__`` does **not** import
this module, so the core/loopback path never requires the extra (the lazy-import discipline).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from verity.composition import TaskRequest, UnknownTaskType
from verity.contracts.errors import GateUnavailable
from verity.logging import get_logger
from verity.service.control_service import ControlService, UnknownTask

__all__ = ["build_app"]

log = get_logger("verity.service.http")

# Default cap on an over-the-wire object upload (bytes held in memory until create_task; the ADR
# 0004 (f) pulled-forward byte data plane — bounded, no streaming).
_DEFAULT_MAX_UPLOAD = 50_000_000


def _summary(record: Any) -> dict[str, Any] | None:
    return record.report.summary.to_dict() if record is not None else None


def build_app(
    service: ControlService,
    *,
    exchange_in: Path,
    exchange_out: Path,
    auth_token: str | None = None,
    max_upload_bytes: int = _DEFAULT_MAX_UPLOAD,
) -> FastAPI:
    """A FastAPI app over ``service``; ingress from ``exchange_in``, egress to ``exchange_out``.

    When ``auth_token`` is set (the network/TCP adapter, 8.4), every route except ``/health``
    requires ``Authorization: Bearer <auth_token>`` (401 otherwise) and the authenticated principal
    is stashed on ``request.state`` — the ADR 0004 Q3 identity hook. When it is ``None`` (the
    daemon's Unix-socket binding, 8.2/8.3), no auth is applied: the socket is the trust boundary.
    """

    def _require_bearer(request: Request) -> None:
        if request.url.path == "/health":  # readiness probe stays open
            return
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        assert auth_token is not None  # only registered when a token is configured
        if scheme.lower() != "bearer" or not hmac.compare_digest(token, auth_token):
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")
        request.state.principal = "token"  # identity hook (single shared principal in v1)

    dependencies = [Depends(_require_bearer)] if auth_token else []
    app = FastAPI(title="verity-control-plane", dependencies=dependencies)
    staging: dict[str, bytes] = {}  # ingest handle (content hash) -> bytes, until create_task

    def _error(status_code: int) -> Any:
        def handler(_: Request, exc: Exception) -> JSONResponse:
            return JSONResponse(status_code=status_code, content={"detail": str(exc)})
        return handler

    # Map the domain/service errors to advisory HTTP statuses (the body always carries the detail).
    app.add_exception_handler(UnknownTask, _error(404))
    app.add_exception_handler(UnknownTaskType, _error(404))
    app.add_exception_handler(ValueError, _error(422))
    app.add_exception_handler(GateUnavailable, _error(503))

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/catalog")
    async def catalog() -> dict[str, Any]:
        return {"task_types": service.catalog_describe()}

    @app.get("/catalog/{type_name}")
    async def catalog_one(type_name: str) -> dict[str, Any]:
        return {"task_types": service.catalog_describe(type_name)}

    @app.post("/objects")
    async def ingest(request: Request) -> dict[str, str]:
        """Ingest an object → a content-addressed handle (resolved by ``create_task``).

        Two forms: a JSON ``{"name": …}`` body reads a file from the server-side **exchange**
        (the local/CLI path), while any other body is treated as **raw bytes** uploaded
        over the wire (the network client with no shared volume — ADR 0004 (f), bounded).
        """
        raw = await request.body()
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                body = json.loads(raw) if raw else {}
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="invalid JSON body") from exc
            name = body.get("name")
            if not isinstance(name, str):
                raise HTTPException(status_code=422, detail="'name' (an exchange file) is required")
            # Subdirs are allowed (role bundles like `agent/train.csv`), but the resolved path must
            # stay inside the exchange — no `..`/absolute traversal out of it.
            if "\\" in name or ".." in name.split("/") or name.startswith("/"):
                raise HTTPException(status_code=422, detail=f"invalid exchange filename: {name}")
            path = exchange_in / name
            if not path.is_file():
                raise HTTPException(status_code=404, detail=f"no such file in the exchange: {name}")
            data = path.read_bytes()
        else:
            if not raw:
                raise HTTPException(status_code=422, detail="empty object body")
            if len(raw) > max_upload_bytes:
                raise HTTPException(
                    status_code=413, detail=f"object exceeds the {max_upload_bytes}-byte limit"
                )
            data = raw
        handle = hashlib.sha256(data).hexdigest()
        staging[handle] = data
        log.info("ingested", handle=handle, bytes=len(data))
        return {"handle": handle}

    @app.post("/tasks")
    async def create_task(request: Request) -> dict[str, str]:
        body = await request.json()
        spec = body.get("request")
        if not isinstance(spec, dict):
            raise HTTPException(status_code=422, detail="'request' (a TaskRequest) is required")
        task_request = TaskRequest.from_dict(spec)

        # `files` maps role -> {filename: ingest-handle}; resolve each handle to its staged bytes.
        # The files are opaque to the service — it routes a role's set into that role's worker.
        raw_files = body.get("files") or {}
        if not isinstance(raw_files, dict):
            raise HTTPException(status_code=422, detail="'files' must be {role:{name:handle}}")
        files: dict[str, dict[str, bytes]] = {}
        for role, named in raw_files.items():
            if not isinstance(named, dict):
                raise HTTPException(status_code=422, detail=f"files[{role!r}] must be a name->handle map")  # noqa: E501
            files[role] = {}
            for name, handle in named.items():
                if handle not in staging:
                    raise HTTPException(status_code=404, detail=f"unknown data handle: {handle}")
                files[role][name] = staging[handle]

        task_id = service.create_task(task_request, files=files)
        return {"task_id": task_id}

    @app.get("/tasks")
    async def list_tasks() -> dict[str, list[str]]:
        return {"tasks": service.list_tasks()}

    @app.post("/tasks/{task_id}/runs")
    async def start_run(task_id: str, request: Request) -> dict[str, str]:
        body = await request.json() if await request.body() else {}
        run_id = await service.submit_run(task_id, goal=body.get("goal"))
        return {"run_id": run_id}

    @app.get("/runs/{run_id}")
    async def run_status(run_id: str) -> dict[str, Any]:
        status = await service.status(run_id)  # raises UnknownTask -> 404
        return {
            "run_id": run_id,
            "status": status.value,
            "summary": _summary(service.results(run_id)),
        }

    @app.get("/runs/{run_id}/results")
    async def run_results(run_id: str) -> dict[str, Any]:
        record = service.results(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no results for run: {run_id}")
        return record.to_dict()

    @app.post("/runs/{run_id}/export")
    async def export(run_id: str) -> dict[str, Any]:
        record = service.results(run_id)
        task_id = service.task_for_run(run_id)
        if record is None or task_id is None:
            raise HTTPException(status_code=404, detail=f"no results for run: {run_id}")
        out_dir = exchange_out / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(
            json.dumps(record.to_dict(), indent=2), encoding="utf-8"
        )
        exported = ["results.json"]
        by_id = {a.id: a for a in await service.accepted_artifacts(task_id)}
        for artifact_id in record.accepted_artifact_ids:
            artifact = by_id.get(artifact_id)
            if artifact is None:
                continue
            for name, ref in artifact.objects:
                dest = out_dir / artifact_id / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(service.get_object(task_id, ref.content_hash))
                exported.append(f"{artifact_id}/{name}")
        log.info("exported", run_id=run_id, count=len(exported))
        return {"run_id": run_id, "out_dir": str(out_dir), "exported": exported}

    @app.get("/artifacts/{run_id}/{object_path:path}")
    async def artifact(run_id: str, object_path: str) -> Response:
        """Egress over the wire (no shared volume): a single durable artifact object's bytes.

        ``object_path`` matches either ``<artifact_id>/<name>`` (as ``export`` lays it out) or a
        bare ``<name>`` (first match) — the same resolution ``export`` does, for one object.
        """
        record = service.results(run_id)
        task_id = service.task_for_run(run_id)
        if record is None or task_id is None:
            raise HTTPException(status_code=404, detail=f"no results for run: {run_id}")
        by_id = {a.id: a for a in await service.accepted_artifacts(task_id)}
        for artifact_id in record.accepted_artifact_ids:
            art = by_id.get(artifact_id)
            if art is None:
                continue
            for name, ref in art.objects:
                if object_path in (f"{artifact_id}/{name}", name):
                    data = service.get_object(task_id, ref.content_hash)
                    return Response(content=data, media_type="application/octet-stream")
        raise HTTPException(status_code=404, detail=f"no artifact object: {run_id}/{object_path}")

    return app
