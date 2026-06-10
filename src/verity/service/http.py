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
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from verity.composition import TaskRequest, UnknownTaskType
from verity.contracts.errors import GateUnavailable
from verity.logging import get_logger
from verity.service.control_service import ControlService, UnknownTask

__all__ = ["build_app"]

log = get_logger("verity.service.http")


def _summary(record: Any) -> dict[str, Any] | None:
    return record.report.summary.to_dict() if record is not None else None


def build_app(service: ControlService, *, exchange_in: Path, exchange_out: Path) -> FastAPI:
    """A FastAPI app over ``service``; ingress from ``exchange_in``, egress to ``exchange_out``."""
    app = FastAPI(title="verity-control-plane")
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

    @app.get("/catalog")
    async def catalog() -> dict[str, Any]:
        return {"task_types": service.catalog_describe()}

    @app.get("/catalog/{type_name}")
    async def catalog_one(type_name: str) -> dict[str, Any]:
        return {"task_types": service.catalog_describe(type_name)}

    @app.post("/objects")
    async def ingest(request: Request) -> dict[str, str]:
        body = await request.json()
        name = body.get("name")
        if not isinstance(name, str):
            raise HTTPException(status_code=422, detail="'name' (an exchange file) is required")
        path = exchange_in / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"no such file in the exchange: {name}")
        data = path.read_bytes()
        handle = hashlib.sha256(data).hexdigest()
        staging[handle] = data
        log.info("ingested", name=name, handle=handle, bytes=len(data))
        return {"handle": handle}

    @app.post("/tasks")
    async def create_task(request: Request) -> dict[str, str]:
        body = await request.json()
        spec = body.get("request")
        if not isinstance(spec, dict):
            raise HTTPException(status_code=422, detail="'request' (a TaskRequest) is required")
        task_request = TaskRequest.from_dict(spec)
        data: bytes | None = None
        handle = body.get("data")
        if handle is not None:
            if handle not in staging:
                raise HTTPException(status_code=404, detail=f"unknown data handle: {handle}")
            data = staging[handle]
        task_id = service.create_task(task_request, data=data)
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

    return app
