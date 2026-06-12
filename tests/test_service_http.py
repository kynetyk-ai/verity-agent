"""The HTTP control surface end-to-end (ROADMAP 8.2, ADR 0004 (d)): the daemon's verbs over HTTP.

Drives the FastAPI app (the surface the daemon serves over a Unix socket, and 8.4 rebinds to TCP)
through the full flow — ingest → create → run → poll status → results → export — once in-process via
httpx `ASGITransport` (deterministic), and once over a **real Unix socket** served by uvicorn (the
`uds=` binding + the `Client`). Needs the `service` extra; auto-skips without it.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx

from tests._fe_offline import FeWorkers
from verity.composition.dataset import stratified_split, subsample
from verity.composition.task_request import DataRequest, PolicyRequest, TaskRequest
from verity.provisioning import FakeBackend
from verity.service.control_service import ControlService
from verity.service.http import build_app

_RAW = b"id,a,class\n0,0,X\n1,2,X\n2,4,X\n3,6,Y\n4,8,Y\n5,10,Y\n"


def _reserved() -> dict[str, str]:
    return stratified_split(
        subsample(_RAW, per_class=100, target="class"),
        target="class", id_column="id", reserved_fraction=0.5,
    ).reserved_labels


def _fe_request_body() -> dict:
    return TaskRequest(
        type_name="fe", goal="improve balanced accuracy",
        policy=PolicyRequest(stop_on_accept=True),
        data=DataRequest(per_class=100, reserved_fraction=0.5),
    ).to_dict()


def _build(tmp_path):
    exchange_in, exchange_out = tmp_path / "in", tmp_path / "out"
    exchange_in.mkdir()
    exchange_out.mkdir()
    (exchange_in / "train.csv").write_bytes(_RAW)
    service = ControlService(
        backend=FakeBackend(script=FeWorkers(steps=[("submit", ("ds",), b"good", ["feat0"])],
                                             by_marker={b"good": _reserved()})),
        root=None,
    )
    app = build_app(service, exchange_in=exchange_in, exchange_out=exchange_out)
    return app, exchange_out, service


async def _poll_terminal(client: httpx.AsyncClient, run_id: str) -> str:
    for _ in range(200):
        status = (await client.get(f"/runs/{run_id}")).json()["status"]
        if status in ("done", "failed"):
            return status
        await asyncio.sleep(0.01)
    raise AssertionError("run did not terminate")


def test_http_full_flow_in_process(tmp_path) -> None:
    async def scenario() -> None:
        app, exchange_out, _ = _build(tmp_path)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://verity") as client:
            # catalog
            cat = (await client.get("/catalog")).json()
            assert {t["type_name"] for t in cat["task_types"]} == {"fe", "fe-kaggle", "code"}

            # ingest -> create -> run
            handle = (await client.post("/objects", json={"name": "train.csv"})).json()["handle"]
            task_id = (
                await client.post("/tasks", json={"request": _fe_request_body(), "data": handle})
            ).json()["task_id"]
            run_id = (await client.post(f"/tasks/{task_id}/runs", json={})).json()["run_id"]

            assert await _poll_terminal(client, run_id) == "done"

            # results: an accepted Submission landed
            results = (await client.get(f"/runs/{run_id}/results")).json()
            assert results["status"] == "complete"
            assert results["accepted_artifact_ids"]

            # export: the durable artifact bytes (the submitted code) reach the exchange out-dir
            exported = (await client.post(f"/runs/{run_id}/export")).json()
            assert "results.json" in exported["exported"]
            assert (exchange_out / run_id / "results.json").is_file()
            code_files = list((exchange_out / run_id).rglob("submission.py"))
            assert code_files, "the accepted submission's code was not exported"

            # an unknown run is a clean 404
            assert (await client.get("/runs/nope/results")).status_code == 404

    asyncio.run(scenario())


def test_http_over_real_unix_socket(tmp_path) -> None:
    """The `uds=` binding + the `Client`: serve the app on a real socket, hit /catalog."""
    import uvicorn

    from verity.service.client import Client

    async def scenario() -> None:
        app, _, _ = _build(tmp_path)
        # A short /tmp path: macOS caps AF_UNIX socket paths at ~104 chars (pytest's tmp_path is too
        # long). Clean it up in finally.
        sock = Path("/tmp") / f"vty-{uuid.uuid4().hex[:8]}.sock"
        config = uvicorn.Config(app, uds=str(sock), log_level="error", lifespan="off")
        server = uvicorn.Server(config)
        serve_task = asyncio.create_task(server.serve())
        try:
            for _ in range(200):  # wait for the bind
                if server.started and sock.exists():
                    break
                await asyncio.sleep(0.01)
            assert server.started, "uvicorn did not start"

            client = Client(str(sock))
            cat = await client.catalog()
            assert any(t["type_name"] == "fe" for t in cat["task_types"])
        finally:
            server.should_exit = True
            await serve_task
            sock.unlink(missing_ok=True)

    asyncio.run(scenario())
