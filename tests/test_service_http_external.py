"""The external HTTP/REST control surface (ROADMAP 8.4, ADR 0004 (e)): bearer auth + byte I/O.

8.4 hosts the *same* FastAPI `build_app` over a network port. Two things the Unix-socket binding did
not need, exercised here: a **bearer token** (every route but `/health` requires it) and an
**over-the-wire byte data plane** (raw-bytes ingest + artifact download, for a client with no shared
exchange volume). This drives both in-process via httpx `ASGITransport` (deterministic); the
real-TCP round trip over uvicorn + the `Client` is added alongside the entrypoint. Needs the
`service` extra; auto-skips without it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx
from tools.harness.dataset import stratified_split, subsample

from tests._fe_offline import (
    FeWorkers,
    entrypoint_bytes,
    offline_catalog,
    role_files_from_raw,
)
from verity.composition.task_request import DataRequest, PolicyRequest, TaskRequest
from verity.domains.feature_engineering import ENTRYPOINT
from verity.provisioning import FakeBackend
from verity.service.control_service import ControlService
from verity.service.http import build_app

_RAW = b"id,a,class\n0,0,X\n1,2,X\n2,4,X\n3,6,Y\n4,8,Y\n5,10,Y\n"
_REAL_TEST = b"id\n100\n101\n102\n103\n"
_TOKEN = "s3cret-token"  # noqa: S105 — a test fixture, not a real credential


def _reserved() -> dict[str, str]:
    return stratified_split(
        subsample(_RAW, per_class=100, target="class"),
        target="class", id_column="id", reserved_fraction=0.5,
    ).reserved_labels


def _fe_request_body() -> dict:
    return TaskRequest(
        type_name="fe-kaggle", goal="improve balanced accuracy",
        policy=PolicyRequest(stop_on_accept=True),
        data=DataRequest(),
    ).to_dict()


async def _upload_role_files(client: httpx.AsyncClient, headers: dict) -> dict[str, dict[str, str]]:
    """Upload each prepared role file as raw bytes over the wire → a {role: {name: handle}} map."""
    role_files = role_files_from_raw(_RAW, _REAL_TEST, per_class=100, reserved_fraction=0.5)
    files: dict[str, dict[str, str]] = {}
    for role, named in role_files.items():
        files[role] = {}
        for name, blob in named.items():
            resp = await client.post("/objects", content=blob, headers=headers)
            assert resp.status_code == 200
            files[role][name] = resp.json()["handle"]
    return files


def _build(tmp_path: Path, *, auth_token: str | None = None, max_upload_bytes: int = 50_000_000):
    exchange_in, exchange_out = tmp_path / "in", tmp_path / "out"
    exchange_in.mkdir()
    exchange_out.mkdir()
    service = ControlService(
        backend=FakeBackend(script=FeWorkers(steps=[("submit", ("ds",), b"good", ["feat0"])],
                                             by_marker={b"good": _reserved()})),
        root=None, catalog=offline_catalog(),
    )
    return build_app(
        service, exchange_in=exchange_in, exchange_out=exchange_out,
        auth_token=auth_token, max_upload_bytes=max_upload_bytes,
    )


async def _poll_terminal(client: httpx.AsyncClient, run_id: str, headers: dict) -> str:
    for _ in range(200):
        status = (await client.get(f"/runs/{run_id}", headers=headers)).json()["status"]
        if status in ("done", "failed"):
            return status
        await asyncio.sleep(0.01)
    raise AssertionError("run did not terminate")


def test_bearer_auth_gates_every_route_but_health(tmp_path) -> None:
    async def scenario() -> None:
        transport = httpx.ASGITransport(app=_build(tmp_path, auth_token=_TOKEN))
        async with httpx.AsyncClient(transport=transport, base_url="http://verity") as c:
            assert (await c.get("/health")).status_code == 200  # the probe is open
            assert (await c.get("/catalog")).status_code == 401  # no token -> refused
            wrong = await c.get("/catalog", headers={"Authorization": "Bearer nope"})
            assert wrong.status_code == 401
            ok = await c.get("/catalog", headers={"Authorization": f"Bearer {_TOKEN}"})
            assert ok.status_code == 200
            assert {t["type_name"] for t in ok.json()["task_types"]} == {"fe-kaggle", "code"}

    asyncio.run(scenario())


def test_no_auth_binding_leaves_routes_open(tmp_path) -> None:
    # The Unix-socket binding (auth_token=None) is the local trust boundary — no token required.
    async def scenario() -> None:
        transport = httpx.ASGITransport(app=_build(tmp_path, auth_token=None))
        async with httpx.AsyncClient(transport=transport, base_url="http://verity") as c:
            assert (await c.get("/catalog")).status_code == 200

    asyncio.run(scenario())


def test_byte_upload_and_artifact_download_round_trip(tmp_path) -> None:
    auth = {"Authorization": f"Bearer {_TOKEN}"}
    octet = {**auth, "Content-Type": "application/octet-stream"}

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=_build(tmp_path, auth_token=_TOKEN))
        async with httpx.AsyncClient(transport=transport, base_url="http://verity") as c:
            # Upload the user's role-keyed files as raw bytes over the wire (no shared volume).
            files = await _upload_role_files(c, octet)

            # create (role-keyed inputs) -> run -> an accepted Submission lands.
            task_id = (await c.post(
                "/tasks",
                json={"request": _fe_request_body(), "files": files},
                headers=auth,
            )).json()["task_id"]
            started = await c.post(f"/tasks/{task_id}/runs", json={}, headers=auth)
            run_id = started.json()["run_id"]
            assert await _poll_terminal(c, run_id, auth) == "done"
            results = (await c.get(f"/runs/{run_id}/results", headers=auth)).json()
            assert results["accepted_artifact_ids"]

            # Download the accepted submission's code over the wire — bytes match what was proposed.
            dl = await c.get(f"/artifacts/{run_id}/{ENTRYPOINT}", headers=auth)
            assert dl.status_code == 200
            assert dl.content == entrypoint_bytes(b"good", ["feat0"])
            # a missing object is a clean 404
            assert (await c.get(f"/artifacts/{run_id}/nope.py", headers=auth)).status_code == 404

    asyncio.run(scenario())


def test_oversize_upload_is_refused(tmp_path) -> None:
    auth = {"Authorization": f"Bearer {_TOKEN}", "Content-Type": "application/octet-stream"}

    async def scenario() -> None:
        app = _build(tmp_path, auth_token=_TOKEN, max_upload_bytes=10)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://verity") as c:
            assert (await c.post("/objects", content=b"x" * 11, headers=auth)).status_code == 413
            assert (await c.post("/objects", content=b"x" * 9, headers=auth)).status_code == 200

    asyncio.run(scenario())


def test_tcp_serve_refuses_without_a_token(tmp_path, monkeypatch) -> None:
    """The daemon will not expose a network port unauthenticated — it requires VERITY_API_TOKEN."""
    from verity.service import daemon

    monkeypatch.setenv("VERITY_EXCHANGE", str(tmp_path / "ex"))
    monkeypatch.setenv("VERITY_STORE_ROOT", str(tmp_path / "store"))
    monkeypatch.delenv("VERITY_API_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        daemon.main(http="127.0.0.1:0")


def test_over_real_tcp_port_with_bearer_auth(tmp_path) -> None:
    """The 8.4 binding for real: serve on a TCP port, reach it via the `Client` with a token."""
    import uvicorn

    from verity.service.client import Client, ControlServiceError

    async def scenario() -> None:
        app = _build(tmp_path, auth_token=_TOKEN)
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", lifespan="off")
        server = uvicorn.Server(config)
        serve_task = asyncio.create_task(server.serve())
        try:
            for _ in range(200):  # wait for the bind + the assigned ephemeral port
                if server.started and server.servers:
                    break
                await asyncio.sleep(0.01)
            assert server.started, "uvicorn did not start"
            port = server.servers[0].sockets[0].getsockname()[1]
            base = f"http://127.0.0.1:{port}"

            # an authenticated client reaches the verbs
            cat = await Client(base_url=base, token=_TOKEN).catalog()
            assert any(t["type_name"] == "fe-kaggle" for t in cat["task_types"])

            # a tokenless client is refused with a typed 401
            try:
                await Client(base_url=base).catalog()
                raise AssertionError("expected a 401 without a token")
            except ControlServiceError as exc:
                assert exc.status_code == 401
        finally:
            server.should_exit = True
            await serve_task

    asyncio.run(scenario())
