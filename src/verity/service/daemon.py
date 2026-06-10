"""The long-lived control-plane daemon (ROADMAP 8.2, ADR 0004 (a)): ``verity serve``.

Keeps alive — as the process's state — a **generic** `ControlService` (the per-task `ControlPlane`+
store multiplexer) over a `DockerBackend` and the built-in `TaskCatalog`, and serves the HTTP
surface (`service/http.py`) over a **Unix-domain socket** via uvicorn (``uds=``). The socket is the
v1 trust boundary: local/`docker exec`-only, no network exposure, so no auth is needed yet (the
external HTTP API + auth is 8.4 — the *same* `build_app`, rebound to a TCP port).

The control plane is constructed generic here — no domain import — a task is applied through the
service's API (via the catalog). Config is by env (the container surface). uvicorn/fastapi are
imported lazily (the ``service`` extra), mirroring `verifier/__main__.py`.
"""

from __future__ import annotations

import os
from pathlib import Path

from verity.composition import default_catalog
from verity.logging import configure_logging, get_logger
from verity.service.control_service import ControlService

__all__ = ["build_service", "exchange_dirs", "main"]

log = get_logger("verity.service.daemon")

DEFAULT_SOCKET = "/run/verity.sock"
DEFAULT_EXCHANGE = "/exchange"


def exchange_dirs(root: str | None = None) -> tuple[Path, Path]:
    """The ``(in, out)`` exchange dirs (created if absent). Client↔CP only — never a worker."""
    base = Path(root or os.environ.get("VERITY_EXCHANGE", DEFAULT_EXCHANGE))
    exchange_in, exchange_out = base / "in", base / "out"
    exchange_in.mkdir(parents=True, exist_ok=True)
    exchange_out.mkdir(parents=True, exist_ok=True)
    return exchange_in, exchange_out


def build_service(*, root: Path | None = None) -> ControlService:
    """A generic `ControlService` over a `DockerBackend` + the built-in catalog (no domain here)."""
    from verity.provisioning import DockerBackend  # local: only the daemon needs a real backend

    return ControlService(backend=DockerBackend(), root=root, catalog=default_catalog())


def main() -> None:
    """Run the daemon: build the generic service + app, serve over the Unix socket until killed."""
    configure_logging(json_output=True)
    import uvicorn  # lazy: only the running daemon needs the `service` extra

    from verity.service.http import build_app

    socket_path = os.environ.get("VERITY_SOCKET", DEFAULT_SOCKET)
    store_root = os.environ.get("VERITY_STORE_ROOT")
    exchange_in, exchange_out = exchange_dirs()
    service = build_service(root=Path(store_root) if store_root else None)
    app = build_app(service, exchange_in=exchange_in, exchange_out=exchange_out)

    sock = Path(socket_path)
    if sock.exists():  # clear a stale socket from a prior process so the bind succeeds
        sock.unlink()
    sock.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "daemon_start", socket=socket_path, store_root=store_root,
        exchange=str(exchange_in.parent),
    )
    uvicorn.run(app, uds=socket_path)


if __name__ == "__main__":
    main()
