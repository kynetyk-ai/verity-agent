"""The verifier service entrypoint (ROADMAP Phase 7.1/7.2 → 9.1) — the networked verifier (#10/#73).

Runs the advisory verifier as a standing HTTP service the control plane dispatches to over an
``HttpTransport`` (the CP selects it with ``verifier_key="remote"``). The service wraps a
:class:`~verity.verifier.service.SdkVerifier` in a :class:`~verity.transport.server.VerifierServer`
and serves it with uvicorn — mirroring how :mod:`verity.sandbox.container_entry` wraps the agent.

``VERITY_VERIFIER`` selects the verifier. Two shapes:

* a **dataless** verifier (``fake``) is built immediately and served as a fixed impl;
* a **data-bearing** verifier (``fe-kaggle``) is built **server-side from the per-task**
  :class:`~verity.contracts.ports.VerifierSetup` the control plane ships at provisioning (§9.1) —
  so the gates + the ``kaggle`` extra live in *this* image, never the control plane's. It is
  registered as a *builder*, and the :class:`VerifierServer` constructs it on ``setup``.

The serving imports (uvicorn, the HTTP transport) are **lazy inside :func:`main`**, so the module
imports without the ``service`` extra (the entrypoint test needs no socket and no FastAPI).
"""

from __future__ import annotations

import os

from verity.logging import configure_logging, get_logger
from verity.transport.server import VerifierServer
from verity.verifier.registry import VerifierRegistry, load_verifier_plugins

log = get_logger("verity.verifier.service")

_DEFAULT_HOST = "0.0.0.0"  # noqa: S104 — a containerized service binds all interfaces by design
_DEFAULT_PORT = 8001


def build_server_from_env() -> VerifierServer:
    """Build the :class:`VerifierServer` named by ``VERITY_VERIFIER`` (default ``fake``).

    The available verifiers are **discovered** from the ``verity.verifier_types`` entry-point group
    (ADR 0006) — the built-in ``fake`` (dataless impl) and ``fe-kaggle`` (data-bearing setup) are
    dogfooded as entry points alongside any third-party verifier, so a new verifier ships its image
    + an entry point with no edit to this module. A fixed-impl server for a dataless verifier, or a
    builder-backed server for a data-bearing one (which receives its data via the ``setup``
    handshake). Neither this nor the registry imports the ``service`` extra, so the lightweight
    entrypoint test stays callable.
    """
    registry = load_verifier_plugins(VerifierRegistry())
    name = os.environ.get("VERITY_VERIFIER", "fake")
    impl = registry.impl(name)
    if impl is not None:
        return VerifierServer(impl())
    setup_builder = registry.setup(name)
    if setup_builder is not None:
        return VerifierServer(builder=setup_builder)
    raise SystemExit(f"unknown VERITY_VERIFIER={name!r}; known: {', '.join(registry.names())}")


def main() -> None:
    configure_logging(json_output=True)
    import uvicorn  # lazy: only the running service needs the `service` extra

    from verity.transport.http import build_asgi_app

    server = build_server_from_env()
    host = os.environ.get("VERITY_VERIFIER_HOST", _DEFAULT_HOST)
    port = int(os.environ.get("VERITY_VERIFIER_PORT", str(_DEFAULT_PORT)))
    log.info("verifier_service_start", verifier=os.environ.get("VERITY_VERIFIER", "fake"),
             host=host, port=port)
    uvicorn.run(build_asgi_app(server), host=host, port=port)


if __name__ == "__main__":
    main()
