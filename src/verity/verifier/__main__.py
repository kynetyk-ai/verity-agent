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
from collections.abc import Callable

from verity.contracts.ports import VerifierPort, VerifierSetup
from verity.domains.fake import build_fake_verifier
from verity.logging import configure_logging, get_logger
from verity.transport.server import VerifierServer
from verity.verifier.fe_kaggle_service import build_fe_kaggle_verifier_from_setup

log = get_logger("verity.verifier.service")

# Dataless verifiers: ready as soon as the service starts (no per-task setup payload).
_IMPL_BUILDERS: dict[str, Callable[[], VerifierPort]] = {
    "fake": build_fake_verifier,
}
# Data-bearing verifiers: built server-side from the control-plane's per-task VerifierSetup (§9.1).
_SETUP_BUILDERS: dict[str, Callable[[VerifierSetup], VerifierPort]] = {
    "fe-kaggle": build_fe_kaggle_verifier_from_setup,
}

_DEFAULT_HOST = "0.0.0.0"  # noqa: S104 — a containerized service binds all interfaces by design
_DEFAULT_PORT = 8001


def build_server_from_env() -> VerifierServer:
    """Build the :class:`VerifierServer` named by ``VERITY_VERIFIER`` (default ``fake``).

    A fixed-impl server for a dataless verifier, or a builder-backed server for a data-bearing one
    (which receives its data via the ``setup`` handshake). ``VerifierServer`` imports no ``service``
    extra, so this stays callable in the lightweight entrypoint test.
    """
    name = os.environ.get("VERITY_VERIFIER", "fake")
    if name in _IMPL_BUILDERS:
        return VerifierServer(_IMPL_BUILDERS[name]())
    if name in _SETUP_BUILDERS:
        return VerifierServer(builder=_SETUP_BUILDERS[name])
    known = ", ".join(sorted({*_IMPL_BUILDERS, *_SETUP_BUILDERS}))
    raise SystemExit(f"unknown VERITY_VERIFIER={name!r}; known: {known}")


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
