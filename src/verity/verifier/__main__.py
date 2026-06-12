"""The verifier service entrypoint (ROADMAP Phase 7.1/7.2) — the networked verifier (#10).

Runs the advisory verifier as a standing HTTP service the control plane dispatches to over an
``HttpTransport`` (the CP selects it with ``verifier_key="remote"``). The
service wraps the in-process :class:`~verity.verifier.service.SdkVerifier` in a
:class:`~verity.transport.server.VerifierServer` and serves it with uvicorn — mirroring how
:mod:`verity.sandbox.container_entry` wraps the agent. ``build_verifier_from_env`` selects the
verifier by ``VERITY_VERIFIER`` (a registry, defaulting to the fake domain, which needs no data);
data-bearing domains (feature-engineering's CSVs) wire their inputs here and are a follow-up.

The serving imports (uvicorn, the HTTP transport) are **lazy inside :func:`main`**, so the module —
and ``build_verifier_from_env`` — import without the ``service`` extra (the entrypoint test needs no
socket and no FastAPI).
"""

from __future__ import annotations

import os
from collections.abc import Callable

from verity.contracts.ports import VerifierPort
from verity.domains.fake import build_fake_verifier
from verity.logging import configure_logging, get_logger

log = get_logger("verity.verifier.service")

# Verifier builders selectable by env. Data-bearing domains (feature-engineering) add an entry that
# reads its dataset/labels from mounted paths — deferred until the data-plane mount story (#3).
_VERIFIER_BUILDERS: dict[str, Callable[[], VerifierPort]] = {
    "fake": build_fake_verifier,
}

_DEFAULT_HOST = "0.0.0.0"  # noqa: S104 — a containerized service binds all interfaces by design
_DEFAULT_PORT = 8001


def build_verifier_from_env() -> VerifierPort:
    """Construct the verifier named by ``VERITY_VERIFIER`` (default ``fake``)."""
    name = os.environ.get("VERITY_VERIFIER", "fake")
    builder = _VERIFIER_BUILDERS.get(name)
    if builder is None:
        known = ", ".join(sorted(_VERIFIER_BUILDERS))
        raise SystemExit(f"unknown VERITY_VERIFIER={name!r}; known: {known}")
    return builder()


def main() -> None:
    configure_logging(json_output=True)
    import uvicorn  # lazy: only the running service needs the `service` extra

    from verity.transport.http import build_asgi_app
    from verity.transport.server import VerifierServer

    verifier = build_verifier_from_env()
    host = os.environ.get("VERITY_VERIFIER_HOST", _DEFAULT_HOST)
    port = int(os.environ.get("VERITY_VERIFIER_PORT", str(_DEFAULT_PORT)))
    log.info("verifier_service_start", identity=verifier.identity, host=host, port=port)
    uvicorn.run(build_asgi_app(VerifierServer(verifier)), host=host, port=port)


if __name__ == "__main__":
    main()
