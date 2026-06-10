"""``python -m verity.service`` → the control-plane daemon (ROADMAP 8.4).

The module entrypoint, mirroring ``python -m verity.verifier``: it delegates to the daemon's
``main()``, which is env-configured (``VERITY_HTTP`` selects the network/TCP binding, else the Unix
socket). The ``verity serve [--http …]`` console script is the equivalent flag-driven path.
"""

from __future__ import annotations

from verity.service.daemon import main

if __name__ == "__main__":
    main()
