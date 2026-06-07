"""Sample submission scripts — the genuine code objects the stub agent emits (ROADMAP Phase 2.4).

NON-PRODUCT. These are *real* Python programs (not scripted markers): the stub agent writes one to
its outbox, the control plane harvests it, and the code domain's gates actually parse and execute
it. They let the verifier earn its verdicts on real artifacts before the feature-engineering domain
(§12) supplies real submissions.

* :data:`CLEAN` — computes a result, writes it to ``/out``, and exits 0 (self-contained).
* :data:`RAISES` — parses fine but raises at runtime, so the execution gate rejects it.
* :data:`SYNTAX_ERROR` — does not parse, so the cheap structural gate rejects it before any run.
"""

from __future__ import annotations

__all__ = ["CLEAN", "RAISES", "SYNTAX_ERROR"]

CLEAN = b"""
import json, pathlib

total = sum(n * n for n in range(10))
pathlib.Path('/out/result.json').write_text(json.dumps({'total': total}))
print('computed total', total)
"""

RAISES = b"""
raise ValueError('feature pipeline blew up')
"""

SYNTAX_ERROR = b"""
def feature(df)
    return df['a'] * 2
"""
