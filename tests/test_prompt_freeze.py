"""Prompt freeze pins (#138) — the agent-facing surface's spec of "unchanged".

The ablation ladder's construct validity depends on every rung seeing byte-identical instructions.
These pins hash the two composed-prompt layers the experiments rely on: the kernel orientation
(workspace contract v2) and the FE domain instructions at the ladder's calibrated gate budget.

A failing pin means the agent-facing surface changed. That is sometimes intended — the #138
commodious-workspace package changed it deliberately — but it must be a *decision*, not drift:
consult the study owner, record the change (experimental-design §7), bump
``WORKSPACE_CONTRACT_VERSION`` if the orientation moved, and re-pin. Never adjust a pin casually
mid-study. (The exp6 branch carries an older pin set in ``test_propose_guidance.py``; re-sync it
against these when the branches merge.)
"""

from __future__ import annotations

import hashlib

from verity.control_plane.config import _render_orientation, orientation_digest
from verity.control_plane.workspace import WORKSPACE_CONTRACT
from verity.domains.feature_engineering import feature_engineering_instructions

# Pinned 2026-07-07 with the #138 package (contract v2; parametrized FE instructions).
ORIENTATION_DIGEST_V2 = "740eada88f00be0dd46867fb51436a2d17aefe9970593d92b3e05c15a5ba1daa"
FE_INSTRUCTIONS_2400_SHA256 = "57102c2138fc90453da28d9359fc042badd4c3d7fb3b9fcc4dd4ecf0a1ccc3d4"


def test_kernel_orientation_is_pinned() -> None:
    assert WORKSPACE_CONTRACT.version == 2
    assert orientation_digest(WORKSPACE_CONTRACT) == ORIENTATION_DIGEST_V2


def test_orientation_names_the_provisioned_artifacts_path() -> None:
    """The #138 discoverability fix: the map names scratch/provided/ + INDEX.md."""
    rendered = _render_orientation(WORKSPACE_CONTRACT)
    assert "scratch/provided/" in rendered
    assert "INDEX.md" in rendered


def test_fe_instructions_at_ladder_budget_are_pinned() -> None:
    text = feature_engineering_instructions(2400.0)
    assert hashlib.sha256(text.encode()).hexdigest() == FE_INSTRUCTIONS_2400_SHA256


def test_fe_instructions_state_the_real_runner_budget() -> None:
    """#138: the budget is a stated venue fact, parametrized per task — never a magic constant."""
    assert "~40 minutes" in feature_engineering_instructions(2400.0)
    assert "~216 minutes" in feature_engineering_instructions(12960.0)
