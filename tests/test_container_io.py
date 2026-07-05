"""The host↔container cycle-input protocol + the step/recursion budget derivations (#103).

Driver-agnostic: these exercise :mod:`verity.sandbox.container_io` alone (no Docker, no Deep
Agents). Moved out of the retired legacy-container-driver test module (#129).
"""

from __future__ import annotations

from verity.control_plane.registries import OperationSignature
from verity.sandbox.container_io import CycleInput


def test_cycle_input_roundtrips() -> None:
    ci = CycleInput(
        system_prompt="SYS",
        user_message="do it",
        operations=(
            OperationSignature(
                "submit", ("Dataset",), "Submission",
                object_payload_keys=("entrypoint", "requirements"),  # declared outbox files
            ),
        ),
        recursion_limit=42,
        deadline_s=1500.0,  # soft wrap-up budget (5.2)
        tool_names=("read_pdf",),  # extra sandbox tools by name (5.4)
        step_budget=30,  # per-cycle model-step budget (5.1)
    )
    back = CycleInput.from_json(ci.to_json())
    assert back == ci
    assert back.deadline_s == 1500.0 and back.tool_names == ("read_pdf",)
    assert back.step_budget == 30
    # the op's object-payload keys survive the host -> container hop (drives the presence check)
    assert back.operations[0].object_payload_keys == ("entrypoint", "requirements")


def test_effective_step_budget_is_a_fixed_model_step_default() -> None:
    # #103: an explicit step budget wins; an unset one is a FIXED default in MODEL-STEP units (NOT a
    # fraction of recursion_limit, which counts super-steps — the bug that defeated the safety net).
    from verity.sandbox.container_io import _DEFAULT_STEP_BUDGET, effective_step_budget

    assert effective_step_budget(300, 250) == 250  # explicit override
    # The default no longer depends on recursion_limit (was 0.8*recursion_limit; now fixed).
    assert effective_step_budget(300, None) == _DEFAULT_STEP_BUDGET == 80
    assert effective_step_budget(80, None) == 80
    assert effective_step_budget(1, None) == 80


def test_effective_recursion_limit_is_a_backstop_clamped_above_the_step_budget() -> None:
    # The recursion backstop must sit ABOVE the model-step budget so the graceful StepBudgetExceeded
    # fires before the ungraceful GraphRecursionError — by construction, clamped UP only.
    from verity.sandbox.container_io import (
        _RECURSION_HEADROOM,
        effective_recursion_limit,
        effective_step_budget,
    )

    assert effective_recursion_limit(200, None) == 640  # 80*8 floor raises the 200 default
    assert effective_recursion_limit(800, None) == 800  # explicit higher request honored (kept)
    assert effective_recursion_limit(300, 250) == 2000  # 250*8 — sized to the explicit step budget
    # Invariant: the backstop is always >= step_budget * headroom, for any inputs.
    for r, b in [(80, None), (200, None), (300, 250), (1000, 50), (50, None)]:
        assert effective_recursion_limit(r, b) >= effective_step_budget(r, b) * _RECURSION_HEADROOM
