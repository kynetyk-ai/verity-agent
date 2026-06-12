"""Task config + composed-prompt tests (spec §3.4) — ROADMAP Phase 1.2."""

from __future__ import annotations

from verity.control_plane.config import (
    TaskConfig,
    compose_system_prompt,
    render_workspace_layout,
)
from verity.control_plane.registries import DefaultRetrievalPolicy
from verity.control_plane.workspace import WORKSPACE_CONTRACT
from verity.domains.fake import build_fake_domain


def test_layout_render_lists_every_role() -> None:
    rendered = render_workspace_layout(WORKSPACE_CONTRACT)
    for role in ("data", "context", "tools", "spec", "scratch", "outbox"):
        assert f"{role}/" in rendered


def test_compose_is_deterministic_and_three_layered() -> None:
    a = compose_system_prompt(
        contract=WORKSPACE_CONTRACT,
        domain_instructions="Produce a Note.",
        task_instructions="Summarize the source.",
    )
    b = compose_system_prompt(
        contract=WORKSPACE_CONTRACT,
        domain_instructions="Produce a Note.",
        task_instructions="Summarize the source.",
    )
    assert a == b  # pure string assembly — deterministic (§3.4)
    assert "# Kernel orientation (invariant)" in a
    assert "# Domain instructions" in a
    assert "# Task instructions" in a
    assert a.index("# Kernel orientation") < a.index("# Domain") < a.index("# Task")
    assert "Produce a Note." in a
    assert "Summarize the source." in a
    # the invariant orientation mentions the loop and the outbox harvest source
    assert "read -> propose -> gate -> commit" in a
    assert "outbox/" in a


def test_task_config_system_prompt_uses_its_layers() -> None:
    domain = build_fake_domain()
    config = TaskConfig(
        task_id="t1",
        instructions="Write a tidy note.",
        domain_instructions="A Note has a text field.",
        schema=domain.schema,
        gated_types=domain.gated_types,
        retrieval=DefaultRetrievalPolicy(),
        shape_validator=domain.shape_validator,
        object_namer=lambda _a: frozenset(),
        sandbox_key="stub",
        verifier_key="stub",
    )
    prompt = config.system_prompt()
    assert "Write a tidy note." in prompt
    assert "A Note has a text field." in prompt
