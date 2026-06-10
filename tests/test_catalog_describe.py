"""Registry self-description (ROADMAP 8.2, ADR 0004 (c)): published task-type contracts.

`verity catalog` lets a client read each task type's contract before choosing it — the output-shape
the agent must produce (artifact types + operation signatures, sourced from the *same* domain schema
that renders the system prompt) and the verifier's approval-semantics prose. These pin that the
catalog publishes both halves, offline (no `service` extra).
"""

from __future__ import annotations

from verity.composition import UnknownTaskType, default_catalog


def test_describe_fe_publishes_shape_and_approach() -> None:
    desc = default_catalog().describe("fe").to_dict()
    assert desc["type_name"] == "fe"
    assert {"DatasetVersion", "Submission", "Feature"} <= set(desc["artifact_types"])
    assert "Submission" in desc["gated_types"] and "Feature" in desc["gated_types"]
    op_names = {o["name"] for o in desc["operations"]}
    assert {"submit", "revises", "harvest"} <= op_names
    assert all("required_payload_keys" in o for o in desc["operations"])  # the shape contract
    assert desc["domain_instructions"]  # the formal, agent-facing half
    assert "balanced accuracy" in desc["verifier_approach"]  # the approval-semantics prose
    assert desc["sandbox_notes"]


def test_describe_code_publishes_its_contract() -> None:
    desc = default_catalog().describe("code").to_dict()
    assert desc["type_name"] == "code"
    assert "Submission" in desc["gated_types"]
    assert {"submit"} <= {o["name"] for o in desc["operations"]}
    assert "parses" in desc["verifier_approach"] and "runs-clean" in desc["verifier_approach"]


def test_describe_all_covers_every_catalog_type() -> None:
    catalog = default_catalog()
    described = {d.type_name for d in catalog.describe_all()}
    assert described == set(catalog.types()) == {"fe", "code"}


def test_describe_unknown_type_raises() -> None:
    try:
        default_catalog().describe("nope")
        raise AssertionError("expected UnknownTaskType for an undescribed type")
    except UnknownTaskType:
        pass
