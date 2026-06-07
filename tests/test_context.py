"""Context-assembly tests (spec §9) — ROADMAP Phase 1.3.

Includes the bounded-context guarantee (§9, acceptance criterion §13.5): assembled context
size must not grow with store size.
"""

from __future__ import annotations

from tests.helpers import bundle, decision, make_store, propose, returns
from verity.control_plane.commit import run_commit
from verity.control_plane.context import ContextAssembler
from verity.control_plane.store import ArtifactStatus, SqliteStore
from verity.domains.fake import FAKE_VERIFIER_IDENTITY, NOTE, SOURCE, build_fake_domain


def _fill(store: SqliteStore, n: int) -> None:
    """Propose ``n`` artifacts with fixed-width ids so per-entry rendering is constant-length."""
    for i in range(n):
        propose(store, artifact_id=f"a{i:05d}", artifact_type="Thing", payload={"v": i})


def test_manifest_is_bounded_and_reports_total() -> None:
    store = make_store()
    _fill(store, 500)
    assembler = ContextAssembler(manifest_limit=50)
    manifest = assembler.manifest(store)
    assert len(manifest.entries) == 50  # capped, regardless of 500 artifacts
    assert manifest.total == 500  # but the true total is still reported


def test_manifest_is_payload_free() -> None:
    store = make_store()
    propose(store, artifact_id="a1", artifact_type="Thing", payload={"secret": "do-not-leak"})
    rendered = ContextAssembler().manifest(store).render()
    assert "do-not-leak" not in rendered  # values never appear — only the key does
    assert "{secret}" in rendered


def test_bounded_context_guarantee_size_does_not_grow_with_store() -> None:
    prompt = "SYSTEM PROMPT"
    goal = "find the best Thing"

    small = make_store()
    _fill(small, 50)
    big = make_store()
    _fill(big, 2000)

    assembler = ContextAssembler(manifest_limit=50, tail_limit=12)
    small_size = assembler.assemble(small, system_prompt=prompt, goal=goal).size()
    big_size = assembler.assemble(big, system_prompt=prompt, goal=goal).size()

    # 40x the artifacts, but the assembled context is the same size bar the total-count label.
    assert abs(big_size - small_size) < 50


def test_assemble_is_deterministic_regenerate_never_append() -> None:
    store = make_store()
    _fill(store, 10)
    assembler = ContextAssembler()
    a = assembler.assemble(store, system_prompt="P", goal="g")
    b = assembler.assemble(store, system_prompt="P", goal="g")
    assert a.render() == b.render()  # pure function of the store + turn inputs


def test_stable_prefix_is_independent_of_goal_and_scratch() -> None:
    store = make_store()
    _fill(store, 10)
    assembler = ContextAssembler()
    first = assembler.assemble(store, system_prompt="P", goal="goal one", scratch="notes A")
    second = assembler.assemble(store, system_prompt="P", goal="goal two", scratch="notes B")
    # the prefix (prompt + manifest) is the prompt-cache-stable part; only the tail moves
    assert first.stable_prefix == second.stable_prefix
    assert first.volatile_tail != second.volatile_tail


def test_tail_is_bounded_by_tail_limit() -> None:
    store = make_store()
    _fill(store, 100)
    assembler = ContextAssembler(tail_limit=5)
    context = assembler.assemble(store, system_prompt="P", goal="g")
    # exactly tail_limit artifact lines under the retrieved section
    retrieved_block = context.volatile_tail.split("# Retrieved artifacts")[1]
    artifact_lines = [ln for ln in retrieved_block.splitlines() if ln.strip().startswith("- ")]
    assert len(artifact_lines) == 5


def test_committed_artifact_appears_in_regenerated_manifest() -> None:
    """Flush-on-commit: a committed artifact is retrievable from the store, not held in scratch."""
    store = make_store()
    domain = build_fake_domain()
    propose(store, artifact_id="src", artifact_type=SOURCE, is_root=True)
    propose(store, artifact_id="n1", artifact_type=NOTE, parents=["src"], payload={"text": "hi"})
    run_commit(
        "n1",
        store=store,
        sink=store,
        resolve_coverage=domain.gated_types.resolve,
        dispatch=returns(bundle(ArtifactStatus.ACCEPTED, decision("worth-keeping"))),
        verifier_identity=FAKE_VERIFIER_IDENTITY,
    )
    manifest = ContextAssembler().manifest(store)
    note_entries = [e for e in manifest.entries if e.artifact_id == "n1"]
    assert len(note_entries) == 1
    assert note_entries[0].status is ArtifactStatus.ACCEPTED
