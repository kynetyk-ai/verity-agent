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


# ------------------------------------------------------- the prior-attempts digest (#132)


def _reject(
    store: SqliteStore, artifact_id: str, *, gate: str = "selection",
    rationale: str = "does not improve", score: float | None = None,
) -> None:
    from verity.control_plane.store import VerdictKind

    propose(store, artifact_id=artifact_id, artifact_type=NOTE, parents=["src"],
            payload={"text": artifact_id})
    run_commit(
        artifact_id,
        store=store,
        sink=store,
        resolve_coverage=build_fake_domain().gated_types.resolve,
        dispatch=returns(bundle(
            ArtifactStatus.REJECTED,
            decision(gate, VerdictKind.REJECT, rationale, score=score),
        )),
        verifier_identity=FAKE_VERIFIER_IDENTITY,
    )


def _tail(store: SqliteStore, assembler: ContextAssembler | None = None) -> str:
    return (assembler or ContextAssembler()).assemble(
        store, system_prompt="SYS", goal="improve"
    ).volatile_tail


def test_digest_lists_every_rejects_gate_rationale_and_score() -> None:
    # The agent sees WHY past attempts failed — all of them — even though rejected artifacts are
    # excluded from the retrieved tail and their bytes may never be provisioned (#132).
    store = make_store()
    propose(store, artifact_id="src", artifact_type=SOURCE, is_root=True)
    _reject(store, "n1", gate="runs-clean",
            rationale="submission exited 1: ModuleNotFoundError: No module named 'xgboost'")
    _reject(store, "n2", gate="selection",
            rationale="does not improve: balanced-accuracy 0.9481 vs bar 0.9532", score=0.9481)

    tail = _tail(store)
    assert "# Prior attempts (rejected)" in tail
    assert "Note/n1 [runs-clean] — submission exited 1: ModuleNotFoundError" in tail
    assert "Note/n2 [selection] score=0.9481 — does not improve" in tail
    # most recent failure first
    assert tail.index("Note/n2") < tail.index("Note/n1")


def test_digest_is_bounded_and_truncates_rationales() -> None:
    store = make_store()
    propose(store, artifact_id="src", artifact_type=SOURCE, is_root=True)
    for i in range(30):
        _reject(store, f"n{i:03d}", rationale="x" * 5_000)

    assembler = ContextAssembler(digest_limit=5, digest_chars=100)
    tail = _tail(store, assembler)
    digest_lines = [ln for ln in tail.splitlines() if ln.strip().startswith("- Note/")]
    assert len(digest_lines) == 5  # capped at digest_limit, not the 30 rejects in the store
    assert all(len(ln) < 160 for ln in digest_lines)  # per-line rationale truncation
    assert "…" in digest_lines[0]


def test_digest_disabled_and_absent_when_no_rejects() -> None:
    store = make_store()
    propose(store, artifact_id="src", artifact_type=SOURCE, is_root=True)
    assert "# Prior attempts" not in _tail(store)  # nothing rejected yet

    _reject(store, "n1")
    assert "# Prior attempts" in _tail(store)
    assert "# Prior attempts" not in _tail(store, ContextAssembler(digest_limit=0))  # opt-out


def test_bounded_context_holds_with_many_rejects() -> None:
    # The §9 guarantee extends to the digest: 10 vs 500 rejects assemble to the same order of size.
    def _sized(n: int) -> int:
        store = make_store()
        propose(store, artifact_id="src", artifact_type=SOURCE, is_root=True)
        for i in range(n):
            _reject(store, f"n{i:05d}", rationale="does not improve " * 20)
        return ContextAssembler().assemble(store, system_prompt="SYS", goal="g").size()

    # both points sit past every cap (manifest 50, tail 12, digest 10), like the existing §13.5
    # guarantee test — below the caps the context legitimately grows toward them
    small, large = _sized(100), _sized(1000)
    assert large <= small * 1.1  # no growth with store size beyond fixed-width jitter
