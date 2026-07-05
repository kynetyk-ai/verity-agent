"""Context assembly — regenerate the volatile context from the store each turn (spec §9).

The volatile context window is a *bounded, regenerated projection* of the durable store
(Principle 2), assembled by the control plane and served to the sandbox at the start of each
cycle. Each turn's context is two parts:

* A **stable prefix** — the composed system prompt (§3.4) and a slow-changing **manifest** (a
  compact, ranked *index* of what the store holds — never the artifacts themselves). The prompt's
  orientation/layout layers are invariant, so the prefix changes rarely, which preserves prompt
  caching. (The agent-facing *tool definitions* §9 also places here are harness-bound and arrive
  from the sandbox adapter in Phase 3; the control plane carries no tool registry — see
  :mod:`.registries`.)
* A **volatile tail** — the artifacts retrieved for *this* step (via the retrieval policy §8.4),
  the current goal, and a short rolling scratch. Rebuilt every turn.

Two rules make this a discipline, not a habit (§9):

* **Regenerate, never append.** :meth:`ContextAssembler.assemble` rebuilds from the store each
  call; nothing is carried forward and grown. It is a pure function of the store and the turn's
  goal/scratch.
* **Flush on commit.** Committing moves an artifact into durable state, so it appears in the
  regenerated manifest/retrieval rather than being held in the tail.

It carries the spec's one hard performance guarantee — the **bounded-context guarantee** (§9,
§13.5): the assembled context size must not grow with the size of the store. That is enforced
structurally here: the manifest is capped to ``manifest_limit`` fixed-width entries, the tail to
``tail_limit`` artifacts, and every per-item rendering is truncated — so a run that accumulates
thousands of artifacts assembles a context no larger than a short run did.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verity.control_plane.registries import DefaultRetrievalPolicy, RetrievalPolicy
from verity.control_plane.store import (
    Artifact,
    ArtifactStatus,
    ObjectRef,
    Payload,
    Store,
    VerdictKind,
)
from verity.logging import get_logger

__all__ = [
    "ManifestEntry",
    "Manifest",
    "AssembledContext",
    "ContextAssembler",
]

log = get_logger("verity.control_plane.context")


# Manifest ranking: trusted artifacts surface first (§6, §8.4), recency as tiebreak.
_STATUS_RANK: dict[ArtifactStatus, int] = {
    ArtifactStatus.ACCEPTED: 0,
    ArtifactStatus.TENTATIVE: 1,
    ArtifactStatus.PROPOSED: 2,
    ArtifactStatus.REVISED: 3,
    ArtifactStatus.SUPERSEDED: 4,
    ArtifactStatus.REJECTED: 5,
}


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One compact, payload-free index row: enough to know an artifact exists and ask for it."""

    artifact_id: str
    type: str
    status: ArtifactStatus
    summary: str

    def render(self) -> str:
        return f"  [{self.status.value:<10}] {self.type}/{self.artifact_id} — {self.summary}"


@dataclass(frozen=True, slots=True)
class Manifest:
    """A bounded, ranked summary of the store — the bridge between store and agent (§9)."""

    entries: tuple[ManifestEntry, ...]
    total: int

    def render(self) -> str:
        header = f"Store manifest (showing {len(self.entries)} of {self.total} artifacts):"
        if not self.entries:
            return header + "\n  (empty)"
        return "\n".join([header, *(e.render() for e in self.entries)])


@dataclass(frozen=True, slots=True)
class AssembledContext:
    """The two-part context served to the sandbox for one cycle (§9)."""

    stable_prefix: str
    volatile_tail: str

    def render(self) -> str:
        return f"{self.stable_prefix}\n\n{self.volatile_tail}"

    def size(self) -> int:
        """A proxy for context size (characters) — what the bounded-context test measures."""
        return len(self.render())


@dataclass(frozen=True, slots=True)
class ContextAssembler:
    """Assembles the bounded context from the store each turn (§9).

    The caps are what make the bounded-context guarantee structural: ``manifest_limit`` entries,
    ``tail_limit`` retrieved artifacts, and per-item truncation at ``summary_chars`` /
    ``tail_payload_chars``. Assembled size is therefore independent of store size.
    """

    retrieval: RetrievalPolicy = field(default_factory=DefaultRetrievalPolicy)
    manifest_limit: int = 50
    tail_limit: int = 12
    summary_chars: int = 120
    tail_payload_chars: int = 800
    # The prior-attempts digest (#132): the last N REJECTED artifacts with the rejecting gate's
    # rationale, so the agent sees WHY past attempts failed — all of them, not just the latest
    # correction — without their bytes being provisioned. 0 disables the section.
    digest_limit: int = 10
    digest_chars: int = 200

    def manifest(self, store: Store) -> Manifest:
        artifacts = store.query_artifacts()
        # Two stable passes: recency descending, then status ascending — trusted-and-recent first.
        ranked = sorted(artifacts, key=lambda a: a.created_at, reverse=True)
        ranked.sort(key=lambda a: _STATUS_RANK[a.status])
        entries = tuple(
            ManifestEntry(
                artifact_id=a.id,
                type=a.type,
                status=a.status,
                summary=_summarize(a.payload, self.summary_chars),
            )
            for a in ranked[: self.manifest_limit]
        )
        return Manifest(entries=entries, total=len(artifacts))

    def assemble(
        self,
        store: Store,
        *,
        system_prompt: str,
        goal: str,
        scratch: str = "",
        retrieval: RetrievalPolicy | None = None,
    ) -> AssembledContext:
        """Regenerate the full context for one turn — a pure function of the store + goal (§9).

        ``retrieval`` is the task's policy (§8.4); when given it drives the tail instead of this
        assembler's default, so a domain-supplied policy actually feeds context assembly. The
        bounded-context caps are the assembler's regardless, so the guarantee is unaffected.
        """
        manifest = self.manifest(store)
        stable_prefix = f"{system_prompt}\n\n{manifest.render()}"

        policy = retrieval if retrieval is not None else self.retrieval
        retrieved = policy.select(goal, store, limit=self.tail_limit)
        tail_sections = [
            "# Current goal",
            goal.strip() or "(none)",
            "# Retrieved artifacts",
            _render_artifacts(retrieved, self.tail_payload_chars),
        ]
        digest = self._rejection_digest(store)
        if digest:
            tail_sections += ["# Prior attempts (rejected)", digest]
        if scratch.strip():
            tail_sections += ["# Scratch", scratch.strip()]
        volatile_tail = "\n\n".join(tail_sections)

        context = AssembledContext(stable_prefix=stable_prefix, volatile_tail=volatile_tail)
        log.debug(
            "context_assembled",
            manifest_entries=len(manifest.entries),
            store_total=manifest.total,
            retrieved=len(retrieved),
            size=context.size(),
        )
        return context


    def _rejection_digest(self, store: Store) -> str:
        """The prior-attempts digest (#132/#32): the last ``digest_limit`` failed attempts — both
        REJECTED artifacts (with the rejecting gate's rationale + recorded score) and FAILED
        cycles that minted no artifact at all (a timed-out/crashed sandbox, a malformed proposal
        — read from the durable ``cycle_failures`` rows), interleaved by recency, one capped line
        per attempt.

        Store-derived every turn (never accumulated loop state — restart-safe), and bounded by
        construction (``digest_limit`` × ``digest_chars``), so the bounded-context guarantee
        holds. §10 permits feeding gate reasons TO the proposer; the digest never carries the
        proposer's own rationale. Gate-unavailable failures are recorded durably but NOT surfaced
        here — their correction explicitly says "this was not a rejection; submit again", and
        teaching the agent to route around transient infrastructure is the wrong lesson. The
        single ``# Correction`` feedback line stays the salient latest-failure channel; this is
        the trail of everything before it, so the agent stops re-walking dead ends even when the
        failed attempts left no artifact bytes to provision.
        """
        if self.digest_limit <= 0:
            return ""
        # (created_at, line) pairs from both sources, merged by recency.
        attempts: list[tuple[str, str]] = []
        for artifact in store.query_artifacts():
            if artifact.status is not ArtifactStatus.REJECTED:
                continue
            gate, rationale, score = _rejecting_decision(store, artifact.id)
            score_part = f" score={score:.4f}" if score is not None else ""
            attempts.append((
                artifact.created_at,
                f"  - {artifact.type}/{artifact.id} [{gate}]{score_part} — "
                f"{self._cap(rationale)}",
            ))
        for failure in store.cycle_failures():
            if failure.kind == "gate-error":
                continue  # transient infra, not an attempt to learn from (see docstring)
            attempts.append((
                failure.created_at,
                f"  - (no proposal) [{failure.kind}] — {self._cap(failure.detail)}",
            ))
        attempts.sort(key=lambda pair: pair[0], reverse=True)  # most recent attempts first
        return "\n".join(line for _, line in attempts[: self.digest_limit])

    def _cap(self, text: str) -> str:
        return text if len(text) <= self.digest_chars else text[: self.digest_chars - 1] + "…"


def _rejecting_decision(store: Store, artifact_id: str) -> tuple[str, str, float | None]:
    """The (gate, rationale, score) of the decision that rejected ``artifact_id``.

    Reads the durable ``Decision`` rows; falls back to a generic line when none carries a reject
    verdict (e.g. a shape-level termination) so the digest never crashes context assembly.
    """
    for decision in store.decisions_for(artifact_id):
        if decision.verdict is VerdictKind.REJECT:
            return decision.gate, decision.rationale, decision.score
    return "gate", "rejected (no recorded rationale)", None


def _render_artifacts(artifacts: list[Artifact], payload_chars: int) -> str:
    if not artifacts:
        return "  (none retrieved)"
    return "\n".join(
        f"  - {a.type}/{a.id} [{a.status.value}]: {_summarize(a.payload, payload_chars)}"
        for a in artifacts
    )


def _summarize(payload: Payload, limit: int) -> str:
    """A compact, length-capped rendering of a payload — never the full bytes (§9)."""
    if isinstance(payload, ObjectRef):
        return f"<object {payload.blob_ref}>"
    if isinstance(payload, dict):
        rendered = "{" + ", ".join(sorted(payload)) + "}"
    elif isinstance(payload, list):
        rendered = f"[list of {len(payload)}]"
    else:
        rendered = str(payload)
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"
