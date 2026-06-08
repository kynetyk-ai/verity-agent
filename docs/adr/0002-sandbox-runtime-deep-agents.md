# ADR 0002 — The sandbox runtime: Deep Agents, behind a framework-neutral adapter

- **Status:** Accepted — Sprint 1 implemented on `feat/phase3-sandbox`
- **Date:** 2026-06-08
- **Affects spec:** §3.4–§3.5 (the sandbox service), §8.2 (the harness-bound tool form). Realizes the
  "sandbox adapter" already anticipated by the §8.2 collapse note in `control_plane/registries.py`.

## Context

Phase 3 replaces the scripted stub agent with a real sandbox: the agent runtime + workspace where the
loop runs, behind the `SandboxPort` contract. We evaluated three open-source frameworks (pi-mono, the
Claude Agent SDK, LangChain Deep Agents) and ran a spike (`spikes/phase3_sandbox/`, on the
`spike/phase3-deepagents-sandbox` branch) that classified a real 577k-row stellar dataset at ~96%
inside an isolated container.

Two product requirements drove the choice:

1. **Provider-agnosticism is a thesis requirement.** A chief goal is reaching good proposals from
   *cheap open-source models*. A Claude-only runtime (the Claude Agent SDK) would force a framework
   swap to go open, so it is at best a prototype accelerator, not the foundation.
2. **The sandbox is a general-purpose coding agent in YOLO mode.** Full coding capability incl. code
   execution, **no** permission prompts / human-in-the-loop — the **isolation boundary** is the
   safety, not in-loop gating. All project-specific steer comes through the control plane (the composed
   system prompt, the read-only workspace roles, the operation tools derived from the schema); nothing
   task-specific is hardcoded in the sandbox.

pi-mono is excellent but TypeScript: the sandbox would be a Node service driven over RPC, every tool a
mounted `.ts`, with no over-the-wire tool registration and no MCP. Deep Agents is Python-first (tools
are plain functions in our own language), provider-agnostic (one-line model swap: `anthropic:…` ↔ a
local OpenAI-compatible endpoint), has first-class skills, and — confirmed in the build — is usable as
a black box (`.invoke`, no `StateGraph` authoring, ephemeral by default).

## Decision

**Adopt Deep Agents as the first sandbox runtime, behind a framework-neutral adapter**, and build
Phase 3 as two sprints sharing one `SandboxPort`.

### (a) The proposal-descriptor seam — the agent declares, the host mints

The agent never mints ids or touches the control plane. Each bound operation tool writes a JSON
**descriptor** `{op_name, parents, payload, metadata}` to a reserved outbox file
(`__proposal__.json`); object attachments (a script, a data file) are ordinary outbox files. The
trusted host (`AgentSandbox`) harvests the outbox, splits the descriptor from the attachments, and
**mints** the typed `Artifact` + `Operation` from an injected clock / id-source. So a (later, possibly
hostile/containerized) agent only *declares* op/parents/payload — it cannot forge ids or lineage. This
also makes in-process and container isolation share identical host-side code.

### (b) Two layers + a thin seam

- A **framework-neutral core** (`AgentSandbox`, the `SandboxPort`): workspace lifecycle over
  `DefaultLayout`, harvest/split/mint, envelope assembly. Imports no framework; strict-typed.
- A **`SandboxDriver`** seam: "run the agent loop until a descriptor lands in the outbox," returns
  nothing.
- A **Deep Agents driver** (`DeepAgentsInProcessDriver`): the one module importing
  deepagents/langchain. Generates **one propose tool per `OperationSignature`** plus the general
  coding toolset + a `run_shell` execution tool, with **no `interrupt_on`/HITL** (YOLO).

### (c) Harness-bound tools (realizes §8.2)

The §8.2 tool registry stays collapsed in the control plane (a tool's harness-agnostic content is its
`OperationSignature`). The **harness-bound, executable form** lives here in the sandbox adapter,
parallel to `WorkspaceLayout` — exactly as the registries module's note anticipated.

## Additional sandbox configurations (the extensibility design)

New configs are **additive registrations or new drivers** — never changes to the control plane,
registries, or domains. Each axis maps to one seam:

| Axis | Seam | Cost |
|---|---|---|
| Another model (cheap open model, another Claude) | `model` in the registration closure | one `register(...)` line; a task picks it via `TaskConfig.sandbox_key` |
| Isolation (in-process → container → networked) | a new `SandboxDriver` sharing the host-side mint core | one driver class |
| Another framework (pi-mono, Claude SDK, hand-rolled) | a new `SandboxDriver` honoring the descriptor contract | one driver module |
| Different workspace layout | a different `WorkspaceLayout` passed to the core | a factory arg |

Two structural guarantees: the control plane depends only on `SandboxPort` + `ProviderRegistry`
(config-keyed), and the framework-specific code is quarantined behind the descriptor contract, so the
trusted core (mint, provenance, harvest, lifecycle) is shared by every present and future config.

## Sprints

- **Sprint 1 (this change) — in-process.** `src/verity/sandbox/` (core, driver seam, Deep Agents
  driver, factory); `deepagents` graduated from the `spike` group to a `sandbox` optional extra; mypy
  override quarantines the untyped import. Offline tests drive `read→propose→gate→commit` on the fake
  + code domains (fake driver, and a fake-model Deep Agents run); a live Claude smoke commits an
  accepted `Note` end to end. **In-process runs untrusted code on the host**, so it is the wiring/dev
  harness — the live smoke is code-execution-free (the `Note` domain).
- **Sprint 2 — container + cross-cycle exit.** `DeepAgentsContainerDriver` (the spike's hostile-input
  posture: outbound network for the API, non-root, caps dropped, outbox the only writable mount), full
  `data_sources` mounting, cross-cycle ephemerality + shape-error/refine feedback-driven revision, and
  the §13 exit tests. Safe YOLO execution of arbitrary code lands here.

## Consequences

- The same framework spans prototype (Claude) and the open-model thesis (one-line model swap), so the
  prototype is the foundation rather than a throwaway.
- A new runtime dependency (deepagents + its langchain/langgraph stack, ~53 pkgs) is added — confined
  to the `sandbox` extra and the sandbox container, never the lean core.
- The `spikes/phase3_sandbox/` tree stays as cited reference; it is removed once Sprint 2 lands.
