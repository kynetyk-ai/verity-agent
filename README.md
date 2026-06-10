# Verity

**An implementation of the Self-Revising Discovery Harness** — a domain-agnostic kernel that gives an
agent a *typed provenance record* (artifacts, the operations that produced them, and the gate
decisions about them) as its durable state, behind a verifier-aware commit lifecycle. The shorthand
is *git + a type system + a verifier-aware lifecycle for agent artifacts.*

The name carries the point: **verity** = truth, and *verify*. The system exists to answer one
question — *can you trust, and audit, what the system believes?*

> **Status: post-MVP.** Phases 0–5 are complete — the control plane, the opaque verifier, the real
> Deep Agents sandbox, and the feature-engineering domain (spec §12) reached MVP with the **twelve §13
> acceptance criteria passing**, plus the Phase 5 reliability & observability hardening
> (degrade-don't-crash on both service boundaries, typed errors, retries, atomic commits, a
> store-derived `RunReport`). **Phase 6 (model breadth)** is code-complete and **live-validated on a
> local open model** (`good proposals from cheap models` — the thesis); hosted-OpenAI live validation
> is pending a key. **Phase 7 (service split & full containerization)** has reached its done-line: a
> task-agnostic control plane runs *in a container* and launches ephemeral sandbox + code-runner
> **worker containers**, running the §12 FE test by configuring the CP through its API. See
> [ROADMAP.md](ROADMAP.md) for the live plan and *Run the feature-engineering demo* below. Stack:
> Python, managed with [uv](https://docs.astral.sh/uv/) (see *Coding habits* in [CLAUDE.md](CLAUDE.md)).

## The specification (self-contained)

The full natural-language spec and its grounding references are **vendored into this repo** so it is
self-contained and can be worked on anywhere:

```
spec/
  self-revising-discovery-harness.md      the spec (v0.2) — the authority for this implementation
  references/
    2606.01444v1.pdf                      Wang & Buehler, the conceptual-origin paper
    scienceclaw-evaluation.md             prior-art analysis (typed provenance, no gate)
    pilar-comparison.md                   prior-art analysis (enforced soft gate)
```

These are a copy synced from the sibling spec repo (`NL-specs`,
`agents-and-harnesses/self-revising-discovery/`). If the upstream spec changes, re-sync the `spec/`
tree. Treat `spec/self-revising-discovery-harness.md` as the source of truth for what to build;
implementation decisions should trace to a spec section (cited `§N`).

## Architecture in brief

Three cooperating services (spec §3.3–§3.6):

- **Control plane** — the only service that mutates durable state; deliberately unintelligent. Owns
  the typed-provenance store, the commit path, the lifecycle, context assembly, the registries, the
  task config, loop control, and the invariant workspace contract + composed system prompt.
- **Sandbox** — the agent runtime + workspace; ephemeral (regenerated each cycle); proposes but never
  writes; emits objects to the workspace **outbox** for harvest.
- **Verifier** — advisory; hosts the gate plugins; takes a shaped proposal + a declared store-slice
  and returns a verdict; never writes.

The loop is **read → propose → gate → commit**. Load-bearing rules: **no implicit accept**, the
proposer is never its own gate, a status richer than accept/reject
(`proposed → tentative → accepted`, plus `rejected` / `superseded` / `revised`), and the `refine`
verdict for recovering a mostly-sound artifact with a localized defect.

## Roadmap

The path to MVP lived in **[ROADMAP.md](ROADMAP.md)** — the canonical "where are we, what's next."
The arc: build the **control plane** first and prove it against bespoke test doubles (Phase 1), then
swap in the real **verifier** (Phase 2) and **sandbox** (Phase 3), and reach **MVP** with the
feature-engineering domain passing the spec §13 acceptance criteria (Phase 4 — **done**). The roadmap
now tracks the post-MVP backlog.

## Develop

```
just install     # uv sync  (add `uv sync --extra sandbox` for the Deep Agents sandbox)
just check       # lint + type-check + test (must be green before any PR)
just test        # pytest
```

The suite runs offline by default; `@pytest.mark.docker` tests need Docker and `@pytest.mark.live`
tests need an API key — both **auto-skip** when their prerequisites are absent, so plain `just check`
needs neither.

## Run the feature-engineering demo

The §12 feature-engineering domain is the MVP validation target: an agent is given a dataset and asked
to design 1–5 features that improve a model, judged by running its submitted script on a **reserved
hold-out it never sees**. Three ways to exercise it, in increasing cost:

**1. The twelve §13 acceptance criteria (offline — no key, no Docker).** The whole
read → propose → gate → commit loop on a deterministic fake runner:

```
uv run --extra sandbox pytest tests/test_feature_engineering_acceptance.py
```

**2. A real script executed in a container (needs Docker, no API key).** Installs pandas and scores a
genuine script end-to-end through the verifier's container runner:

```
uv run --extra sandbox pytest tests/test_feature_engineering_gates.py -m docker
```

**3. The live multi-round run (real Claude — needs an API key, Docker, and a dataset).** Real agent
proposals, scored in a container, improving on the provisioned incumbent across rounds:

```
# a. provide a key
cp .env.example .env            # then set ANTHROPIC_API_KEY

# b. place the dataset (a CSV with an `id`, a `class` target, and feature columns)
#    at feature-engineering-test/{train.csv,test.csv}  (gitignored)

# c. build the sandbox image the agent runs in (once)
docker build -f Dockerfile.sandbox -t verity-sandbox:latest .

# d. run it (loads .env into the environment first)
set -a; . ./.env; set +a
uv run --extra sandbox pytest tests/test_feature_engineering_acceptance.py::test_feature_engineering_live -s
```

Each cycle spins a fresh, isolated container; expect a few minutes per round. The agent is steered
**only** through the control plane (task + domain instructions and the mounted data) — nothing
task-specific is hardcoded in the sandbox.

**4. The fully containerized run (the Phase 7 done-line — needs Docker; a key or a local model).** A
task-agnostic control plane runs *in* a container and launches the sandbox + code-runner **worker
containers** as siblings on the host daemon, configuring the §12 FE test through its own API:

```
just fe-containerized
```

This builds the sandbox + control-plane images and runs one FE task through `infra/compose.fe.yml`.
Configuration is by environment (`VERITY_MODEL`, `VERITY_LOCAL_BASE_URL`, `VERITY_MAX_CYCLES`, …);
point it at a local open model with `VERITY_LOCAL_BASE_URL`. The full env surface, the worked API
example, and the topology are documented in [docs/api-surface.md](docs/api-surface.md) (*Worked
example — the feature-engineering task*).

## Repo layout

```
README.md     this file
CLAUDE.md     orientation for coding agents working in this repo
ROADMAP.md    the living path (now: post-MVP backlog)
.env.example  copy to .env for the live demo
pyproject.toml / justfile   uv project + dev commands
Dockerfile.sandbox          the image the container sandbox runs the agent in
Dockerfile.verifier         the standing advisory-verifier service image (Phase 7.1)
Dockerfile.controlplane     the task-agnostic control-plane image (launches worker containers)
infra/            compose files for the containerized topology (compose.fe.yml)
docs/             API reference, guides, and architecture decision records (see Docs below)
src/verity/
  contracts/      the cross-service value model + service ports (no service depends on another)
  control_plane/  the sole mutator: store, commit lifecycle, registries, context assembly, loop
  verifier/       the opaque verifier: gate-primitive SDK + the container/worker code-runner
  sandbox/        the real agent runtime (Deep Agents) — in-process + container/worker drivers
  domains/        per-domain wiring; feature_engineering.py is the §12 MVP domain
  composition/    applies a task to a generic control plane (configure_fe_task) + the FE-run entrypoint
  provisioning/   the WorkerBackend seam + DockerBackend + the label-reaper (ADR 0003)
  eval/           the cross-model benchmark harness (quality / cost / latency)
  transport/      networked port-RPC (loopback + HTTP) for the service split
  logging.py / retry.py / telemetry.py   structured logging, bounded-backoff retries, metrics sink
tools/harness/    non-product test doubles + the dataset-split helper
tests/            test suite (offline by default; docker/live auto-skip)
spec/             vendored specification + references (see above)
```

## Docs

Reference material lives in [`docs/`](docs/):

- [`docs/api-surface.md`](docs/api-surface.md) — the **control-plane API reference**, with a worked
  feature-engineering example (programmatic + fully containerized).
- [`docs/local-models.md`](docs/local-models.md) — running against a local / open OpenAI-compatible model.
- [`docs/glossary.md`](docs/glossary.md) — the run-control vocabulary (tenant / task / run / job).
- [`docs/adr/`](docs/adr/) — architecture decision records (opaque verifier; Deep Agents sandbox;
  control-plane + ephemeral-worker provisioning).

---

The vendored specification is © 2026 Kynetyk Holdings LLC; all rights reserved.
