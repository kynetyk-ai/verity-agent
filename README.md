# Verity

**An implementation of the Self-Revising Discovery Harness** — a domain-agnostic kernel that gives an
agent a *typed provenance record* (artifacts, the operations that produced them, and the gate
decisions about them) as its durable state, behind a verifier-aware commit lifecycle. The shorthand
is *git + a type system + a verifier-aware lifecycle for agent artifacts.*

The name carries the point: **verity** = truth, and *verify*. The system exists to answer one
question — *can you trust, and audit, what the system believes?*

> **Status: post-MVP; Phase 9 complete (the dumb control plane).** Phases 0–5 are complete — the
> control plane, the opaque verifier, the real Deep Agents sandbox, and the feature-engineering domain
> (spec §12) reached MVP with the **twelve §13 acceptance criteria passing**, plus the Phase 5
> reliability & observability hardening. **Phase 6 (model breadth)** is code-complete and live-validated
> on both a local open model and hosted OpenAI (`good proposals from cheap models` — the thesis).
> **Phase 7 (service split & full containerization)** reached its done-line. **Phase 8 (the long-lived,
> configurable control-plane service)** is **v1 complete**: one image serves a standing daemon over a
> local Unix socket **and** an authenticated (bearer-token) network HTTP API, configured + run via the
> `verity` CLI with no image rebuild and durable across restart. **Phase 9 (the dumb control plane)** is
> complete: the verifier runs as a **sibling container**, **data prep is user-side** (the CP routes
> opaque role-keyed blobs — [ADR 0005](docs/adr/0005-data-prep-out-of-the-control-plane.md)), and task
> & verifier types are **discovered via entry-point plugins**
> ([ADR 0006](docs/adr/0006-plugin-loader-entry-point-types.md)) — so a new type ships a container + an
> installed entry point with **no control-plane rebuild**. A **`fe-kaggle`** variant has been validated
> **live on the real Kaggle leaderboard** (best public score **0.93945**). See
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
to write a single self-contained script that predicts better than the incumbent — by any means that
fits in one script (engineered features, model choice, ensembling, calibration), judged by running it
on a **reserved hold-out it never sees** (a per-class, deterministic split of the training data; the
agent gets the labelled remainder, the gate keeps the reserved labels). The submission is the unit and
the gate is reject-only. Several ways to exercise it, in increasing cost — and the task you actually
run (#5), whose gate is the **real Kaggle leaderboard**:

**1. The §13 acceptance criteria (offline — no key, no Docker).** The whole
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
#    at prototyping_datasci_test/{train.csv,test.csv}  (gitignored)

# c. build the sandbox image the agent runs in (once)
docker build -f Dockerfile.sandbox -t verity-sandbox:latest .

# d. run it (loads .env into the environment first)
set -a; . ./.env; set +a
uv run --extra sandbox pytest tests/test_feature_engineering_acceptance.py::test_feature_engineering_live -s
```

Each cycle spins a fresh, isolated container; expect a few minutes per round. The agent is steered
**only** through the control plane (task + domain instructions and the mounted data) — nothing
task-specific is hardcoded in the sandbox.

**4. The standing daemon (Phase 8 — one container, many tasks, no rebuild).** Run the control plane
as a **long-lived service** you configure at runtime: a task-agnostic control plane runs *in* a
container and launches the sandbox + code-runner **worker containers** as siblings on the host daemon;
you define tasks, ingest data, run, and pull results — all without rebuilding an image (see
[ADR 0004](docs/adr/0004-long-lived-configurable-control-plane.md)).

```
just cp-serve                                   # build images + start the verity-cp daemon
just cp catalog                                 # list task types and their published contracts
cp -r <role-bundles> "$VERITY_EXCHANGE_HOST/in/"  # prep is yours (ADR 0005); e.g. agent/ + verifier/
just cp ingest agent/train.csv                   # -> a data handle (ingest each role file separately)
just cp create --request-file task.json --file agent:train.csv=<handle> …  # route role files -> a task id
just cp run <task_id>                            # -> a run id (runs in the background)
just cp results <run_id>                         # the RunReport + accepted-artifact ids
just cp export  <run_id>                         # durable artifacts -> $VERITY_EXCHANGE_HOST/out/<run_id>/
just cp-down                                     # stop the daemon (the store + exchange persist)
```

Files cross only through the **exchange** (in/ and out/); the exchange and the persistent **store**
volume are mounted into the control-plane container alone and **never reach a worker**
(`tests/test_daemon_volume_isolation.py`). Task definitions and provenance stores live under the
store volume, so a created task survives a `docker restart`. The `verity` CLI is a thin client over a
Unix socket inside the container — `just cp <subcommand>` is `docker exec verity-cp verity …`.

For **remote / programmatic** use, the same image serves an **authenticated network API** instead of
the local socket — `VERITY_API_TOKEN=<secret> verity serve --http 0.0.0.0:8080` — with bearer auth on
every route but `/health`, and over-the-wire object upload + artifact download (no shared volume).
Reach it with the same CLI: `verity --url http://host:8080 --token <secret> catalog`. See the
*External HTTP/REST* section of [docs/api-surface.md](docs/api-surface.md).

**5. The `fe-kaggle` task — the real Kaggle leaderboard as the final-test gate (needs Docker, a
model, a Kaggle token).** The §12 FE domain with a **goal-seeking gate**: a cheap local-hold-out
proxy filters every cycle, then a **competitive** rung reads the live leaderboard and only submits an
attempt whose calibrated hold-out estimate would reach the top-N% (below the bar it **refines** — the
gap fed back, no submission spent); the hard gate then regenerates the submission on the full train +
the real `test.csv`, **submits to a live Kaggle competition**, and accepts only what climbs our best
**public-leaderboard** score (the competition's daily submission cap is read from its Kaggle
metadata — default 5; the gate blocks until budget frees). The submit happens on the trusted verifier
sibling — `KAGGLE_USERNAME`/`KAGGLE_KEY` never reach a worker. It's a new task type in the catalog
(`verity catalog --type fe-kaggle`). **You prepare the data** (ADR 0005): split it into per-role
bundles outside Verity (`tools/prepare_fe_data.py`), then create with role-keyed
`verity create --request-file task.json --file agent:train.csv=<h> --file verifier:holdout_labels.csv=<h> …`
— the control plane routes opaque blobs and interprets no dataset semantics. The committed, runnable
package — `task.json` (local-agent default), `run.sh` (which runs the prep for you), and the full
setup protocol (token, accepting the competition rules) — lives in
[`prototyping_datasci_test/PROTOCOL.md`](prototyping_datasci_test/PROTOCOL.md). Validated live on
`playground-series-s6e6` with a local model at **0.92253 balanced accuracy**.

## Acceptance modes

"What counts as done" is not one fixed behaviour — it's a property of how a task's **verifier composes
its gates**, combined with the run's **orchestration policy** and **object-provisioning** mode. Three
shapes are supported (no separate "mode" flag — they fall out of those choices):

- **Optimizer — supplant the incumbent with a better one.** The hard gate accepts only when the new
  artifact *beats* the incumbent and **supersedes** it, so one best answer survives. This is the FE /
  `fe-kaggle` behaviour (a scoring gate that emits a supersession). Pair with `LAST_ACCEPTED`
  provisioning so the agent iterates on the current best.
- **Accumulate — keep every acceptable answer.** The hard gate accepts on *validity* and never
  supersedes, so all sound artifacts coexist as `accepted`. Pair with `ALL_ACCEPTED` provisioning.
  Use it when you want a *collection* of good answers, not a single winner.
- **First-acceptable — take the first good one and stop.** A validity gate plus `--stop-on-accept`
  (`OrchestrationPolicy.stop_on_accept`) ends the run as soon as one artifact is accepted. Use it when
  any sound answer is enough and there's no value in optimizing further.

The load-bearing facts: only a **hard** gate reaches `accepted` (cheap checks rest at `tentative`), and
**supersession is entirely verifier-driven** (the gate's verdict names what it replaces). So a task
chooses its mode by how its verifier is built; `--stop-on-accept` is the per-run CLI knob. A task type's
`verity catalog` entry describes its acceptance behaviour under `verifier_approach`.

## Sandboxes and verifiers (what ships today)

A task **selects a sandbox and a verifier**; the control plane stays generic. What's in the current
build (read the live set + each one's published contract with `verity catalog`):

**Sandboxes** — the agent runtime + the model it runs:
- **Harness:** Deep Agents (a general-purpose coding agent in YOLO mode), with in-process and
  container/worker drivers. The harness is **pluggable** (ADR 0002) — Deep Agents is the first one.
- **Model arms** (selected per task with `--model`/`--base-url`, via the `ModelSpec` seam): frontier
  **Anthropic** (`anthropic:claude-…`), any **local / self-hosted OpenAI-compatible** endpoint
  (`local_spec` — Ollama / vLLM / llama.cpp, reached at `host.docker.internal`), and **hosted OpenAI**
  (`openai_spec`). Non-propose executable tools can be bound into the agent (the `read_pdf` seam).

**Verifiers** — the opaque gate package that judges a proposal (ADR 0001), built from a gate-primitive
SDK (deterministic-check, numeric-scorer, LLM-judge, auto-code-runner, human-in-the-loop) staged
cheap → `tentative`, hard → `accepted`. Task types shipping today:
- **`fe-holdout`** — feature engineering: a runs-clean check + a balanced-accuracy selection gate on
  a reserved hold-out (optimizer mode).
- **`fe-kaggle`** — the same domain, but the authoritative gate is the **real Kaggle leaderboard**.
- **`code`** — a minimal code-execution task (parses + runs-clean).

(Prefer `verity catalog` for the live installed set.)

**Extensibility:** both are pluggable extension points — you can add a sandbox arm (or a whole harness)
and a verifier approach without touching the kernel. **A how-to guide for adding sandboxes and verifiers
is forthcoming** ([#66](https://github.com/kynetyk-ai/verity/issues/66)); until then, the FE / `code`
composition builders (`src/verity/composition/`) are the worked references.

## Repo layout

```
README.md     this file
CLAUDE.md     orientation for coding agents working in this repo
ROADMAP.md    the living path (now: post-MVP backlog)
.env.example  copy to .env for the live demo
pyproject.toml / justfile   uv project + dev commands
Dockerfile.sandbox          the base image the container sandbox runs the agent in
Dockerfile.fe-sandbox       FROM verity-sandbox + ML system libs; the FE tasks' agent image (Option A)
Dockerfile.verifier         the standing advisory-verifier service image (Phase 7.1)
Dockerfile.coderunner       the verifier's code-runner image (installs a submission's requirements.txt)
Dockerfile.controlplane     the task-agnostic control-plane image (launches worker containers)
infra/            compose files for the containerized topology (compose.daemon.yml — the standing daemon)
docs/             API reference, guides, and architecture decision records (see Docs below)
src/verity/
  contracts/      the cross-service value model + service ports (no service depends on another)
  control_plane/  the sole mutator: store, commit lifecycle, registries, context assembly, loop
  verifier/       the opaque verifier: gate-primitive SDK + the container/worker code-runner
  sandbox/        the real agent runtime (Deep Agents) — in-process + container/worker drivers
  domains/        per-domain wiring; feature_engineering.py is the §12 MVP domain
  composition/    applies a task to a generic control plane (the task catalog + per-task builders)
  provisioning/   the WorkerBackend seam + DockerBackend + the label-reaper (ADR 0003)
  eval/           the cross-model benchmark harness (quality / cost / latency)
  transport/      networked port-RPC (loopback + HTTP) for the service split
  logging.py / retry.py / telemetry.py   structured logging, bounded-backoff retries, metrics sink
tools/            prepare_fe_data.py (per-role input prep), render_transcript.py (transcript → Markdown)
tools/harness/    non-product test doubles + the dataset-split helper
tests/            test suite (offline by default; docker/live auto-skip)
spec/             vendored specification + references (see above)
```

## Docs

Reference material lives in [`docs/`](docs/):

- [`docs/api-surface.md`](docs/api-surface.md) — the **control-plane API reference**, with a worked
  example of applying a task to a generic control plane and the daemon CLI flow.
- [`docs/local-models.md`](docs/local-models.md) — running against a local / open OpenAI-compatible model.
- [`docs/glossary.md`](docs/glossary.md) — the run-control vocabulary (tenant / task / run / job).
- [`docs/adr/`](docs/adr/) — architecture decision records (opaque verifier; Deep Agents sandbox;
  control-plane + ephemeral-worker provisioning).

---

The vendored specification is © 2026 Kynetyk Holdings LLC; all rights reserved.
