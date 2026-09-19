# Verity

A verifier-gated loop for autonomous agent work. An agent proposes; something that is not the agent
judges; the decision and the reasons behind it become durable, typed state you can query later.

Verity is an ongoing experiment in how to build agentic systems that run unattended and produce
results worth trusting. The name carries the point: *verity* = truth, and *verify*.

The kernel knows nothing about any particular problem. A task supplies a sandbox (the agent runtime)
and a verifier (the gates that judge its output), and both are extension points. Feature engineering
against a live Kaggle leaderboard is the application this was built and validated on, and it is one
application rather than the definition.

> **Status:** post-MVP; Phases 0–9 complete. See [ROADMAP.md](ROADMAP.md) for what remains.
> Stack: Python, managed with [uv](https://docs.astral.sh/uv/).

## The case for independent verification

An agent's own assessment of its work is not evidence about that work. Asked whether its output is
good, a model tends to say yes, and its stated confidence carries little information about whether it
is right. A loop running unattended on self-assessment compounds the problem, because every cycle
builds on work that only its author has vouched for.

The thing that decides whether work is accepted therefore has to be something with no stake in having
produced it. That single property is what makes the result of an unattended run worth anything. For
the result to be auditable as well as trustworthy, the decision and its reasons have to survive as
durable typed state rather than as a passage in a transcript, so that you can ask later why a given
artifact holds the status it holds.

The strength of the guarantee is the strength of the gate, which is why the gate is an extension
point: a deterministic check, a numeric scorer, a code runner in a clean container, an LLM judge, a
human, or a live competition leaderboard. The feature-engineering task uses a real Kaggle leaderboard
as its final test precisely because that is a judgement the system cannot talk itself into.

Four rules enforce this in the kernel:

- **No implicit accept.** An artifact type with no declared gate cannot be committed at all. Silence
  is never consent.
- **The proposer is never its own gate.** The sandbox proposes and writes nothing; the verifier
  judges and writes nothing; only the control plane mutates state.
- **Status is richer than accept/reject.** An artifact moves `proposed → tentative → accepted`, and
  can be `rejected`, `superseded` or `revised`. Cheap checks can only reach `tentative`; reaching
  `accepted` takes a hard gate.
- **`refine` recovers near-misses.** A mostly-sound artifact with a localized defect comes back with
  the defect named, instead of being thrown away.

The design originated in a natural-language specification, which is why `§N` citations appear
throughout the code and the ADRs. That document is kept in the sibling `NL-specs` repo.

## What's in the box

### The loop and the three services

The cycle is **read → propose → gate → commit**.

- **Control plane** — the only service that mutates durable state, and deliberately unintelligent. It
  owns the typed-provenance store, the commit path, the lifecycle, context assembly, the registries,
  the task config, loop control, and the invariant workspace contract plus composed system prompt.
- **Sandbox** — the agent runtime and its workspace. Ephemeral, regenerated each cycle. It proposes
  and emits objects to an outbox for harvest; it never writes durable state.
- **Verifier** — advisory, and opaque to the control plane. It receives a shaped proposal plus a
  declared slice of the store and returns a verdict. It never writes either.

### What it guarantees when left alone

These are the properties that make an unattended run survivable, and they are asserted as tests
rather than described as intentions.

- **A failed step is recorded, not fatal.** A sandbox crash, a hard timeout, or a verifier that goes
  away becomes a recorded failed cycle that is fed back to the agent, and the run continues.
- **Typed errors at every service and IO boundary**, distinguishing recoverable conditions from
  misuse. A raw `OSError` never escapes to abort a run.
- **Bounded-backoff retries** on transient external calls, with transient and fatal classified
  explicitly rather than by accident.
- **Atomic commits.** Decision rows and terminal status move together or not at all.
- **Invariants are property tests.** The rules above are checked against generated inputs, not only
  against examples.
- **A store-derived run report.** What a run did is reconstructed from durable state, so it survives a
  restart and cannot drift from what was actually committed.

### What it keeps separated

- Each cycle runs in a **fresh, capability-dropped worker container** that is destroyed afterwards.
- The **exchange and store volumes mount into the control-plane container only** and never reach a
  worker, so a worker cannot see another task's files, an export in flight, or an answer key.
- **Answer-key isolation holds by construction at the prep boundary.** You split the data into
  per-role bundles yourself; the control plane routes opaque role-keyed blobs and interprets no
  dataset semantics ([ADR 0005](docs/adr/0005-data-prep-out-of-the-control-plane.md)).
- **Credentials stay on the trusted verifier.** Kaggle submission happens on the verifier sibling;
  `KAGGLE_USERNAME` and `KAGGLE_KEY` never reach a worker.
- Each task instance gets **its own provenance store**, so two tasks of the same type cannot see each
  other's incumbents.

### What ships today

Read the live installed set with `verity catalog`; the current build carries:

| Task type | Gate | Mode |
|---|---|---|
| `fe-kaggle` | a cheap hold-out proxy, a competitive rung read from the live leaderboard, then a real Kaggle submission | optimizer |
| `fe-holdout` | runs-clean plus balanced accuracy on a reserved local hold-out | optimizer |
| `code` | parses and runs clean | first-acceptable |

**Sandboxes.** The harness is Deep Agents (a general-purpose coding agent in YOLO mode), in-process
and container/worker drivers, and it is pluggable ([ADR 0002](docs/adr/0002-sandbox-runtime-deep-agents.md)).
Models are selected per task through the `ModelSpec` seam: frontier Anthropic, any local or
self-hosted OpenAI-compatible endpoint, and hosted OpenAI. Ten models have been driven through the
full loop and scored, hosted (`gpt-5.4-mini`, `gpt-5.4-nano`, `haiku-4-5`, `sonnet-4-6`) and local
(`gpt-oss`, `gemma4`, `qwen`, `granite`, `glm-4.7-flash`, `fugu`).

**Verifiers.** Gate packages built from a primitive SDK: deterministic check, numeric scorer, LLM
judge, auto code runner, and human-in-the-loop. Gates stage cheap → `tentative`, hard → `accepted`
([ADR 0001](docs/adr/0001-opaque-verifier-and-control-plane-boundary.md)).

**Extending either end.** A new task type or verifier type ships as a container plus an installed
entry point, discovered at boot from the `verity.task_types` / `verity.verifier_types` groups. No
control-plane rebuild ([ADR 0006](docs/adr/0006-plugin-loader-entry-point-types.md)). The recipe,
with both snippets, is in the *Adding a task or verifier type* section of
[docs/api-surface.md](docs/api-surface.md).

### Acceptance modes

"What counts as done" is a property of how a task's verifier composes its gates, combined with the
run's orchestration policy and object-provisioning mode. There is no separate mode flag.

- **Optimizer.** The hard gate accepts only what beats the incumbent, and supersedes it, so one best
  answer survives. Pair with `LAST_ACCEPTED` provisioning so the agent iterates on the current best.
- **Accumulate.** The hard gate accepts on validity and never supersedes, so every sound artifact
  coexists as `accepted`. Pair with `ALL_ACCEPTED` provisioning.
- **First-acceptable.** A validity gate plus `--stop-on-accept` ends the run as soon as one artifact
  is accepted.

Supersession is entirely verifier-driven: the gate's verdict names what it replaces. A task type's
`verity catalog` entry describes its behaviour under `verifier_approach`. The fullest treatment,
including the provisioning presets and their explicit form, is in
[`.claude/skills/verity-run-task/references/concepts.md`](.claude/skills/verity-run-task/references/concepts.md).

## Recipes

### Develop

```
just install     # uv sync  (add `uv sync --extra sandbox` for the Deep Agents sandbox)
just check       # lint + type-check + test (must be green before any PR)
just test        # pytest
```

The suite runs offline by default. `@pytest.mark.docker` tests need Docker and `@pytest.mark.live`
tests need an API key; both auto-skip when their prerequisites are absent, so plain `just check`
needs neither.

### Exercise the loop without a model

The whole read → propose → gate → commit path on a deterministic fake runner, offline:

```
uv run --extra sandbox pytest tests/test_feature_engineering_acceptance.py
```

A real script executed in a container, scored end to end through the verifier's container runner
(needs Docker, no API key):

```
uv run --extra sandbox pytest tests/test_feature_engineering_gates.py -m docker
```

### Run a task on the standing daemon

The control plane runs as a long-lived service you configure at runtime. It launches sandbox and
code-runner worker containers as siblings on the host daemon, and you define tasks, ingest data, run,
and pull results without rebuilding an image ([ADR 0004](docs/adr/0004-long-lived-configurable-control-plane.md)).

```
just cp-serve                                     # build images + start the verity-cp daemon
just cp catalog                                   # task types and their published contracts
cp -r <role-bundles> "$VERITY_EXCHANGE_HOST/in/"  # prep is yours; e.g. agent/ + verifier/
just cp ingest agent/train.csv                    # -> a data handle (one per role file)
just cp create --request-file task.json --file agent:train.csv=<handle> …
just cp run <task_id>                             # -> a run id (runs in the background)
just cp results <run_id>                          # the RunReport + accepted-artifact ids
just cp export  <run_id>                          # artifacts -> $VERITY_EXCHANGE_HOST/out/<run_id>/
just cp-down                                      # stop the daemon (store + exchange persist)
```

Created tasks and their provenance stores live on the store volume, so a task survives a
`docker restart`. `just cp <subcommand>` is `docker exec verity-cp verity …`.

For the full CLI surface (every flag, the `TaskRequest` JSON shape, model targeting, pulling an agent
transcript) see
[`references/cli-reference.md`](.claude/skills/verity-run-task/references/cli-reference.md). For
first-time bring-up, the worker resource-limit knobs and a troubleshooting list, see
[`references/setup.md`](.claude/skills/verity-run-task/references/setup.md).

### Run it remotely

The same image serves an authenticated network API instead of the local socket, with bearer auth on
every route but `/health` and over-the-wire object upload and artifact download, so no shared volume
is needed:

```
VERITY_API_TOKEN=<secret> verity serve --http 0.0.0.0:8080
verity --url http://host:8080 --token <secret> catalog
```

### Drive it from Python

`docs/api-surface.md` carries a complete runnable `asyncio` example that configures a task on a
generic control plane, runs it, and reads back the accepted artifacts.

### Point it at a local model

Any OpenAI-compatible server works (Ollama, vLLM, llama.cpp, Docker Model Runner), reached at
`host.docker.internal`. [docs/local-models.md](docs/local-models.md) has per-server recipes and a
three-step curl pre-flight that checks reachability from the host, reachability from inside a
container, and whether the server actually emits structured `tool_calls`. Run the pre-flight before
blaming the harness.

### Run the Kaggle task

`fe-kaggle` gives the §12 feature-engineering domain a goal-seeking gate ladder: a cheap local
hold-out proxy filters every cycle, a competitive rung reads the live leaderboard and refines
anything whose calibrated estimate would not reach the top N% (feeding the gap back rather than
spending a submission), and the hard gate regenerates the submission on the full training set, submits
to the live competition, and accepts only what climbs the best public score. The ladder, the daily
submission cap and the credential handling are documented in the *fe-kaggle task type* section of
[docs/api-surface.md](docs/api-surface.md). Validated live on `playground-series-s6e6` at **0.92253**
balanced accuracy with a local model, best public score **0.93945**.

You prepare the data (`tools/prepare_fe_data.py`), then create the task with role-keyed files. The
runnable task package and its setup protocol are operator-local and versioned in the sibling
`verity-analysis` repo.

## Repo layout

```
README.md     this file
CLAUDE.md     orientation for coding agents working in this repo
ROADMAP.md    outstanding work (completed phases are summarized, not detailed)
.env.example  copy to .env for a live run
pyproject.toml / justfile   uv project + dev commands
Dockerfile    every image, one `--target` each, one shared build context:
                controlplane  the task-agnostic control-plane image (launches worker containers)
                verifier      the standing advisory-verifier service image (gates + kaggle extra)
                sandbox       the base image the container sandbox runs the agent in
                fe-sandbox    FROM sandbox + ML system libs; the FE tasks' agent image
                coderunner    the verifier's code-runner image (installs a submission's requirements.txt)
.dockerignore the single isolation boundary for all of them (no datasets, results or .env in any image)
infra/            compose files for the containerized topology (compose.daemon.yml — the standing daemon)
docs/             API reference, guides, and architecture decision records (see Docs below)
src/verity/
  contracts/      the cross-service value model + service ports (no service depends on another)
  control_plane/  the sole mutator: store, commit lifecycle, registries, context assembly, loop
  verifier/       the opaque verifier: gate-primitive SDK + the container/worker code-runner
  sandbox/        the real agent runtime (Deep Agents) — in-process + container/worker drivers
  domains/        per-domain wiring; feature_engineering.py is the FE domain
  composition/    applies a task to a generic control plane (the task catalog + per-task builders)
  provisioning/   the WorkerBackend seam + DockerBackend + the label-reaper (ADR 0003)
  eval/           the cross-model benchmark harness (quality / cost / latency)
  transport/      networked port-RPC (loopback + HTTP) for the service split
  logging.py / retry.py / telemetry.py   structured logging, bounded-backoff retries, metrics sink
tools/            prepare_fe_data.py (per-role input prep), render_transcript.py (transcript → Markdown)
tools/harness/    non-product test doubles + the dataset-split helper
tests/            test suite (offline by default; docker/live auto-skip)
```

## Docs

- [`docs/api-surface.md`](docs/api-surface.md) — the control-plane API reference, the daemon
  environment surface, a runnable programmatic example, and the plugin recipe.
- [`docs/local-models.md`](docs/local-models.md) — running against a local or self-hosted model.
- [`docs/glossary.md`](docs/glossary.md) — the run-control vocabulary (tenant / task / run / job).
- [`docs/adr/`](docs/adr/) — architecture decision records: the opaque verifier boundary (0001), the
  Deep Agents sandbox runtime (0002), control-plane + ephemeral-worker provisioning (0003), the
  long-lived configurable control plane (0004), data prep out of the control plane (0005), the
  entry-point plugin loader (0006), and the run-scoped verifier lifecycle (0007).
- [`.claude/skills/verity-run-task/`](.claude/skills/verity-run-task/) — the operator's guide: setup,
  the full CLI reference, and the concepts companion.
