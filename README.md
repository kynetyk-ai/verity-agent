# Verity

Verity is part of an ongoing experiment in how to build agentic systems that run unattended and produce
trustable results for knowledge work. The name comes from the same idea: *verity* = truth, and *verify*.
The prototype system works as a managed agent that works as part of an
integrated containerized system. The prototype system has three components:

- **Sandbox**: An isolated tool-using agent (currently built on the LangChain
  DeepAgents harness) inside a workspace where the agent is free to manipulate files, use tools, and
  compose its outputs for verification.
- **Verifier**: an isolated service that the agent cannot reach directly and which  provides
  validation of agent outputs according to a user-defined protocol.  
- **Control Plane**: A deliberately unintelligent service that spawns agent working sessions and their
  appropriate verification gates.  The control plane is the only service that mutates durable 
  state and context, based on the outputs from the **Verifier**.

Verity is designed to allow **Sandbox** agents to complete tasks in multiple modes across loops:
1) Work on a task until a single acceptable output proposal is received.
2) To optimize for the best output proposal across an arbitrary number of attempts
3) To accumulate distinct, acceptable proposals across an arbitrary number of attempts

The **Sandboxes** are deliberately ephemeral, and the agent in each works until it generates an
appropriately shaped proposal or hits its step budget or timeout.  The system places these ephemeral workers
in a loop that continues until an acceptable proposal is reached, a loop budget is reached, or both.
The workers' context includes prompt instructions as well as a filesystem state populated by the 
deterministic **Control Plane**.  This is intended to control what each run knows about previous runs,
provide information about past proposals the worker can reason from, and prevent faulty reasoning or
hallucinated information from an unacceptable proposal from poisoning future runs.


## The case for Verity's architecture

Most knowledge work isn't right or wrong; its 'correctness' depends in large part on being able
to describe the methodology that produced it along with a level of confidence that the methodology was 
actually applied.  This is particularly true in domains where the quality of an output can't be judged
by simple deterministic tests. However, even in coding, quality depends on things that go beyond simply
whether the code runs and the tests pass.

Modern agents are pretty good, and for individual use, you may never need anything like Verity.  However,
for trustable systems at scale in business settings, particularly those running relatively autonomously,
we need to accept certain realities about how the agents work and engineer around them:
1) Agents at some frequency are untruthful about the work they did and how they did it.
2) They can be subject to hallucination at one or many steps in a long-running process.
3) They may persist incorrect information, unhelpful reasoning, or frank hallucinations in their context, 
   whether it's memory systems, or file-based state, or simply the chat history, that can then potentially 
   poison all future steps the agent takes.
4) Because of agents' limited ability to weigh different kinds of context or certainty of assertions in context
   they are also at some frequency very poor judges of the quality of their own work.

Verity is a step towards that kind of engineering, designed to provide:
1) *Bounded Methodological Certainty*: at minimum, the steps taken to validate agents' work are guaranteed
   and isolated from the agents themselves.  The verification gate can also flexibly and optionally include
   assessment of the steps, recorded in the logs, that the agent took to do the work.  This means that 
   an output of a Verity worker can always be described accurately with the steps taken to verify the output
   and the level of certainty those steps provide.
2) *Selective Context Management*: How context evolves across loops is bounded and configurable. Under extreme selection,
   for example, the worker in a subsequent run may only ever see a single best output proposal with instructions to 
   improve upon it.
3) *Ephemeral Workers, Durable State*: one of the key premises of Verity is that the agent should not be a long-
   running friend, but rather a disposable process that can be swiftly pruned if and when it errs in detectable ways.

Verity does not guarantee correctness, and indeed, nothing can, but it does provide an extensible framework
that allows for methodological certainty and rational degrees of belief in outputs scoped by the methodology. To make this
more concrete, think of a coding agent fanning out sub-agents to run a deep research task.  Anyone who has used such 
a workflow understands (a) there are at least some inaccuracies or hallucinations in the outputs; (b) the frequency
of such inaccuracies or hallucinations and the degree to which they may affect conclusions is almost impossible to assess
without checking everything; (c) any verification that may be done is entirely subject to what the orchestrating agent
decides to do or not.  While those workflows are useful (we use them all the time) that kind of uncertainty, from
unaccountable actors, cannot be acceptable within industrial-grade, trustable systems.

## What's in the box

Verity is a work in progress that is designed to be extensible.  We do not consider it production-ready, and it may never 
be developed to be so in its current form. Nonetheless, in our testing it has run stably for long periods including hundreds of runs on an initial prototyping domain, which so far has allowed us to run our experiments. Read the live installed set with `verity catalog`; the current build carries:

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
AGENT.md      orientation for coding agents working in this repo
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

## Contributing

You are welcome to fork and use Verity under the terms of the license, and to get in touch if you
are interested in the project. If you find something worth pointing out, feel free to open an issue.
We are not actively maintaining the project for third-party use.

## License

MIT. See [LICENSE.md](LICENSE.md).
