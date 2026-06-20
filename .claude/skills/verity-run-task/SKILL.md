---
name: verity-run-task
description: >-
  Drive the Verity control-plane daemon with the `verity` CLI to run a discovery task end to end —
  stand up the daemon, discover installed task types, ingest data, create a task, run it, and pull
  back results + durable artifacts. Use when the user wants to run a Verity task, use the verity CLI,
  spin up or operate the Verity control plane, run the feature-engineering (FE) task, or drive the
  standing daemon.
allowed-tools: Bash(docker *) Bash(just *)
---

# Run a Verity task via the CLI

## Mental model

Verity runs as **one standing container** (`verity-cp`) — the control-plane **daemon**. The `verity`
CLI is a **thin client** over that daemon's socket; you invoke it with `docker exec verity-cp verity …`
(or `just cp …`). The daemon launches ephemeral **worker** containers per cycle (the agent sandbox, and
a code-runner for the verifier). Data crosses **only** through the *exchange* directory (`in/` and
`out/`) — never straight into a worker. Runs are **asynchronous**: `run` returns a `run_id` immediately;
you poll `status` until it's terminal, then read `results` and `export` artifacts.

## Step 0 — discover what's installed (do this first)

**Never assume a fixed set of task types or verifiers — read the live catalog.** Run:

```bash
docker exec verity-cp verity catalog                  # all installed task types
docker exec verity-cp verity catalog --type fe-kaggle # one type's full published contract
```

Each entry self-describes its **output-shape contract** (the artifact types / operations the agent must
produce + the domain instructions), its **`verifier_approach`** (how acceptance is judged — i.e. *which
verifier* you're getting and what "accepted" means), and its **`sandbox_notes`** (model / execution
caveats). Read this **before** `create` so you pick a fitting task type, model, and goal. The catalog is
the source of truth; this skill deliberately does not hardcode the list.

## The flow at a glance

Every verb is `docker exec <container> verity <verb>` against the running daemon — **this works with
only the running container; you do NOT need the Verity source repo** (`$CP` defaults to `verity-cp`):

```bash
docker exec $CP verity catalog                       # step 0: discover task types + contracts
# the trivial, data-less `code` task (flag-shaped request, no creds):
docker exec $CP verity create --type code \
  --model anthropic:claude-sonnet-4-6 --max-cycles 2 --stop-on-accept   # -> {"task_id": "..."}

# or a data-bearing task like `fe-kaggle`. Data prep is YOURS (ADR 0005): you split it into
# per-role bundles outside Verity (the CP routes opaque blobs). Ingest each role file, then create
# with `--file ROLE:NAME=HANDLE`. Put the files under the daemon's exchange in/ dir, then:
docker exec $CP verity ingest agent/train.csv        # -> {"handle": "<h1>"} (subdir paths allowed)
docker exec $CP verity ingest verifier/holdout_labels.csv  # -> {"handle": "<h2>"} (answer key, verifier-only)
# ...ingest every agent/* and verifier/* file, then:
docker exec $CP verity create --request-file /exchange/in/task.json \
  --file agent:train.csv=<h1> --file verifier:holdout_labels.csv=<h2> ...  # repeatable per file

docker exec $CP verity run <task_id>                 # -> {"run_id": "..."}  (async)
docker exec $CP verity status <run_id>               # poll until "done" / "failed"
docker exec $CP verity results <run_id>              # RunReport + accepted-artifact ids
docker exec $CP verity export <run_id>               # durable artifacts -> exchange out/<run_id>/
```

**Bringing the daemon up needs the repo** (it builds images + uses compose); see
[references/setup.md](references/setup.md). With a repo checkout, `just cp <verb>` is shorthand for
`docker exec verity-cp verity <verb>` and `just cp-serve` starts it. Or run the whole sequence with
the bundled orchestrator (daemon must already be up):

```bash
bash ${CLAUDE_SKILL_DIR}/scripts/run_task.sh --type code --goal "write a script that runs"
# data-bearing tasks, e.g. fe-kaggle: prepare role bundles first (tools/prepare_fe_data.py), then
# point the package's own run.sh at them (feature-engineering-test/run.sh does the prep for you).
```

## Guardrails (read before running)

- **Local model name:** pass the **full** name with `--base-url`, e.g. `--model qwen3.6:27b-coding-mxfp8
  --base-url http://host.docker.internal:11434/v1`. The colon belongs to the model tag — `--base-url` is
  what makes it a literal OpenAI-compatible name (do not omit it for a local model).
- **Anthropic model:** `--model anthropic:claude-sonnet-4-6` (no `--base-url`); the daemon forwards
  `ANTHROPIC_API_KEY` to the worker, so it must be set in the daemon's environment.
- **Files only via the exchange.** Inputs go in `<exchange>/in/`; outputs land in `<exchange>/out/<run_id>/`.
  Nothing else reaches a worker.
- **Network surface needs auth.** `verity serve --http HOST:PORT` requires `VERITY_API_TOKEN`; clients
  pass `--url`/`--token`. The default Unix-socket (`docker exec`) path is local-only and unauthenticated.
- **Degrade-don't-crash.** A failed cycle is recorded and fed back, not fatal — a run can finish with a
  mix of accept / refine / reject / sandbox-fail outcomes (see `results`).
- **Data prep is the user's, not the control plane's** (ADR 0005 / #74). You split inputs into
  per-role bundles *outside* Verity; the CP routes opaque, role-keyed blobs and interprets none of
  them (no `target`/`id_column`/split). Multi-input tasks (e.g. `fe-kaggle`) are created from a JSON
  request plus repeated `--file ROLE:NAME=HANDLE` flags (one per ingested file); `--request-file`
  reads a full `TaskRequest`. See the CLI ref.

## More detail

- **Stand it up / from a fresh clone (needs the repo)** → [references/setup.md](references/setup.md)
- **Full CLI surface** (every verb, flags, JSON output, the catalog self-description, the network/auth
  mode, the byte data plane) → [references/cli-reference.md](references/cli-reference.md)
- **What Verity is & where it generalizes** + **acceptance modes** (optimizer / accumulate /
  first-acceptable — what each is for and how to set it up) → [references/concepts.md](references/concepts.md)
