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

# or a data-bearing FE task: `fe-kaggle` (real public leaderboard, needs KAGGLE_* creds) or
# `fe-holdout` (local reserved hold-out, NO creds — the ablation task; verifier role is a subset:
# train.csv + holdout.csv + holdout_labels.csv). Data prep is YOURS (ADR 0005): split into per-role
# bundles outside Verity with `uv run python -m tools.prepare_fe_data …` (run as a MODULE). Then put
# the bundles under the daemon's exchange in/ dir, ingest each, and create with `--file ROLE:NAME=HANDLE`:
docker exec $CP verity ingest agent/train.csv        # -> {"handle": "<h1>"} (subdir paths allowed)
docker exec $CP verity ingest verifier/holdout_labels.csv  # -> {"handle": "<h2>"} (answer key, verifier-only)
# ...ingest every agent/* and verifier/* file, then (fe-holdout shown; --request-file or flags both work):
docker exec $CP verity create --type fe-holdout \
  --model qwen3.6:27b-coding-mxfp8 --base-url http://host.docker.internal:11434/v1 \
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
# point the package's own run.sh at them (prototyping_datasci_test/run.sh does the prep for you).
```

## Guardrails (read before running)

- **Local / OpenAI-compatible model:** pass the **full** name with `--base-url`, e.g.
  `--model gpt-oss:20b --base-url http://host.docker.internal:11434/v1`. The colon belongs to the model
  tag — `--base-url` is what makes it a literal OpenAI-compatible name (do not omit it for a local
  model). Keyless local servers need no key; a keyed endpoint takes `api_key_env` (the env var name).
- **Native provider model:** `--model anthropic:claude-sonnet-4-6` or `--model openai:gpt-5.4-nano`
  (no `--base-url`); the daemon forwards `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` to the worker, so the
  matching key must be set in the daemon's environment.
- See **cli-reference.md → "Model targeting"** for the full matrix (native / local / hosted
  OpenAI-compatible), keys, and the `host.docker.internal` gateway behavior.
- **Files only via the exchange.** Inputs go in `<exchange>/in/`; outputs land in `<exchange>/out/<run_id>/`.
  Nothing else reaches a worker.
- **Network surface needs auth.** `verity serve --http HOST:PORT` requires `VERITY_API_TOKEN`; clients
  pass `--url`/`--token`. The default Unix-socket (`docker exec`) path is local-only and unauthenticated.
- **Degrade-don't-crash.** A failed cycle is recorded and fed back, not fatal — a run can finish with a
  mix of accept / refine / reject / sandbox-fail outcomes (see `results`).
- **Pulling a transcript / object by hash.** `results` now carries a per-cycle `transcript_ref` (a
  content hash — the agent's step-by-step record: message types, content, tool calls + args,
  results). There is **no CLI/HTTP verb for a bare object by hash yet**; read it from the
  **per-task** object store (you need the `task_id`, not just the run):
  `docker exec verity-cp cat /var/lib/verity/tasks/<task_id>/objects/<hash>` (or host
  `${VERITY_STORE_HOST:-/tmp/verity-store}/tasks/<task_id>/objects/<hash>`). `export` only writes
  *declared* artifact objects, not the transcript. Pipe the JSON through `python tools/render_transcript.py -`
  (repo checkout) for a readable Markdown view.
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
