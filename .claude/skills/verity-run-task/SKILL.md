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
docker exec verity-cp verity catalog            # all installed task types
docker exec verity-cp verity catalog --type fe  # one type's full published contract
```

Each entry self-describes its **output-shape contract** (the artifact types / operations the agent must
produce + the domain instructions), its **`verifier_approach`** (how acceptance is judged — i.e. *which
verifier* you're getting and what "accepted" means), and its **`sandbox_notes`** (model / execution
caveats). Read this **before** `create` so you pick a fitting task type, model, and goal. The catalog is
the source of truth; this skill deliberately does not hardcode the list.

## The flow at a glance

```bash
just cp-serve                               # build images + start the daemon (see references/setup.md)
just cp catalog                             # step 0: discover task types + their contracts
# put your input file in the exchange in/ dir, then:
just cp ingest train.csv                    # -> {"handle": "<sha>"}
just cp create --type fe --data <handle> \
  --model qwen3.6:27b-coding-mxfp8 \
  --base-url http://host.docker.internal:11434/v1 \
  --max-cycles 4 --per-class 150            # -> {"task_id": "..."}
just cp run <task_id>                        # -> {"run_id": "..."}  (async)
just cp status <run_id>                      # poll until "done" / "failed"
just cp results <run_id>                     # RunReport + accepted-artifact ids
just cp export  <run_id>                     # durable artifacts -> exchange out/<run_id>/
```

Or run the whole sequence with the bundled orchestrator (daemon must already be up):

```bash
bash ${CLAUDE_SKILL_DIR}/scripts/run_task.sh --type fe --data train.csv --max-cycles 4 --per-class 150
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

## More detail

- **Stand it up / from a fresh clone** → [references/setup.md](references/setup.md)
- **Full CLI surface** (every verb, flags, JSON output, the catalog self-description, the network/auth
  mode, the byte data plane) → [references/cli-reference.md](references/cli-reference.md)
- **What Verity is & where it generalizes** (the domain-agnostic kernel; spec/paper-grounded) →
  [references/concepts.md](references/concepts.md)
