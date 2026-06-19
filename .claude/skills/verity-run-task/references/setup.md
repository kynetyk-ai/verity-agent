# Setup — standing up the Verity daemon (from a fresh clone)

The control plane runs as a long-lived container (`verity-cp`) that launches sibling worker containers
on the host Docker daemon. This is the deployment the `verity` CLI talks to.

> **Already have a running daemon?** Skip this whole page. The runtime flow in `SKILL.md`
> (`docker exec <container> verity …`) drives an existing daemon and needs **nothing from the Verity
> source repo**. This page covers *bringing one up*, which **does require the repo** (it builds the
> images and uses the compose/justfile). If you only have the running container, jump to the CLI.

## Prerequisites

- **Docker** running (the CP launches sibling workers via the mounted `/var/run/docker.sock`).
- **A model**, one of:
  - **Local, free:** an OpenAI-compatible server on the host (e.g. Ollama on `:11434`) serving a coding
    model. Workers reach it at `http://host.docker.internal:11434/v1`. Confirm with
    `curl -s http://localhost:11434/api/tags`.
  - **Hosted (Anthropic):** an `ANTHROPIC_API_KEY` (put it in `.env`; `cp .env.example .env`).
- **`uv`** + **`just`** for local dev convenience (optional — raw `docker compose` equivalents below).

## Build the two images (once)

```bash
docker build -f Dockerfile.sandbox      -t verity-sandbox:latest .   # the agent worker
docker build -f Dockerfile.coderunner   -t verity-code-runner:latest .  # the gate's code-runner (system libs for the CPU ML stack)
docker build -f Dockerfile.controlplane -t verity-controlplane:latest .  # the daemon (carries the docker CLI + service extra)
```

`just cp-serve` builds both for you.

## Bring up the daemon

**With `just` (recommended):**

```bash
just cp-serve        # builds images, makes host dirs, `docker compose -f infra/compose.daemon.yml up -d`
just cp catalog      # smoke: lists installed task types
just cp-down         # stop (volumes/exchange persist on the host)
```

**Raw `docker compose` (no `just`):**

```bash
export VERITY_EXCHANGE_HOST=/tmp/verity-exchange VERITY_STORE_HOST=/tmp/verity-store VERITY_WORKER_STAGING=/tmp/verity-staging
mkdir -p "$VERITY_EXCHANGE_HOST/in" "$VERITY_EXCHANGE_HOST/out" "$VERITY_STORE_HOST" "$VERITY_WORKER_STAGING"
docker compose -f infra/compose.daemon.yml up -d
docker exec verity-cp verity catalog
docker compose -f infra/compose.daemon.yml down
```

## Configuration (environment)

Set on the host before `up` (the compose file reads them; defaults in parentheses):

| Var | Purpose |
|---|---|
| `VERITY_EXCHANGE_HOST` (`/tmp/verity-exchange`) | Host dir bind-mounted to `/exchange` in the CP. Put inputs in `in/`, find outputs in `out/`. CP-only — never a worker. |
| `VERITY_STORE_HOST` (`/tmp/verity-store`) | Host dir for the persistent per-task stores + task definitions (durable across restart). CP-only. |
| `VERITY_WORKER_STAGING` (`/tmp/verity-staging`) | Shared staging dir bind-mounted at an **identical host↔container path** so worker bind-mount sources resolve on the host daemon. |
| `VERITY_MODEL` (`qwen3.6:27b-coding-mxfp8`) | Default model for the compose example; per-task you pass `--model` at `verity create`. |
| `VERITY_LOCAL_BASE_URL` (`http://host.docker.internal:11434/v1`) | Local OpenAI-compatible endpoint the workers reach. |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Forwarded to the agent worker when you select a hosted model. Empty for a local model. |

Note: `VERITY_MODEL`/`VERITY_LOCAL_BASE_URL` in compose are conveniences; **the model a run actually
uses is the one you pass to `verity create`** (`--model` / `--base-url`).

## Moving data in and out

- **In:** copy your file into `<VERITY_EXCHANGE_HOST>/in/`, then `verity ingest <filename>` → a content
  handle you pass to `verity create --data <handle>`.
- **Out:** `verity export <run_id>` writes the run's durable artifacts (e.g. the accepted submission's
  code + each feature) to `<VERITY_EXCHANGE_HOST>/out/<run_id>/`.

## Troubleshooting

- **Worker bind-mount / "no such file" on the host daemon** — `VERITY_WORKER_STAGING` must be the
  **same absolute path** inside and outside the container (it is in the compose default). The CP creates
  staging dirs that the *host* daemon resolves.
- **Exported files are root-owned** — the daemon runs as root in the container, so files written to the
  exchange bind mount are root-owned on the host. `chown` them back, or run with a matched uid.
- **Local model "not found" / a mangled model name** — pass the **full** model tag *with* `--base-url`
  (e.g. `--model qwen3.6:27b-coding-mxfp8 --base-url http://host.docker.internal:11434/v1`). Without a
  `--base-url`, `--model` is parsed as a native `provider:model` string.
- **Daemon unreachable** — confirm the container is up (`docker ps | grep verity-cp`) and the socket
  exists (`docker exec verity-cp test -S /run/verity.sock`).
- **Hosted model unauthorized** — ensure `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) is in the daemon
  container's env (set on the host before `up`).

## Kaggle creds (only for the `fe-kaggle` task)

The `fe-kaggle` task type submits to a live competition, so its gate needs a Kaggle API token:

- Create one at <https://www.kaggle.com/settings> → *Create New Token* (downloads `kaggle.json`); put
  its `username`/`key` in the daemon container's env as `KAGGLE_USERNAME` / `KAGGLE_KEY` (the daemon
  compose forwards them to the control plane **only** — never to a worker).
- **Accept the competition's rules** on its Kaggle page once, or the API returns **403** on that
  competition (auth can otherwise be fine — a 401 elsewhere means a bad token).
- The cap is the competition's own daily submission limit (read from its Kaggle metadata, typically
  ~5/day per team); the gate reads remaining budget from the API and blocks until it frees. Full package + protocol: the task's own `PROTOCOL.md` (in the Verity repo, under
  `feature-engineering-test/`).

---

The control-plane API reference (the full env surface, the worked example, the topology) lives in the
Verity repo at `docs/api-surface.md` (the *standing daemon* + *External HTTP/REST* sections). The
authoritative, never-stale source for what's installed is always `verity catalog`.
