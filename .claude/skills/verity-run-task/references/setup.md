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

## Build the images (once)

```bash
docker build -f Dockerfile.sandbox      -t verity-sandbox:latest .   # the agent worker (runtime only)
docker build -f Dockerfile.fe-sandbox   -t verity-fe-sandbox:latest .  # base + ML system libs (libgomp…) — FE tasks default to this so the agent can install reqs + self-test (build AFTER the base)
docker build -f Dockerfile.coderunner   -t verity-code-runner:latest .  # the gate's code-runner (system libs for the CPU ML stack)
docker build -f Dockerfile.verifier     -t verity-verifier:latest .  # the verifier sibling (gates + kaggle extra; §9.1)
docker build -f Dockerfile.controlplane -t verity-controlplane:latest .  # the daemon (carries the docker CLI + service extra)
```

`just cp-serve` builds all of them for you. (The `code` task uses the base `verity-sandbox`; `fe-holdout`/`fe-kaggle` use `verity-fe-sandbox`.)

> **credsStore hang at build:** if a `docker build` hangs reaching a registry, Docker Desktop's
> credential helper is the culprit — build with a throwaway empty config:
> `DOCKER_CONFIG="$(mktemp -d)" docker build …` (or `DOCKER_CONFIG="$(mktemp -d)" just cp-serve`).

## Bring up the daemon

**With `just` (recommended):**

```bash
just cp-serve        # builds images, makes host dirs, `docker compose -f infra/compose.daemon.yml up -d`
just cp catalog      # smoke: lists installed task types
just cp-down         # stop (volumes/exchange persist on the host)
```

**Raw compose (no `just`):**

> **Compose binary:** `just cp-serve` and these examples assume the **Compose V2 plugin**
> (`docker compose`). If your host only has the **standalone** `docker-compose` binary
> (`docker compose` → "unknown command"), substitute `docker-compose` in every command below — it
> behaves the same. (`just cp-serve` hardcodes `docker compose`; if you only have the standalone
> binary, run the steps by hand or edit the recipe.)

```bash
export VERITY_EXCHANGE_HOST=/tmp/verity-exchange VERITY_STORE_HOST=/tmp/verity-store VERITY_WORKER_STAGING=/tmp/verity-staging
mkdir -p "$VERITY_EXCHANGE_HOST/in" "$VERITY_EXCHANGE_HOST/out" "$VERITY_STORE_HOST" "$VERITY_WORKER_STAGING"
# --env-file .env: without it, compose resolves ${KAGGLE_*} against infra/.env (absent) and the
# daemon launches with EMPTY Kaggle creds — fine for `code`/`fe-holdout`, but `fe-kaggle` then fails
# only later at the submission gate. Harmless to always pass it.
docker compose --env-file .env -f infra/compose.daemon.yml up -d   # or: docker-compose --env-file .env -f …
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

- **In:** copy your (already-prepared) input files into `<VERITY_EXCHANGE_HOST>/in/` — subdirs allowed
  for role bundles — then `verity ingest <path>` → a content handle you route with
  `verity create --file ROLE:NAME=<handle>` (data prep is yours, ADR 0005; the CP routes opaque blobs).
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

**Creds matrix:** `code` and `fe-holdout` need **no** credentials (a local model needs no API key
either); only `fe-kaggle` needs Kaggle creds (it submits to the live leaderboard). A hosted *model*
(Anthropic/OpenAI) needs its API key in the daemon env regardless of task type.

The `fe-kaggle` task type submits to a live competition, so its gate needs a Kaggle API token:

- Create one at <https://www.kaggle.com/settings> → *Create New Token* (downloads `kaggle.json`); put
  its `username`/`key` in the daemon container's env as `KAGGLE_USERNAME` / `KAGGLE_KEY` (the daemon
  compose forwards them to the control plane **only** — never to a worker).
- **Accept the competition's rules** on its Kaggle page once, or the API returns **403** on that
  competition (auth can otherwise be fine — a 401 elsewhere means a bad token).
- The cap is the competition's own daily submission limit (read from its Kaggle metadata, typically
  ~5/day per team); the gate reads remaining budget from the API and blocks until it frees. Full package + protocol: the task's own `PROTOCOL.md` (in the Verity repo, under
  `results/prototyping_datasci_test/`).

---

The control-plane API reference (the full env surface, the worked example, the topology) lives in the
Verity repo at `docs/api-surface.md` (the *standing daemon* + *External HTTP/REST* sections). The
authoritative, never-stale source for what's installed is always `verity catalog`.
