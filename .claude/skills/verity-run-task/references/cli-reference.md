# `verity` CLI reference

The `verity` console script is **one binary, two roles**: `verity serve` runs the daemon (the
executor); every other verb is a **thin client** over the daemon. Locally you reach it with
`docker exec verity-cp verity <verb>` (equivalently `just cp <verb>`), which speaks to the daemon over
its in-container Unix socket. All client verbs print **JSON** to stdout and exit non-zero on error.

Global flags (before the verb): `--socket <path>` (`$VERITY_SOCKET`, default `/run/verity.sock`),
`--url <http://host:port>` (`$VERITY_URL`, the network daemon), `--token <secret>` (`$VERITY_API_TOKEN`,
bearer auth for the network daemon).

---

## `catalog` — discover what's installed (read this before `create`)

```bash
verity catalog              # every installed task type
verity catalog --type fe    # one type's full published contract
```

Returns `{"task_types": [ <TaskTypeDescription>, ... ]}`. Each description **self-publishes** the
contract you need to use it — this is the **source of truth**, so always read it rather than assuming a
fixed set:

| Field | What it tells you |
|---|---|
| `type_name` | the value to pass to `verity create --type` |
| `artifact_types`, `operations`, `gated_types` | the **output-shape contract** — what the agent must produce and which types are gated |
| `domain_instructions` | the task's I/O contract / instructions rendered into the agent's prompt |
| `verifier_approach` | **how acceptance is judged** — the opaque verifier's published "what I check / what *accepted* means" prose. This is how you learn *which verifier* you're getting. |
| `sandbox_notes` | model / execution caveats (e.g. needs the container sandbox to execute code; tool-calling caveats for weak local models) |

There is **no separate "list verifiers" or "list sandbox configs" command** — a verifier approach is
bundled per task type (read `verifier_approach`), and the **sandbox configuration is what you choose at
`create`** (`--model` / `--base-url` + provisioning defaults). Pick a model whose capabilities match the
`sandbox_notes`.

---

## `ingest` — stage an input file → a data handle

```bash
verity ingest train.csv     # reads <exchange>/in/train.csv -> {"handle": "<sha256>"}
```

Content-addresses the file into the daemon and returns a `handle` you pass to `create --data`. (File
must be in the exchange `in/` dir — see setup.md. Remote/programmatic clients can instead upload raw
bytes over HTTP; see *Network mode* below.)

## `create` — define a task instance → a `task_id`

```bash
verity create --type fe --data <handle> \
  --model qwen3.6:27b-coding-mxfp8 --base-url http://host.docker.internal:11434/v1 \
  --goal "Improve balanced accuracy via feature engineering." \
  --max-cycles 4 --per-class 150 --reserved-fraction 0.5
# -> {"task_id": "..."}
```

| Flag | Default | Meaning |
|---|---|---|
| `--type` | — | task type from `verity catalog` (e.g. `fe`, `code`, `fe-kaggle`); required unless `--request-file` is given |
| `--request-file` | — | path to a full `TaskRequest` JSON (e.g. a task package's `task.json`); supersedes the request-shaping flags — only `--data`/`--test-data`/`--goal` still apply |
| `--data` | — | a handle from `verity ingest` (the primary/train input; omit for data-less types) |
| `--test-data` | — | a second `verity ingest` handle — a task's second input (e.g. `fe-kaggle`'s real test set) |
| `--goal` | `""` | the run goal (task instructions) |
| `--model` | builder default (Anthropic) | the sandbox model. **With `--base-url`** it is a literal OpenAI-compatible name (Ollama `name:tag`); **without** it is a native `provider:model` string (`anthropic:claude-sonnet-4-6`). |
| `--base-url` | — | OpenAI-compatible endpoint for a local/hosted model |
| `--max-cycles` | `4` | refine cycles before the run stops |
| `--stop-on-accept` | off | stop at the first accepted artifact (omit to keep improving / supersede) |
| `--per-class` | `300` | FE: rows per class to subsample |
| `--reserved-fraction` | `0.5` | FE: held-out share scored by the verifier |

`--model`/`--base-url` (+ image/memory/runtime provisioning defaults) **are** the sandbox
configuration. Task definitions are durable — a created task survives a daemon restart.

## `run` — start a run (async) → a `run_id`

```bash
verity run <task_id> [--goal "override goal"]   # -> {"run_id": "..."}  returns immediately
```

The run executes in the background on the daemon (serialized behind a run lock). Poll for completion.

## `status` / `results` / `export`

```bash
verity status <run_id>    # {"run_id","status","summary"}  status ∈ queued|claimed|done|failed
verity results <run_id>   # full RunRecord: status, accepted_artifact_ids, report.summary{cycles_run, outcomes, accepted, superseded, ...}
verity export <run_id>    # writes durable artifacts to <exchange>/out/<run_id>/ -> {"out_dir","exported":[...]}
```

`tasks` lists created task ids: `verity tasks` → `{"tasks": [...]}`.

Poll pattern: call `status` until it is `done` or `failed`, then read `results`. A run can finish with a
mix of outcomes (`accepted` / `revised` / `rejected` / `sandbox_failed`) — a failed cycle is **recorded
and fed back, not fatal** (degrade-don't-crash).

---

## Serving the daemon

```bash
verity serve                       # default: Unix socket (local / docker-exec only, no auth)
VERITY_API_TOKEN=… verity serve --http 0.0.0.0:8080   # network: TCP + bearer auth (refuses to start without the token)
```

The container ENTRYPOINT/compose runs this; you rarely call it by hand. `--http` (or `$VERITY_HTTP`)
selects the network surface and **requires** `VERITY_API_TOKEN`.

## Network mode (remote clients)

Target a network daemon with `--url` + `--token`:

```bash
verity --url http://host:8080 --token "$VERITY_API_TOKEN" catalog
```

Every route but `GET /health` requires `Authorization: Bearer <token>` (401 otherwise). For clients
with **no shared exchange volume**, the HTTP surface adds an over-the-wire byte data plane:
`POST /objects` with a raw body → a handle (vs the JSON `{"name": …}` exchange form the CLI `ingest`
uses), and `GET /artifacts/{run_id}/{object_path}` → the durable bytes. The Python `Client`
(`verity.service.client`) exposes these as `ingest_bytes()` / `download()`.

Authoritative source for the verbs/flags: `src/verity/service/cli.py`; the HTTP routes:
`src/verity/service/http.py`.
