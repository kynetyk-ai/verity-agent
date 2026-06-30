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
verity catalog                   # every installed task type
verity catalog --type fe-kaggle  # one type's full published contract
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
verity ingest agent/train.csv   # reads <exchange>/in/agent/train.csv -> {"handle": "<sha256>"}
```

Content-addresses the file into the daemon and returns a `handle` you pass to `create --file`. Subdir
paths are allowed (role bundles like `agent/train.csv`), as long as they stay inside the exchange.
(File must be in the exchange `in/` dir — see setup.md. Remote/programmatic clients can instead upload
raw bytes over HTTP; see *Network mode* below.)

## `create` — define a task instance → a `task_id`

```bash
# trivial, data-less task — the request is shaped from flags:
verity create --type code \
  --model qwen3.6:27b-coding-mxfp8 --base-url http://host.docker.internal:11434/v1 \
  --goal "Write a submission script that runs cleanly." --max-cycles 4 --stop-on-accept
# -> {"task_id": "..."}

# data-bearing task (e.g. fe-kaggle: per-role files + a competition slug) — from a request file:
verity create --request-file task.json \
  --file agent:train.csv=<h1> --file agent:test.csv=<h2> \
  --file verifier:train.csv=<h1> --file verifier:holdout.csv=<h3> ...   # repeatable, one per file
# (task.json carries type/model/knobs; only --file/--goal still apply) -> {"task_id": "..."}
```

| Flag | Default | Meaning |
|---|---|---|
| `--type` | — | task type from `verity catalog` (e.g. `code`, `fe-holdout`, `fe-kaggle`); required unless `--request-file` is given |
| `--request-file` | — | path to a full `TaskRequest` JSON (e.g. a task package's `task.json`); supersedes the request-shaping flags — only `--file`/`--goal` still apply |
| `--file` | — | `ROLE:NAME=HANDLE` — route a pre-prepared input (from `verity ingest`) to a worker role; **repeatable**. Data prep is yours (ADR 0005); the CP routes opaque blobs. |
| `--goal` | `""` | the run goal (task instructions) |
| `--model` | builder default (Anthropic) | the sandbox model. **With `--base-url`** it is a literal OpenAI-compatible name (Ollama `name:tag`); **without** it is a native `provider:model` string (`anthropic:claude-sonnet-4-6`). |
| `--base-url` | — | OpenAI-compatible endpoint for a local/hosted model |
| `--max-cycles` | `4` | refine cycles before the run stops |
| `--stop-on-accept` | off | stop at the first accepted artifact (omit to keep improving / supersede) |

`--model`/`--base-url` (+ image/memory/runtime provisioning defaults) **are** the sandbox
configuration. Task definitions are durable — a created task survives a daemon restart.

### Multi-input / request-file tasks (e.g. `fe-kaggle`)

Some task types take per-role input files or per-verifier knobs that aren't plain flags. **You
prepare the data outside Verity** (ADR 0005 / #74): split it into per-role bundles — the control
plane routes opaque, role-keyed blobs and interprets none of them. Provide a full `TaskRequest` via
`--request-file` (it supersedes the request-shaping flags; only `--file`/`--goal` still apply),
ingest each role file, and route it with `--file ROLE:NAME=HANDLE`:

```bash
verity ingest agent/train.csv                    # -> {h1}
verity ingest verifier/holdout_labels.csv        # -> {h2}  (the answer key — verifier role only)
# ...ingest every agent/* and verifier/* file...
verity create --request-file task.json \
  --file agent:train.csv={h1} --file verifier:holdout_labels.csv={h2} ...   # -> {task_id}
```

The `TaskRequest` JSON shape (a `--request-file`; committed examples under
`results/prototyping_datasci_test/task*.json`):

```jsonc
{
  "type_name": "fe-holdout",                 // a catalog task type
  "goal": "…the run goal…",
  "sandbox": {                               // model + per-worker budgets (all optional)
    "model": "qwen3.6:27b-coding-mxfp8",     // bare literal name WHEN base_url is set (else provider:model)
    "base_url": "http://host.docker.internal:11434/v1",
    "recursion_limit": 300, "sandbox_timeout_s": 2400, "code_timeout_s": 1800
  },
  "verifier": { "knobs": { "accept_policy": "always" } },  // fe-kaggle uses knobs.competition (the slug)
  "policy": { "max_cycles": 1, "stop_on_accept": false, "provisioning": "none" },
  "data": {}                                 // role files come from the repeated --file flags
}
```

`fe-holdout` (local hold-out, **no creds**) and `fe-kaggle` (real leaderboard) both use the role-file
flow: prepare the bundles with `uv run python -m tools.prepare_fe_data …` (run as a MODULE; or let
`results/prototyping_datasci_test/run.sh` do it). `fe-holdout`'s verifier role is a subset
(`train.csv` + `holdout.csv` + `holdout_labels.csv`); `fe-kaggle` additionally needs
`verifier.knobs.competition` (the slug) and `KAGGLE_USERNAME`/`KAGGLE_KEY` in the daemon env
(setup.md → *Kaggle creds*). Read the live contract with `verity catalog --type <name>`.

### Model targeting — native, local, and OpenAI-compatible endpoints

The sandbox model is set by **`sandbox.model`** (+ optional `base_url`, `api_key_env`) in a request
file, or **`--model`** (+ `--base-url`) on `create`. Two paths:

- **Native provider** — a `provider:model` string, **no** `base_url`. The provider's key env is
  forwarded into the worker automatically, so it just has to be set in the **daemon** env (the worker
  inherits nothing else). Supported today: `anthropic:` (`ANTHROPIC_API_KEY`) and `openai:`
  (`OPENAI_API_KEY`). The `:` splits provider from model — e.g. `anthropic:claude-sonnet-4-6`,
  `openai:gpt-5.4-nano`.
- **OpenAI-compatible endpoint** (any server speaking the OpenAI API — Ollama, vLLM, llama.cpp, LM
  Studio, or a hosted gateway) — set **both** `base_url` **and** `model`, where `model` is the
  endpoint's **literal** name and is **not** split on `:` (so an Ollama `name:tag` like `gpt-oss:20b`
  stays intact). The system depends on the OpenAI-compatible *contract*, not any one server.

Keys & host reachability for the endpoint path:
- **Keyless local servers** (Ollama, a bare vLLM): omit `api_key_env` — the resolver sends `"EMPTY"`.
- **Keyed endpoints**: set `api_key_env` to the env var **name** holding the key (e.g.
  `"api_key_env": "OPENROUTER_API_KEY"`) and make sure that var is set in the daemon env; the worker
  forwards it by name.
- **Reaching a server on the Docker host**: use `host.docker.internal` in the `base_url`
  (`http://host.docker.internal:11434/v1` for Ollama). The driver auto-adds
  `--add-host=host.docker.internal:host-gateway` for any `*.docker.internal` host (required on Linux,
  a no-op on macOS). A public URL needs no gateway.

| Target | `model` | `base_url` | key |
|---|---|---|---|
| Anthropic (native) | `anthropic:claude-sonnet-4-6` | — | `ANTHROPIC_API_KEY` in daemon env |
| OpenAI (native) | `openai:gpt-5.4-nano` | — | `OPENAI_API_KEY` in daemon env |
| Ollama (host-local) | `gpt-oss:20b` | `http://host.docker.internal:11434/v1` | none (keyless → `EMPTY`) |
| vLLM (host-local) | `<served-model-name>` | `http://host.docker.internal:8000/v1` | none, or `api_key_env` |
| Hosted OpenAI-compatible | `<vendor model id>` | `https://…/v1` | `api_key_env` (var name) |

```bash
# native OpenAI, from flags:
verity create --type fe-holdout --model openai:gpt-5.4-nano --goal "…" --file …
# local Ollama, from flags (model NOT split on ':'):
verity create --type fe-holdout --model gpt-oss:20b --base-url http://host.docker.internal:11434/v1 …
```

Match the model to the task: weaker local models may stumble on tool-calling or long agent loops
(see `sandbox_notes` in `verity catalog --type <name>`).

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

### Pulling the agent transcript (or any object by content hash)

`results` includes, per cycle, a `transcript_ref` — the content hash of the agent's rendered
**step transcript** (message types, content, tool calls + args, results; the step-by-step record).
There is **no CLI/HTTP verb for a bare object by hash yet**, and `export` only writes *declared
artifact* objects — so fetch the transcript straight from the **per-task** object store by its hash
(the store is keyed by `task_id`, so you need both):

```bash
R=$(verity results <run_id>)
TASK=$(echo "$R" | python3 -c "import sys,json;print(json.load(sys.stdin)['task_id'])")
HASH=$(echo "$R" | python3 -c "import sys,json;print(json.load(sys.stdin)['report']['cycles'][0]['transcript_ref'])")
docker exec verity-cp cat /var/lib/verity/tasks/$TASK/objects/$HASH   # host: ${VERITY_STORE_HOST:-/tmp/verity-store}/tasks/$TASK/objects/$HASH
```

The transcript is a JSON array; from a repo checkout, pipe it through the standalone renderer for a
readable Markdown view (`--full` to disable truncation, `-o file.md` to write):

```bash
docker exec verity-cp cat /var/lib/verity/tasks/$TASK/objects/$HASH | python tools/render_transcript.py -
```

(Known gap — a `verity object <run_id> <hash>` verb / `GET /objects/{hash}` route is tracked.)

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

The always-current source of truth at runtime is `verity --help` / `verity <verb> --help` and
`verity catalog`. (In the Verity repo, the verbs/flags are defined in `src/verity/service/cli.py` and
the HTTP routes in `src/verity/service/http.py`.)
