# Running Verity on a local / self-hosted model (Phase 6.1)

Verity's sandbox can drive the agent loop with **any OpenAI-compatible model server** running on your
host — a local open-weights model instead of a frontier API. The system depends only on the
OpenAI-compatible HTTP contract, **not** on any specific server: vLLM, Ollama, and llama.cpp are
interchangeable. This is the thesis payoff — *good proposals from cheap models*.

## Architecture: the model server is a separate, networked service

Verity holds **no** model runtime — no weights, no inference, just an OpenAI-compatible HTTP client.
The model server is an independent service the sandbox reaches **over the network** at a `base_url`;
its lifecycle is fully decoupled from Verity. The **target** shape is the model server in its **own
container** (or a remote endpoint) — and on Linux with an NVIDIA GPU it is exactly that.

### Why a host process on macOS (a hardware constraint, not the design we'd choose)

**On macOS / Apple Silicon the model server cannot run in a container — this is platform-imposed, not
ideal.** Docker Desktop runs containers in a Linux VM with **no Metal GPU passthrough**, and the
Apple-Silicon inference engines (MLX) are Metal-only. So the GPU-bound model server *must* run as a
**host process**, and Verity's sandbox container networks *out* to it. We accept this only because the
alternative on a Mac is no GPU at all. The decoupling itself is unchanged — only *where the server
lives* differs; the `base_url` seam is identical.

Verified topology of a local run (Ollama on an M3 Ultra, Phase 6 validation):

```
HOST (macOS, Metal GPU)                          DOCKER (Linux VM — no GPU)
┌──────────────────────────────────┐            ┌─────────────────────────────┐
│ Ollama.app  (native host process)│            │ verity-sandbox  (container) │
│  ├─ ollama serve                 │   HTTP     │  the agent loop             │
│  ├─ ollama runner --mlx-engine ◀─┼────────────┤  ChatOpenAI → base_url      │
│  │     (Metal GPU inference)     │            │  http://host.docker.internal│
│  └─ LISTEN 127.0.0.1:11434       │            │              :11434/v1      │
└──────────────────────────────────┘            └─────────────────────────────┘
        ▲                                         the ONLY container in the loop
        └─ Docker Desktop maps host.docker.internal → host loopback;
           the driver adds --add-host=host.docker.internal:host-gateway
```

The model server (Ollama, vLLM-mlx, or Docker Model Runner) is **not** in Docker — it's a host process
on the Metal GPU. The **only** container is Verity's sandbox, reaching *out* to the host. The
`127.0.0.1` bind is fine because Docker Desktop forwards `host.docker.internal` to the host loopback.

> **This is a concession to the platform, not the target architecture.** The moment the model service
> runs where GPUs *are* containerizable — a Linux/CUDA host, or a hosted endpoint (the Phase 7 service
> split) — it becomes a real container / remote service with **no Verity change**, just a different
> `base_url`.

## How it fits together

- A `ModelSpec` with a `base_url` (instead of a bare `provider:model` string) targets an
  OpenAI-compatible endpoint. `local_spec(...)` (in `verity.sandbox.providers`) is the ready-made
  constructor.
- The sandbox container has **outbound network on**, and when the `base_url` is any
  `*.docker.internal` host (`host.docker.internal`, or Docker Model Runner's
  `model-runner.docker.internal`) the driver automatically adds `--add-host=<that-host>:host-gateway`
  so the container can reach a server on your host.
- Most local servers are keyless; the resolver sends a placeholder `EMPTY` key (the vLLM/Ollama
  convention). Set `api_key_env` if your server requires a key.

> **The containerized path.** To drive a local model without writing any wiring, use the standing
> daemon: start your server, then `just cp-serve` with `VERITY_MODEL=<your-model>` and
> `VERITY_LOCAL_BASE_URL=http://host.docker.internal:<port>/v1` set, and create a task with
> `--model <your-model> --base-url …`. The control-plane container launches the sandbox worker, which
> reaches your host server over the same `host.docker.internal` gateway described below. See the
> *Worked example* in [api-surface.md](api-surface.md) for the env surface.

## Recipe (any OpenAI-compatible server)

1. **Start a model server on the host**, exposing an OpenAI-compatible API (default port `8000`).
2. **Rebuild the sandbox image** so it has the current code:
   ```
   docker build -f Dockerfile.sandbox -t verity-sandbox:latest .
   ```
   (The image already bakes in `langchain-openai`; no image change is needed for local models.)
3. **Register a local sandbox** and point a task at it:
   ```python
   from verity.sandbox.providers import local_spec
   from verity.sandbox.registration import build_container_sandbox, register_sandbox

   register_sandbox(sandbox_providers, "deepagents-local", lambda: build_container_sandbox(
       schema=schema, root=root, model="unused",
       model_spec=local_spec("qwen2.5-coder"),  # the name your server serves
   ))
   # ...then TaskConfig(..., sandbox_key="deepagents-local")
   ```

## Server examples (interchangeable)

### Docker Model Runner (DMR) — the cleaner managed path (with a current caveat)
DMR gives the model a **Docker-managed lifecycle** (pull / schedule / start-stop) and a
container-facing OpenAI endpoint, while running the GPU work (vllm-metal) on the host — the closest
thing to "a separate, managed model container" that Metal allows, and the tidier option **when it
works for your model**.

> **Known issue (2026-06, revisit for hardening — tracked as #40).** We tried DMR first as the cleaner solution but its
> `vllm-metal` backend **failed `EngineCore` initialization on Qwen3.6** (`docker model logs` →
> "Engine core initialization failed") — the architecture is too new for that backend build. We fell
> back to **Ollama** (below), which ran the same model fine. DMR is worth revisiting once its
> `vllm-metal` tracks newer architectures; the Verity seam already supports it unchanged. To sanity-check
> DMR itself, pull a model it documents (e.g. `mlx-community/Llama-3.2-1B-Instruct-4bit`) — if that
> loads, the failure is model-specific, not your DMR install.

```
docker model install-runner --backend vllm        # one-time: the vLLM/Metal backend
docker desktop enable model-runner --tcp 12434     # expose the OpenAI API on the host
docker model pull <mlx-community/...-model>         # MLX safetensors model
```

Point Verity at it via the host gateway (reuses the standard path, **no Verity change**):
```
local_spec("<model>", base_url="http://host.docker.internal:12434/engines/v1")
```
DMR also publishes the in-container hostname `http://model-runner.docker.internal/engines/v1`; the
driver maps it automatically (it is a `*.docker.internal` host), so that `base_url` works too.

> **Verify tool-calling control.** Our propose seam needs *structured* `tool_calls`. With raw
> `vllm-mlx serve` you set the parser explicitly (below); confirm DMR's vLLM backend does the same
> for your model before relying on it — use the pre-flight check below.

### vLLM-mlx (raw, on Apple Silicon) — full tool-parser control
OpenAI-compatible server, Metal backend. **Tool-calling is off by default and the propose seam needs
it**, so enable the parser explicitly:
```
vllm-mlx serve unsloth/Qwen3.6-27B-MLX-8bit --port 8000 \
    --continuous-batching --enable-auto-tool-choice --tool-call-parser qwen3_coder
```
```
local_spec("unsloth/Qwen3.6-27B-MLX-8bit")  # -> http://host.docker.internal:8000/v1
```
(8-bit 27B is ~35 GB of weights → wants ~48 GB+ unified memory; drop to a 4-bit MLX repo on a 32 GB
Mac. Note: `mxfp8` is an *Ollama* quant tag — there is no MXFP8 *MLX* build; the 8-bit MLX repo is the
faithful substitute.)

### Ollama
Ollama serves an OpenAI-compatible API on port `11434`. Override the `base_url` and port:
```
local_spec("qwen2.5-coder", base_url="http://host.docker.internal:11434/v1")
```

### llama.cpp / any other
Anything that speaks the OpenAI `/v1` contract works — set `base_url` to its host/port.

## Pre-flight: prove reachability + tool-calling before a full agent run

Debug the network and the tool-call parsing **in isolation**, so a failure is unambiguous (a failed
agent cycle is a slow, noisy way to discover the server can't emit `tool_calls`).

```
# 1) reachable from the host?
curl http://localhost:8000/v1/models                  # (or :12434/engines/v1/models for DMR)

# 2) reachable from inside a container (the real path)?
docker run --rm --add-host=host.docker.internal:host-gateway curlimages/curl \
    http://host.docker.internal:8000/v1/models

# 3) does it return STRUCTURED tool_calls (not the call buried in message content)?
curl http://localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "<model>", "messages": [{"role":"user","content":"call ping with x=1"}],
  "tools": [{"type":"function","function":{"name":"ping",
    "parameters":{"type":"object","properties":{"x":{"type":"integer"}}}}}],
  "tool_choice": "auto"}'
# PASS = the reply has choices[0].message.tool_calls; FAIL = the call is plain text in .content
```

Step 3 directly tests the known Qwen3-coder tool-parsing flakiness (vLLM #22975). If it fails, the
propose seam will show **accepted-count = 0** even though the model "runs" — fix the parser flag (or
switch to raw `vllm-mlx serve` for control) before benchmarking.

## Notes

- **Small context windows.** Local/open models often have a smaller window than frontier models; the
  sandbox's compaction (Phase 5.2, auto-injected `SummarizationMiddleware`) and soft-deadline nudge
  keep long cycles from blowing the budget.
- **Tool-calling quality varies.** The propose seam needs the model to emit tool calls reliably;
  weaker models will show up as a low accepted-count in the eval harness (Phase 6.3) — that *is* the
  measurement, not a bug.
- **Linux vs macOS.** The `--add-host=<host>:host-gateway` the driver adds is required on Linux
  (Docker 20.10+) and a harmless no-op on macOS/Docker Desktop, where `*.docker.internal` resolves
  automatically.
