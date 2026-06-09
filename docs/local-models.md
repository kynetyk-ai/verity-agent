# Running Verity on a local / self-hosted model (Phase 6.1)

Verity's sandbox can drive the agent loop with **any OpenAI-compatible model server** running on your
host — a local open-weights model instead of a frontier API. The system depends only on the
OpenAI-compatible HTTP contract, **not** on any specific server: vLLM, Ollama, and llama.cpp are
interchangeable. This is the thesis payoff — *good proposals from cheap models*.

## How it fits together

- A `ModelSpec` with a `base_url` (instead of a bare `provider:model` string) targets an
  OpenAI-compatible endpoint. `local_spec(...)` (in `verity.sandbox.providers`) is the ready-made
  constructor.
- The sandbox container has **outbound network on**, and when the `base_url` points at
  `host.docker.internal` the driver automatically adds `--add-host=host.docker.internal:host-gateway`
  so the container can reach a server on your host.
- Most local servers are keyless; the resolver sends a placeholder `EMPTY` key (the vLLM/Ollama
  convention). Set `api_key_env` if your server requires a key.

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

### vLLM (incl. vLLM-mlx on Apple Silicon)
OpenAI-compatible server, Metal backend on Apple Silicon. Serve on `:8000`; the default
`local_spec` `base_url` already matches:
```
local_spec("qwen2.5-coder")  # -> http://host.docker.internal:8000/v1
```

### Ollama
Ollama serves an OpenAI-compatible API on port `11434`. Override the `base_url` and port:
```
local_spec("qwen2.5-coder", base_url="http://host.docker.internal:11434/v1")
```

### llama.cpp / any other
Anything that speaks the OpenAI `/v1` contract works — set `base_url` to its host/port.

## Notes

- **Small context windows.** Local/open models often have a smaller window than frontier models; the
  sandbox's compaction (Phase 5.2, auto-injected `SummarizationMiddleware`) and soft-deadline nudge
  keep long cycles from blowing the budget.
- **Tool-calling quality varies.** The propose seam needs the model to emit tool calls reliably;
  weaker models will show up as a low accepted-count in the eval harness (Phase 6.3) — that *is* the
  measurement, not a bug.
- **Linux vs macOS.** `--add-host=host.docker.internal:host-gateway` is required on Linux (Docker
  20.10+) and a harmless no-op on macOS/Docker Desktop, where the hostname resolves automatically.
