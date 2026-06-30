#!/usr/bin/env bash
# Run an ablation sweep as a ONE-SHOT container from the verity-controlplane image, mirroring
# infra/compose.daemon.yml so the §9.1 verifier sibling resolves by Docker DNS on verity-net.
#
# The sweep is the orchestrator (drives ControlService directly via build_service), NOT the daemon —
# so it must itself run inside a container on verity-net with the docker socket + identical-path
# staging, exactly like the standing CP. The repo is mounted at /repo for the `experiments` package,
# the prepared data dir, and the results output.
#
# Persistence: passes --store-root "$OUT/store" so the per-task SqliteStore + object store land on
# the bind-mounted repo at results/<out>/store/ and SURVIVE the --rm container. Without it the store
# is in-memory and the agent transcripts + submission objects are LOST at run-end (only the RunReport
# JSONs persist). See the package README's "Running it" box. The store dir is root-owned on the host
# (the container runs as root) — chown if a non-root user needs it.
#
#   run_sweep_container.sh <spec.json> <data-dir> <out-dir>
# Paths are relative to the repo root (this script's grandparent). Loads ./.env for model creds.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SPEC="${1:?spec json (repo-relative)}"
DATA="${2:?data dir (repo-relative)}"
OUT="${3:?out dir (repo-relative)}"

# Model creds (ANTHROPIC_API_KEY / OPENAI_API_KEY) from the repo .env, same as compose.
set -a; . "$REPO/.env"; set +a

mkdir -p "$REPO/$OUT"

exec docker run --rm \
  --name "verity-sweep-$(date +%s)" \
  --network verity-net \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/verity-staging:/tmp/verity-staging \
  -v "$REPO":/repo \
  -w /repo \
  -e PYTHONPATH=/repo \
  -e VERITY_VERIFIER_NETWORK=verity-net \
  -e VERITY_VERIFIER_IMAGE=verity-verifier:latest \
  -e VERITY_WORKER_STAGING=/tmp/verity-staging \
  -e VERITY_SANDBOX_IMAGE=verity-sandbox:latest \
  -e VERITY_CODE_IMAGE=verity-code-runner:latest \
  -e VERITY_CODE_MEMORY="${VERITY_CODE_MEMORY:-16g}" \
  -e VERITY_CODE_CPUS="${VERITY_CODE_CPUS:-16}" \
  -e VERITY_CODE_PIDS="${VERITY_CODE_PIDS:-4096}" \
  -e VERITY_SANDBOX_CPUS="${VERITY_SANDBOX_CPUS:-16}" \
  -e VERITY_SANDBOX_PIDS="${VERITY_SANDBOX_PIDS:-4096}" \
  -e VERITY_VERIFIER_MEMORY="${VERITY_VERIFIER_MEMORY:-4g}" \
  -e VERITY_VERIFIER_TIMEOUT="${VERITY_VERIFIER_TIMEOUT:-38880}" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  --entrypoint /app/.venv/bin/python \
  verity-controlplane:latest \
  -m experiments.ablation.sweep "$SPEC" --data "$DATA" --out "$OUT" --store-root "$OUT/store"
