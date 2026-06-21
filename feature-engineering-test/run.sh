#!/usr/bin/env bash
# Run the stellar fe-kaggle task end to end against the standing Verity daemon.
#
# Prepares the data USER-SIDE (ADR 0005 — the control plane does no splitting), then drives the
# verity CLI inside the running control-plane container:
#   prepare (split -> agent/ + verifier/ bundles) -> ingest each role file -> create (--file) ->
#   run -> poll -> results -> export
# The hard gate submits the regenerated predictions to the REAL Kaggle leaderboard (see PROTOCOL.md).
#
# PREREQUISITES (see PROTOCOL.md):
#   - the daemon is up:        just cp-serve
#   - Kaggle creds in the CP:  KAGGLE_USERNAME / KAGGLE_KEY in .env (compose passes them through)
#   - competition rules accepted on kaggle.com (the API 403s otherwise)
#   - train.csv + test.csv present in THIS directory (gitignored; download from the competition)
#
# Usage:   ./feature-engineering-test/run.sh [path/to/task.json]
# Env:     VERITY_CP (container, default verity-cp), VERITY_EXCHANGE_HOST (default /tmp/verity-exchange),
#          PER_CLASS (default: full competitive train — set N for a quick smoke), RESERVED_FRACTION
#          (default 0.15) — the user-side prep knobs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
CP="${VERITY_CP:-verity-cp}"
EXCHANGE_HOST="${VERITY_EXCHANGE_HOST:-/tmp/verity-exchange}"
TASK_JSON="${1:-$HERE/task.json}"

for f in "$HERE/train.csv" "$HERE/test.csv" "$TASK_JSON"; do
  [ -f "$f" ] || { echo "error: missing $f (see PROTOCOL.md)" >&2; exit 2; }
done
if ! docker exec "$CP" test -S /run/verity.sock 2>/dev/null; then
  echo "error: daemon container '$CP' not reachable — start it with: just cp-serve" >&2
  exit 1
fi

jget() { python3 -c 'import sys,json; print(json.load(sys.stdin)["'"$1"'"])'; }
cli()  { docker exec "$CP" verity "$@"; }

# User-side prep (ADR 0005): split train.csv into role-keyed bundles. The CP never sees the split.
# Default is the full competitive train (the data is the bottleneck for a top-N% score); set
# PER_CLASS=N to subsample for a quick smoke.
echo "==> prepare role-keyed data (agent/ + verifier/)"
PREP_ARGS=( --train "$HERE/train.csv" --test "$HERE/test.csv" --out "$HERE"
            --reserved-fraction "${RESERVED_FRACTION:-0.15}" )
[ -n "${PER_CLASS:-}" ] && PREP_ARGS+=( --per-class "$PER_CLASS" )
( cd "$REPO_ROOT" && python3 -m tools.prepare_fe_data "${PREP_ARGS[@]}" )

# Stage the role bundles + task.json into the exchange in/ dir the daemon reads (mounted at /exchange).
mkdir -p "$EXCHANGE_HOST/in/agent" "$EXCHANGE_HOST/in/verifier"
cp "$HERE/agent/"*.csv "$EXCHANGE_HOST/in/agent/"
cp "$HERE/verifier/"*.csv "$EXCHANGE_HOST/in/verifier/"
cp "$TASK_JSON" "$EXCHANGE_HOST/in/task.json"

echo "==> ingest role files -> handles"
FILE_ARGS=()
for role in agent verifier; do
  for path in "$HERE/$role/"*.csv; do
    name="$(basename "$path")"
    handle="$(cli ingest "$role/$name" | jget handle)"
    FILE_ARGS+=( --file "$role:$name=$handle" )
    echo "    $role/$name"
  done
done

echo "==> create (fe-kaggle, from task.json)"
TID="$(cli create --request-file /exchange/in/task.json "${FILE_ARGS[@]}" | jget task_id)"
echo "    task_id=$TID"

echo "==> run (async; the Kaggle gate may block on the daily cap — that's expected)"
RID="$(cli run "$TID" | jget run_id)"
echo "    run_id=$RID"

echo "==> polling status (Ctrl-C to detach; the run keeps going on the daemon)"
while true; do
  ST="$(cli status "$RID" | jget status)"
  echo "    status=$ST"
  case "$ST" in done|failed) break ;; esac
  sleep 30
done

echo "==> results"; cli results "$RID"
echo "==> export -> exchange out/$RID/"; cli export "$RID"
echo "==> done (run_id=$RID)"
