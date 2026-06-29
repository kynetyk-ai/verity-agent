#!/usr/bin/env bash
# Run a Verity task end to end against the standing daemon (ROADMAP 8.x).
#
# Drives the full flow over the verity CLI inside the running control-plane container:
#   ingest -> create -> run -> poll status -> results -> export
#
# PREREQUISITE: the daemon must already be up (`just cp-serve`). This script does NOT start it.
# Put your input file in the exchange in/ dir first (default host dir: $VERITY_EXCHANGE_HOST/in,
# which is /tmp/verity-exchange/in unless you overrode it when starting the daemon).
#
# Usage:
#   scripts/run_task.sh --type code --goal "write a script that runs"          # trivial, flag-driven
#   scripts/run_task.sh --request-file task.json \                             # e.g. fe-kaggle
#     --file agent:train.csv=agent/train.csv --file verifier:holdout_labels.csv=verifier/holdout_labels.csv ...
#
# DATA PREP IS YOURS (ADR 0005): split inputs into per-role bundles OUTSIDE Verity; the control plane
# routes opaque, role-keyed blobs. For fe-kaggle/fe-holdout, prepare with
# `uv run python -m tools.prepare_fe_data …` (run as a MODULE, not by path — it imports tools.harness)
# (or just use prototyping_datasci_test/run.sh, which does the prep + role routing for you).
#
# Options (with defaults):
#   --type NAME             task type from `verity catalog`            (required unless --request-file)
#   --request-file FILE     a TaskRequest JSON already in exchange in/ (carries type/model/knobs;
#                           supersedes the flag-shaped request — only --file/--goal apply)
#   --file ROLE:NAME=PATH   route a pre-prepared input (PATH under exchange in/, subdirs ok) to a
#                           worker role; repeatable, one per file (data-bearing tasks)
#   --goal "TEXT"          the run goal                                ("")
#   --model NAME            sandbox model                              ($VERITY_MODEL or qwen3.6:27b-coding-mxfp8)
#   --base-url URL          OpenAI-compatible endpoint (local/hosted)  ($VERITY_LOCAL_BASE_URL or http://host.docker.internal:11434/v1)
#   --max-cycles N          refine cycles                              (4)
#   --stop-on-accept        stop at first accepted artifact            (off)
#   --no-export             skip the final export step                 (export on)
#   --timeout SECONDS       max seconds to wait for the run            (3000)
#
# Env:
#   VERITY_CP   the control-plane container name (default: verity-cp)
set -euo pipefail

CP="${VERITY_CP:-verity-cp}"
TYPE="" REQUEST_FILE="" GOAL=""
FILES=()
MODEL="${VERITY_MODEL:-qwen3.6:27b-coding-mxfp8}"
BASE_URL="${VERITY_LOCAL_BASE_URL:-http://host.docker.internal:11434/v1}"
MAX_CYCLES=4 STOP_ON_ACCEPT=0 EXPORT=1 TIMEOUT=3000

while [ $# -gt 0 ]; do
  case "$1" in
    --type) TYPE="$2"; shift 2 ;;
    --request-file) REQUEST_FILE="$2"; shift 2 ;;
    --file) FILES+=("$2"); shift 2 ;;
    --goal) GOAL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --max-cycles) MAX_CYCLES="$2"; shift 2 ;;
    --stop-on-accept) STOP_ON_ACCEPT=1; shift ;;
    --no-export) EXPORT=0; shift ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ -n "$TYPE" ] || [ -n "$REQUEST_FILE" ] || {
  echo "error: --type or --request-file is required (see: docker exec $CP verity catalog)" >&2; exit 2
}

# A small JSON field extractor (python3 is present in the toolchain).
jget() { python3 -c 'import sys,json; print(json.load(sys.stdin)["'"$1"'"])'; }

cli() { docker exec "$CP" verity "$@"; }

# Fail early with a clear message if the daemon is not reachable.
if ! docker exec "$CP" test -S /run/verity.sock 2>/dev/null; then
  echo "error: daemon container '$CP' not reachable (start it with: just cp-serve)" >&2
  exit 1
fi

echo "==> catalog: confirming '$TYPE' is installed"
cli catalog >/dev/null

# A request file supersedes the flag-shaped request (it carries type/model/knobs/competition);
# only --file/--goal still apply. Otherwise the request is shaped from the flags.
if [ -n "$REQUEST_FILE" ]; then
  CREATE_ARGS=(create --request-file "/exchange/in/$REQUEST_FILE")
  [ -n "$GOAL" ] && CREATE_ARGS+=(--goal "$GOAL")
else
  CREATE_ARGS=(create --type "$TYPE" --goal "$GOAL" --max-cycles "$MAX_CYCLES")
fi
# Each --file is ROLE:NAME=PATH — ingest the PATH (already in exchange in/, subdirs ok) and route
# its handle to the role under NAME. The control plane treats every file as an opaque blob.
for entry in ${FILES+"${FILES[@]}"}; do
  rolename="${entry%%=*}"; path="${entry#*=}"
  case "$entry" in *:*=*) ;; *) echo "error: --file must be ROLE:NAME=PATH, got '$entry'" >&2; exit 2 ;; esac
  echo "==> ingest: $path  (-> $rolename)"
  HANDLE="$(cli ingest "$path" | jget handle)"
  echo "    handle=$HANDLE"
  CREATE_ARGS+=(--file "$rolename=$HANDLE")
done
# Model flags are flag-mode only (a request file carries its own model). A model with a base_url is an
# OpenAI-compatible literal name; a bare model is a native provider:model.
if [ -z "$REQUEST_FILE" ]; then
  if [ -n "$BASE_URL" ]; then
    CREATE_ARGS+=(--model "$MODEL" --base-url "$BASE_URL")
  else
    CREATE_ARGS+=(--model "$MODEL")
  fi
  [ "$STOP_ON_ACCEPT" = "1" ] && CREATE_ARGS+=(--stop-on-accept)
fi

echo "==> create: ${REQUEST_FILE:+request-file=$REQUEST_FILE}${TYPE:+type=$TYPE model=$MODEL cycles=$MAX_CYCLES}"
TASK_ID="$(cli "${CREATE_ARGS[@]}" | jget task_id)"
echo "    task_id=$TASK_ID"

echo "==> run (async)"
RUN_ID="$(cli run "$TASK_ID" | jget run_id)"
echo "    run_id=$RUN_ID"

echo "==> polling status (timeout ${TIMEOUT}s)"
START=$(python3 -c 'import time; print(int(time.time()))')
while true; do
  ST="$(cli status "$RUN_ID" | jget status)"
  EL=$(( $(python3 -c 'import time; print(int(time.time()))') - START ))
  echo "    [${EL}s] status=$ST"
  case "$ST" in
    done|failed) break ;;
  esac
  if [ "$EL" -gt "$TIMEOUT" ]; then echo "error: run did not finish within ${TIMEOUT}s" >&2; exit 1; fi
  sleep 15
done

echo "==> results"
cli results "$RUN_ID"

if [ "$EXPORT" = "1" ]; then
  echo "==> export -> exchange out/$RUN_ID/"
  cli export "$RUN_ID"
fi

echo "==> done (run_id=$RUN_ID)"
