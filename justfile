# Verity dev commands. Run `just` to list.

default:
    @just --list

# Sync the environment (creates .venv, installs deps + dev group).
install:
    uv sync

# Run the test suite.
test:
    uv run pytest

# Lint.
lint:
    uv run ruff check .

# Format.
fmt:
    uv run ruff format .

# Type-check.
typecheck:
    uv run mypy src

# Full gate: lint + type-check + test. Must be green before any PR.
check: lint typecheck test

# Bring up the long-lived control-plane DAEMON (ROADMAP 8.3): build the images, then start the
# standing `verity serve` container. Drive it with `just cp <subcommand>`; stop with `just cp-down`.
# Files cross via the exchange: drop inputs into the exchange's in/ dir, find exports under out/.
cp-serve:
    docker build -f Dockerfile.sandbox -t verity-sandbox:latest .
    docker build -f Dockerfile.coderunner -t verity-code-runner:latest .
    docker build -f Dockerfile.verifier -t verity-verifier:latest .
    docker build -f Dockerfile.controlplane -t verity-controlplane:latest .
    mkdir -p /tmp/verity-staging "${VERITY_EXCHANGE_HOST:-/tmp/verity-exchange}/in" \
        "${VERITY_EXCHANGE_HOST:-/tmp/verity-exchange}/out" "${VERITY_STORE_HOST:-/tmp/verity-store}"
    # --env-file .env: compose otherwise resolves ${KAGGLE_*} against infra/.env (the compose-file's
    # project dir), not repo-root .env, so the daemon would launch with EMPTY Kaggle creds and fail
    # only later at the submission gate. Point compose at the repo-root .env explicitly.
    docker compose --env-file .env -f infra/compose.daemon.yml up -d

# Run a `verity` client subcommand against the standing daemon, e.g. `just cp catalog`.
cp *args:
    docker exec verity-cp verity {{args}}

# Stop and remove the standing control-plane daemon (volumes/exchange persist on the host).
cp-down:
    docker compose -f infra/compose.daemon.yml down
