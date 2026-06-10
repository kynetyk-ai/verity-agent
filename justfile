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

# Fully containerized FE run (ROADMAP 7.4): the control-plane container launches sibling worker
# containers on the host daemon. Builds the images, then runs one FE task end to end.
fe-containerized:
    docker build -f Dockerfile.sandbox -t verity-sandbox:latest .
    docker build -f Dockerfile.controlplane -t verity-controlplane:latest .
    mkdir -p /tmp/verity-staging
    docker compose -f infra/compose.fe.yml run --rm controlplane

# Bring up the long-lived control-plane DAEMON (ROADMAP 8.3): build the images, then start the
# standing `verity serve` container. Drive it with `just cp <subcommand>`; stop with `just cp-down`.
# Files cross via the exchange: drop inputs into the exchange's in/ dir, find exports under out/.
cp-serve:
    docker build -f Dockerfile.sandbox -t verity-sandbox:latest .
    docker build -f Dockerfile.controlplane -t verity-controlplane:latest .
    mkdir -p /tmp/verity-staging "${VERITY_EXCHANGE_HOST:-/tmp/verity-exchange}/in" \
        "${VERITY_EXCHANGE_HOST:-/tmp/verity-exchange}/out" "${VERITY_STORE_HOST:-/tmp/verity-store}"
    docker compose -f infra/compose.daemon.yml up -d

# Run a `verity` client subcommand against the standing daemon, e.g. `just cp catalog`.
cp *args:
    docker exec verity-cp verity {{args}}

# Stop and remove the standing control-plane daemon (volumes/exchange persist on the host).
cp-down:
    docker compose -f infra/compose.daemon.yml down
