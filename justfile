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
