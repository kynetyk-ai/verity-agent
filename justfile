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
