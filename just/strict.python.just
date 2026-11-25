set shell := ["bash", "-euo", "pipefail", "-c"]

default: all

python_source := "GenADCS/backends/python"
python_tests := "GenADCS/tests"

# Sync the project and development dependencies from pyproject.toml.
build:
    command -v uv >/dev/null
    uv sync --extra dev

# Run Ruff's lint rules on the Python backend and its tests.
lint:
    command -v uv >/dev/null
    uv run ruff check --extend-per-file-ignores 'GenADCS/**:N999' {{python_source}} {{python_tests}}

# Run Pyright with strict checking enabled for the backend and tests.
typecheck:
    command -v uv >/dev/null
    uv run pyright --project pyrightconfig.json

# Pyright provides the static analysis pass as well as strict type checking.
analyze: typecheck

# Run pytest with coverage reporting for the Python backend.
coverage:
    command -v uv >/dev/null
    uv run pytest --cov={{python_source}} --cov-report=term-missing GenADCS/tests

# Audit the active uv environment for known vulnerable dependencies.
audit:
    command -v uv >/dev/null
    uv run pip-audit

all: build lint typecheck analyze coverage audit
