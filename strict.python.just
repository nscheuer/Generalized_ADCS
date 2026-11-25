set shell := ["bash", "-euo", "pipefail", "-c"]

default: all

python_source := "GenADCS/backends/python"

# Sync the project and development dependencies from pyproject.toml.
build:
    command -v uv >/dev/null
    uv sync --active --extra dev

# Run Ruff's lint rules on the Python backend, excluding GenADCS/tests.
lint:
    command -v uv >/dev/null
    uv run --active ruff check --extend-per-file-ignores 'GenADCS/**:N999' {{python_source}}

# Run Pyright with strict checking enabled for the Python backend only.
typecheck:
    command -v uv >/dev/null
    uv run --active pyright --project pyproject.toml {{python_source}}

# Pyright provides the static analysis pass as well as strict type checking.
analyze: typecheck

# Run pytest with coverage reporting for the Python backend.
coverage:
    command -v uv >/dev/null
    uv run --active pytest --cov={{python_source}} --cov-report=term-missing GenADCS/tests

# Audit the active uv environment for known vulnerable dependencies.
audit:
    command -v uv >/dev/null
    uv run --active pip-audit

# Coverage remains available as an explicit recipe; strict defaults skip tests.
all: build lint typecheck audit
