set shell := ["bash", "-euo", "pipefail", "-c"]

default: format

# Apply Ruff's formatter to the Python backend and its tests.
format:
    command -v ruff >/dev/null
    ruff format GenADCS/backends/python GenADCS/tests
