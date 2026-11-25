set shell := ["bash", "-euo", "pipefail", "-c"]

default: all

# Install the pinned test environment, then install the Python package in editable mode.
build:
    python -m pip install -r requirements.txt
    python -m pip install --no-deps -e .

# Run the Python backend suite; native binding suites are run after building their backend.
test:
    python -m pytest -n auto GenADCS/tests

all: build test
