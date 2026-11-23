"""The common Python-facing state contract runs against every backend."""

from importlib import import_module

import pytest


@pytest.fixture(params=("python", "cpp", "rust"), ids=lambda name: name)
def state_backend(request):
    name = request.param
    try:
        return import_module(f"GenADCS.backends.{name}.state")
    except ImportError as exc:
        pytest.fail(f"{name} state backend is unavailable; build its extension before conformance tests: {exc}")
