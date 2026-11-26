"""Load all backend modules for propagation conformance checks."""

from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture(params=("python", "cpp", "rust"), ids=lambda name: name)
def propagation_backend(request: Any) -> SimpleNamespace:
    name = request.param
    try:
        return SimpleNamespace(
            name=name,
            config=import_module(f"GenADCS.backends.{name}.config"),
            environment=import_module(f"GenADCS.backends.{name}.environment"),
            hardware=import_module(f"GenADCS.backends.{name}.hardware"),
            propagation=import_module(f"GenADCS.backends.{name}.propagation"),
            state=import_module(f"GenADCS.backends.{name}.state"),
        )
    except ImportError as exc:
        pytest.fail(f"{name} propagation backend is unavailable: {exc}")
