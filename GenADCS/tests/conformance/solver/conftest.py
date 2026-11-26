"""Load all backend modules for solver conformance checks."""

from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture(params=("python", "cpp", "rust"), ids=lambda name: name)
def solver_backend(request: Any) -> SimpleNamespace:
    name = request.param
    try:
        sim_mod = import_module(f"GenADCS.backends.{name}.simulate")
        return SimpleNamespace(
            name=name,
            simulate=sim_mod.simulate,
            ResultsFormat=sim_mod.ResultsFormat,
            config=import_module(f"GenADCS.backends.{name}.config"),
            environment=import_module(f"GenADCS.backends.{name}.environment"),
            hardware=import_module(f"GenADCS.backends.{name}.hardware"),
            propagation=import_module(f"GenADCS.backends.{name}.propagation"),
            solver=import_module(f"GenADCS.backends.{name}.solver"),
            state=import_module(f"GenADCS.backends.{name}.state"),
        )
    except ImportError as exc:
        pytest.fail(f"{name} solver backend is unavailable: {exc}")
