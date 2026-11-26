"""Load all backends so differential solver checks fail clearly if missing."""

from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture(scope="module")
def solver_backends() -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for name in ("python", "cpp", "rust"):
        try:
            sim_mod = import_module(f"GenADCS.backends.{name}.simulate")
            modules[name] = SimpleNamespace(
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
            pytest.fail(
                f"build the {name} solver backend before differential tests: {exc}"
            )
    return modules
