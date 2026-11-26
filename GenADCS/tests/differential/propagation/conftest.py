"""Load all backends so differential propagation checks fail clearly if missing."""

from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture(scope="module")
def propagation_backends() -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for name in ("python", "cpp", "rust"):
        try:
            modules[name] = SimpleNamespace(
                name=name,
                config=import_module(f"GenADCS.backends.{name}.config"),
                environment=import_module(f"GenADCS.backends.{name}.environment"),
                hardware=import_module(f"GenADCS.backends.{name}.hardware"),
                propagation=import_module(f"GenADCS.backends.{name}.propagation"),
                state=import_module(f"GenADCS.backends.{name}.state"),
            )
        except ImportError as exc:
            pytest.fail(
                f"build the {name} propagation backend before differential tests: {exc}"
            )
    return modules
