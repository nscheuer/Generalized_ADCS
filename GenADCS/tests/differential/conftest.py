"""Load all backends so differential checks fail clearly when one is missing."""

from importlib import import_module
from typing import Any

import pytest


@pytest.fixture(scope="module")
def state_backends() -> Any:
    modules = {}
    for name in ("python", "cpp", "rust"):
        try:
            modules[name] = import_module(f"GenADCS.backends.{name}.state")
        except ImportError as exc:
            pytest.fail(
                f"build the {name} state backend before differential tests: {exc}"
            )
    return modules
