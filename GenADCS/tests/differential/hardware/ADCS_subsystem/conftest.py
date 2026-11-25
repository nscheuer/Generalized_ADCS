from importlib import import_module
from typing import Any

import pytest


@pytest.fixture(scope="module")
def adcs_backends() -> Any:
    result = {}
    for name in ("python", "cpp", "rust"):
        try:
            result[name] = import_module(
                f"GenADCS.backends.{name}.hardware.ADCS_subsystem"
            )
        except ImportError as exc:
            pytest.fail(f"build {name} ADCS subsystem before differential tests: {exc}")
    return result
