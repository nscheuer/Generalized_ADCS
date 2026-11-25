from importlib import import_module
from typing import Any

import pytest


@pytest.fixture(scope="module")
def geometry_backends() -> dict[str, Any]:
    backends: dict[str, Any] = {}
    for name in ("python", "cpp", "rust"):
        try:
            backends[name] = import_module(
                f"GenADCS.backends.{name}.hardware.satellite_geometry"
            )
        except ImportError as exc:
            pytest.fail(
                f"build the {name} satellite geometry backend before differential tests: {exc}"
            )
    return backends
