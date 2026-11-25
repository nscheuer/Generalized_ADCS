from importlib import import_module
from typing import Any

import pytest


@pytest.fixture(scope="module")
def time_backends() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("python", "cpp", "rust"):
        try:
            result[name] = import_module(f"GenADCS.backends.{name}.helpers.math.time")
        except ImportError as exc:
            pytest.fail(f"build {name} time backend before differential tests: {exc}")
    return result
