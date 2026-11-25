from importlib import import_module
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture(params=("python", "cpp", "rust"), ids=("python", "cpp", "rust"))
def geometry_api(request: Any) -> ModuleType:
    try:
        return import_module(
            f"GenADCS.backends.{request.param}.hardware.satellite_geometry"
        )
    except ImportError as exc:
        pytest.fail(
            f"{request.param} satellite geometry backend unavailable; build native extensions first: {exc}"
        )
