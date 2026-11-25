from importlib import import_module
from typing import Any

import pytest


@pytest.fixture(params=("cpp", "rust"), ids=("cpp", "rust"))
def geometry_backend_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def geometry_backend(geometry_backend_name: str) -> Any:
    return import_module(
        f"GenADCS.backends.{geometry_backend_name}.hardware.satellite_geometry"
    )
