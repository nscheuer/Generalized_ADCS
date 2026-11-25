from importlib import import_module
from typing import Any

import pytest


@pytest.fixture(params=("cpp", "rust"), ids=("cpp", "rust"))
def subsystem_backend_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def subsystem_backend(subsystem_backend_name: str) -> Any:
    return import_module(
        f"GenADCS.backends.{subsystem_backend_name}.hardware.ADCS_subsystem"
    )
