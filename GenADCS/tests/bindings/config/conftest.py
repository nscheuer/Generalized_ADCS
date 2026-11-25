from importlib import import_module
from typing import Any

import pytest


@pytest.fixture(params=("python", "cpp", "rust"), ids=("python", "cpp", "rust"))
def config_backend_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def config_backend(config_backend_name: str) -> Any:
    return import_module(f"GenADCS.backends.{config_backend_name}.config")
