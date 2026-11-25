from importlib import import_module
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture(params=("python", "cpp", "rust"), ids=lambda name: name)
def config_api(request: Any) -> ModuleType:
    try:
        return import_module(f"GenADCS.backends.{request.param}.config")
    except ImportError as exc:
        pytest.fail(f"{request.param} config backend unavailable: {exc}")
