from importlib import import_module
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture(params=("python", "cpp", "rust"), ids=("python", "cpp", "rust"))
def dynamics_api(request: Any) -> ModuleType:
    try:
        return import_module(
            f"GenADCS.backends.{request.param}.dynamics.{request.node.path.parent.name}"
        )
    except ImportError as exc:
        pytest.fail(
            f"{request.param} dynamics backend unavailable; build extensions first: {exc}"
        )
