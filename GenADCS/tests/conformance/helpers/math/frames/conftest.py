from importlib import import_module
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture(params=("python", "cpp", "rust"), ids=("python", "cpp", "rust"))
def frames(request: Any) -> ModuleType:
    try:
        return import_module(f"GenADCS.backends.{request.param}.helpers.math.frames")
    except ImportError as exc:
        pytest.fail(
            f"{request.param} frame backend unavailable; build native extensions first: {exc}"
        )
