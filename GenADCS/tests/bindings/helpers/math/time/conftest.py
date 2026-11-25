from importlib import import_module
from typing import Any

import pytest


@pytest.fixture(params=("cpp", "rust"), ids=("cpp", "rust"))
def time_api(request: Any) -> Any:
    return import_module(f"GenADCS.backends.{request.param}.helpers.math.time")
