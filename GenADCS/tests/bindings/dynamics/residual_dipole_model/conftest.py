from importlib import import_module
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture(params=("cpp", "rust"), ids=("cpp", "rust"))
def dynamics_backend(request: Any) -> ModuleType:
    return import_module(
        f"GenADCS.backends.{request.param}.dynamics.residual_dipole_model"
    )
