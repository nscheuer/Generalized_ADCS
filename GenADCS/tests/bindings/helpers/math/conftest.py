"""Fixtures for Python calls through the native math bindings."""

from importlib import import_module
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture(params=("cpp", "rust"), ids=("cpp", "rust"))
def backend_name(request: Any) -> str:
    return request.param


@pytest.fixture
def math_backend(backend_name: str) -> ModuleType:
    return import_module(f"GenADCS.backends.{backend_name}.helpers.math")
