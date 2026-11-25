"""Shared fixtures for Python calls into native rotation extensions."""

from importlib import import_module
from typing import Any

import pytest


@pytest.fixture(params=("cpp", "rust"), ids=("cpp", "rust"))
def backend(request: pytest.FixtureRequest) -> None:
    return import_module(f"GenADCS.backends.{request.param}.helpers.math")


@pytest.fixture
def rotations(backend: Any) -> None:
    return backend
