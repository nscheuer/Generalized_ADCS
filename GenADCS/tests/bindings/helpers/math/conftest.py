"""Fixtures for Python calls through the native math bindings."""

from importlib import import_module

import pytest


@pytest.fixture(params=("cpp", "rust"), ids=("cpp", "rust"))
def backend_name(request):
    return request.param


@pytest.fixture
def math_backend(backend_name):
    return import_module(f"GenADCS.backends.{backend_name}.helpers.math")
