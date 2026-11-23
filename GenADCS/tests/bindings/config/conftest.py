from importlib import import_module

import pytest


@pytest.fixture(params=("python", "cpp", "rust"), ids=("python", "cpp", "rust"))
def config_backend_name(request):
    return request.param


@pytest.fixture
def config_backend(config_backend_name):
    return import_module(f"GenADCS.backends.{config_backend_name}.config")
