from importlib import import_module

import pytest


@pytest.fixture(params=("cpp", "rust"), ids=("cpp", "rust"))
def geometry_backend_name(request):
    return request.param


@pytest.fixture
def geometry_backend(geometry_backend_name):
    return import_module(f"GenADCS.backends.{geometry_backend_name}.hardware.satellite_geometry")
