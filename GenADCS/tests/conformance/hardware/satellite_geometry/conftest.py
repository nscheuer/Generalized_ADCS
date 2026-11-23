from importlib import import_module

import pytest


@pytest.fixture(params=("python", "cpp", "rust"), ids=("python", "cpp", "rust"))
def geometry_api(request):
    try:
        return import_module(f"GenADCS.backends.{request.param}.hardware.satellite_geometry")
    except ImportError as exc:
        pytest.fail(f"{request.param} satellite geometry backend unavailable; build native extensions first: {exc}")
