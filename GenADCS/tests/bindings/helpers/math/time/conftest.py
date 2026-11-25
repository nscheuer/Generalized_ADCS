from importlib import import_module
import pytest


@pytest.fixture(params=("cpp", "rust"), ids=("cpp", "rust"))
def time_api(request):
    return import_module(f"GenADCS.backends.{request.param}.helpers.math.time")
