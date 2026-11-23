from importlib import import_module

import pytest


@pytest.fixture(params=("cpp", "rust"), ids=("cpp", "rust"))
def dynamics_backend(request):
    return import_module(f"GenADCS.backends.{request.param}.dynamics.drag_model")
