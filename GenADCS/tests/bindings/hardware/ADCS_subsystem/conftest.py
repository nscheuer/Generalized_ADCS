from importlib import import_module

import pytest


@pytest.fixture(params=("cpp", "rust"), ids=("cpp", "rust"))
def subsystem_backend_name(request):
    return request.param


@pytest.fixture
def subsystem_backend(subsystem_backend_name):
    return import_module(
        f"GenADCS.backends.{subsystem_backend_name}.hardware.ADCS_subsystem"
    )
