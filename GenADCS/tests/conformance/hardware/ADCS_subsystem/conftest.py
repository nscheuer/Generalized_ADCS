from importlib import import_module

import pytest


@pytest.fixture(params=("python", "cpp", "rust"), ids=lambda name: name)
def adcs_api(request):
    try:
        return import_module(f"GenADCS.backends.{request.param}.hardware.ADCS_subsystem")
    except ImportError as exc:
        pytest.fail(f"{request.param} ADCS subsystem unavailable: {exc}")
