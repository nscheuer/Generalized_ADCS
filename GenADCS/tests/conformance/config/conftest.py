from importlib import import_module
import pytest

@pytest.fixture(params=("python", "cpp", "rust"), ids=lambda name: name)
def config_api(request):
    try:
        return import_module(f"GenADCS.backends.{request.param}.config")
    except ImportError as exc:
        pytest.fail(f"{request.param} config backend unavailable: {exc}")
