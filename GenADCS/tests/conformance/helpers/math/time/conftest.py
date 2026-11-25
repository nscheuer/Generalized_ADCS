from importlib import import_module
import pytest


@pytest.fixture(params=("python", "cpp", "rust"), ids=("python", "cpp", "rust"))
def time_api(request):
    try:
        return import_module(f"GenADCS.backends.{request.param}.helpers.math.time")
    except ImportError as exc:
        pytest.fail(
            f"{request.param} time backend unavailable; build its extension first: {exc}"
        )
