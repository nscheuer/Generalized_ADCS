from importlib import import_module

import pytest


@pytest.fixture(scope="module")
def frame_backends():
    try:
        return {
            name: import_module(f"GenADCS.backends.{name}.helpers.math.frames")
            for name in ("python", "cpp", "rust")
        }
    except ImportError as exc:
        pytest.fail(f"all frame extensions must be built for differential tests: {exc}")
