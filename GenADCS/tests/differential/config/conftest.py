from importlib import import_module

import pytest


@pytest.fixture(scope="module")
def config_backends():
    try:
        return {
            name: import_module(f"GenADCS.backends.{name}.config")
            for name in ("python", "cpp", "rust")
        }
    except ImportError as exc:
        pytest.fail(f"build all config backends before differential tests: {exc}")
