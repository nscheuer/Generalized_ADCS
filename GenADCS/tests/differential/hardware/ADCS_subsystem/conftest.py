from importlib import import_module

import pytest


@pytest.fixture(scope="module")
def adcs_backends():
    result = {}
    for name in ("python", "cpp", "rust"):
        try:
            result[name] = import_module(
                f"GenADCS.backends.{name}.hardware.ADCS_subsystem"
            )
        except ImportError as exc:
            pytest.fail(f"build {name} ADCS subsystem before differential tests: {exc}")
    return result
