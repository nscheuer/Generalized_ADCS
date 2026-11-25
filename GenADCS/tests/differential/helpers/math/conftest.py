"""Load every math backend for pairwise numerical comparisons."""

from importlib import import_module

import pytest


@pytest.fixture(scope="module")
def math_backends():
    modules = {}
    for name in ("python", "cpp", "rust"):
        try:
            modules[name] = import_module(f"GenADCS.backends.{name}.helpers.math")
        except ImportError as exc:
            pytest.fail(
                f"build the {name} math backend before differential tests: {exc}"
            )
    return modules
