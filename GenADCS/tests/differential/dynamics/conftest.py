from importlib import import_module

import pytest


@pytest.fixture(scope="module")
def dynamics_backends(request):
    model = request.node.path.parent.name
    modules = {}
    for name in ("python", "cpp", "rust"):
        try:
            modules[name] = import_module(f"GenADCS.backends.{name}.dynamics.{model}")
        except ImportError as exc:
            pytest.fail(f"build the {name} {model} backend before differential tests: {exc}")
    return modules
