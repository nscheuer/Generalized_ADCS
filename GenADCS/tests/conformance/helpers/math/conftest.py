"""Shared fixtures for the public Python math contract across backends."""

from importlib import import_module

import pytest


@pytest.fixture(params=("python", "cpp", "rust"), ids=("python", "cpp", "rust"))
def math_api(request):
    try:
        return import_module(f"GenADCS.backends.{request.param}.helpers.math")
    except ImportError as exc:
        pytest.fail(
            f"{request.param} math backend unavailable; build native extensions first: {exc}"
        )
