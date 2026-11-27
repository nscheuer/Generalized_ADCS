"""Python binding behavior for the native magnetorquer type."""

from typing import Any

import pytest


def test_magnetorquer_noise_and_bias_at_the_binding_boundary(subsystem_backend: Any) -> None:
    torquer = subsystem_backend.Magnetorquer(
        (1, 2, 3),
        (0, 0, 1),
        0.25,
        noise=0.02,
        bias=-0.01,
    )

    assert torquer.position == (1.0, 2.0, 3.0)
    assert torquer.orientation == (0.0, 0.0, 1.0)
    assert torquer.noise == 0.02
    assert torquer.bias == -0.01

    torquer.noise = 0.05
    torquer.bias = 0.03
    assert torquer.noise == 0.05
    assert torquer.bias == 0.03

    with pytest.raises(ValueError):
        torquer.noise = -0.1
