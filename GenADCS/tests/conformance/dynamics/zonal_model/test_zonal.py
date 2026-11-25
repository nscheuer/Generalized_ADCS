from typing import Any

import numpy as np
import pytest


def test_zonal_force_reduces_to_central_gravity_with_zero_coefficients(dynamics_api: Any) -> None:
    position = np.array([7e6, -2e6, 1e6])
    mass = 10.0
    mu = 3.986004418e14
    coefficients = np.zeros(5)
    radius = np.linalg.norm(position)
    expected = -mu * mass * position / radius**3

    force = dynamics_api.zonal_gravity_force(
        position, mass, 0.0, mu, 6_378_137.0, coefficients
    )
    np.testing.assert_allclose(force, expected, rtol=1e-12, atol=1e-14)


def test_zonal_force_scales_linearly_with_mass(dynamics_api: Any) -> None:
    args = (
        np.array([7e6, 0.0, 0.0]),
        0.0,
        3.986004418e14,
        6_378_137.0,
        np.array(
            [1.08262668e-3, -2.5324105e-6, -1.6198976e-6, -2.2729608e-7, 5.406812e-7]
        ),
    )
    one = dynamics_api.zonal_gravity_force(args[0], 1.0, *args[1:])
    two = dynamics_api.zonal_gravity_force(args[0], 2.0, *args[1:])
    np.testing.assert_allclose(two, 2.0 * one, rtol=1e-12, atol=1e-14)
    assert one[0] < 0.0


def test_zonal_force_rejects_body_centre_singularity(dynamics_api: Any) -> None:
    with pytest.raises(ValueError, match="body's centre"):
        dynamics_api.zonal_gravity_force(
            np.zeros(3),
            10.0,
            0.0,
            3.986004418e14,
            6_378_137.0,
            np.zeros(5),
        )
