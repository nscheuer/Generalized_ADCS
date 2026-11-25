from typing import Any

import numpy as np
import pytest


def test_lunisolar_acceleration_matches_differential_gravity_formula(dynamics_api: Any) -> None:
    position = np.array([7e6, 0.0, 0.0])
    sun = np.array([1.5e11, 0.0, 0.0])
    moon = np.array([0.0, 3.8e8, 0.0])
    sun_mu = 1.3271244e20
    moon_mu = 4.9048695e12

    def differential_acceleration(
        body: np.ndarray, gravitational_parameter: float
    ) -> np.ndarray:
        relative = body - position
        return gravitational_parameter * (
            relative / np.linalg.norm(relative) ** 3 - body / np.linalg.norm(body) ** 3
        )

    expected = differential_acceleration(sun, sun_mu) + differential_acceleration(
        moon, moon_mu
    )
    acceleration = dynamics_api.lunisolar_acceleration(
        position, sun, moon, sun_mu, moon_mu
    )

    np.testing.assert_allclose(acceleration, expected, rtol=5e-11, atol=1e-20)


@pytest.mark.parametrize(
    "sun,moon",
    [
        (np.zeros(3), np.ones(3)),
        (np.ones(3), np.zeros(3)),
        (np.array([7e6, 0.0, 0.0]), np.ones(3)),
    ],
)
def test_lunisolar_rejects_zero_body_or_relative_separation(dynamics_api: Any, sun: Any, moon: Any) -> None:
    with pytest.raises(ValueError, match="distance must be nonzero"):
        dynamics_api.lunisolar_acceleration(
            np.array([7e6, 0.0, 0.0]), sun, moon, 1.0, 1.0
        )
