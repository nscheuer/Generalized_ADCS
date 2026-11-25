import numpy as np
import pytest

from GenADCS.backends.python.environment.atmosphere.atmospheric_model import (
    atmospheric_density,
    co_rotating_atmosphere_velocity,
)


def test_co_rotating_velocity_matches_rigid_rotation() -> None:
    position = np.array((7_000_000.0, 0.0, 0.0))
    velocity = co_rotating_atmosphere_velocity(position, 2_451_545.0)

    np.testing.assert_allclose(velocity, (0.0, 7.2921150e-5 * 7_000_000.0, 0.0))


def test_co_rotating_velocity_accepts_custom_angular_velocity() -> None:
    position = np.array((1.0, 2.0, 3.0))
    omega = np.array((4.0, 5.0, 6.0))

    np.testing.assert_allclose(
        co_rotating_atmosphere_velocity(position, 0.0, omega), np.cross(omega, position)
    )


def test_generic_atmospheric_density_is_explicitly_unimplemented() -> None:
    with pytest.raises(NotImplementedError):
        atmospheric_density(np.array((7_000_000.0, 0.0, 0.0)), 2_451_545.0)
