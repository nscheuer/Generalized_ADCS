from typing import Any

import numpy as np
import pytest


def test_gravity_gradient_matches_central_field_formula(dynamics_api: Any) -> None:
    position = np.array([7e6, 7e6, 0.0])
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    inertia = np.diag([2.0, 3.0, 4.0])
    mu = 3.986e14
    radial = position / np.linalg.norm(position)
    expected = (
        3.0 * mu / np.linalg.norm(position) ** 3 * np.cross(radial, inertia @ radial)
    )

    torque = dynamics_api.static_gravity_gradient_torque(
        position, quaternion, inertia, mu
    )
    np.testing.assert_allclose(torque, expected, rtol=1e-12, atol=1e-15)


def test_gravity_gradient_is_zero_for_spherical_inertia(dynamics_api: Any) -> None:
    torque = dynamics_api.static_gravity_gradient_torque(
        np.array([7e6, -2e6, 1e6]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        4.0 * np.eye(3),
        3.986e14,
    )
    np.testing.assert_allclose(torque, np.zeros(3), atol=1e-15)


def test_gravity_gradient_rejects_body_centre_singularity(dynamics_api: Any) -> None:
    with pytest.raises(ValueError, match="body's centre"):
        dynamics_api.static_gravity_gradient_torque(
            np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.eye(3), 3.986e14
        )
