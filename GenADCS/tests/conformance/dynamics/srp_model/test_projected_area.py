from typing import Any

import numpy as np
import pytest


def test_srp_force_and_torque_match_absorbing_panel_formula(dynamics_api: Any) -> None:
    position = np.zeros(3)
    sun = np.array([149_597_870_700.0, 0.0, 0.0])
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    normals = np.array([[1.0, 0.0, 0.0]])
    arms = np.array([[0.0, 1.0, 0.0]])
    areas = np.array([2.0])
    optical = np.array([[0.0, 0.0, 1.0]])
    pressure = 1361.0 / 299_792_458.0
    expected_force = np.array([-pressure * 2.0, 0.0, 0.0])

    force = dynamics_api.projected_area_srp_force(
        position, sun, quaternion, normals, areas, optical
    )
    torque = dynamics_api.projected_area_srp_torque(
        position, sun, quaternion, normals, arms, areas, optical
    )

    np.testing.assert_allclose(force, expected_force, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(torque, np.cross(arms[0], expected_force), atol=1e-18)


def test_srp_force_scales_with_illumination_and_zero_illumination(dynamics_api: Any) -> None:
    args = (
        np.zeros(3),
        np.array([149_597_870_700.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([[1.0, 0.0, 0.0]]),
        np.array([2.0]),
        np.array([[0.0, 0.0, 1.0]]),
    )
    full = dynamics_api.projected_area_srp_force(*args)
    half = dynamics_api.projected_area_srp_force(*args, 0.5)
    dark = dynamics_api.projected_area_srp_force(*args, 0.0)
    np.testing.assert_allclose(half, 0.5 * full)
    np.testing.assert_array_equal(dark, np.zeros(3))


def test_srp_rejects_coincident_spacecraft_and_sun_positions(dynamics_api: Any) -> None:
    with pytest.raises(ValueError, match="positions coincide"):
        dynamics_api.projected_area_srp_force(
            np.zeros(3),
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([[1.0, 0.0, 0.0]]),
            np.array([1.0]),
            np.array([[0.0, 0.0, 1.0]]),
        )
