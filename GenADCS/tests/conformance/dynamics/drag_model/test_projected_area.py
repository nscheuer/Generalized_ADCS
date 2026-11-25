from typing import Any

import numpy as np
import pytest


def test_drag_force_and_torque_match_panel_physics(dynamics_api: Any) -> None:
    velocity = np.array([0.0, 0.0, 10.0])
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    normals = np.array([[0.0, 0.0, 1.0]])
    arms = np.array([[1.0, 0.0, 0.0]])
    areas = np.array([2.0])
    coefficients = np.array([2.2])

    force = dynamics_api.projected_area_drag_force(
        velocity, quaternion, 1.0, normals, areas, coefficients
    )
    torque = dynamics_api.projected_area_drag_torque(
        velocity, quaternion, 1.0, normals, arms, areas, coefficients
    )
    expected_force = np.array([0.0, 0.0, -0.5 * 1.0 * 10.0 * 2.0 * 2.2 * 1.0 * 10.0])

    np.testing.assert_allclose(force, expected_force, atol=1e-12)
    np.testing.assert_allclose(torque, np.cross(arms[0], expected_force), atol=1e-12)


def test_drag_force_scales_with_density_and_vanishes_without_incidence(dynamics_api: Any) -> None:
    args = (
        np.array([0.0, 0.0, 10.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([2.0]),
        np.array([2.2]),
    )
    force = dynamics_api.projected_area_drag_force(*args[:2], 1.0, *args[2:])
    doubled = dynamics_api.projected_area_drag_force(*args[:2], 2.0, *args[2:])
    away = dynamics_api.projected_area_drag_force(
        np.array([0.0, 0.0, -10.0]), args[1], 1.0, *args[2:]
    )
    zero_density = dynamics_api.projected_area_drag_force(*args[:2], 0.0, *args[2:])

    np.testing.assert_allclose(doubled, 2.0 * force)
    np.testing.assert_array_equal(away, np.zeros(3))
    np.testing.assert_array_equal(zero_density, np.zeros(3))


@pytest.mark.parametrize("kind", ("force", "torque"))
def test_drag_rejects_mismatched_panel_arrays(dynamics_api: Any, kind: Any) -> None:
    common = (np.array([0.0, 0.0, 10.0]), np.array([1.0, 0.0, 0.0, 0.0]), 1.0)
    if kind == "force":
        call = lambda: dynamics_api.projected_area_drag_force(
            *common,
            np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            np.array([2.0]),
            np.array([2.2]),
        )
    else:
        call = lambda: dynamics_api.projected_area_drag_torque(
            *common,
            np.array([[0.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([2.0]),
            np.array([2.2]),
        )
    with pytest.raises(ValueError, match="panel arrays must have matching row counts"):
        call()
