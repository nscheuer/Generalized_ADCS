from typing import Any

import numpy as np
import pytest


def test_native_srp_functions_accept_numpy_inputs_and_scale_with_illumination(
    dynamics_backend: Any,
) -> None:
    position = np.zeros(3, dtype=np.float32)
    sun = np.array([149_597_870_700.0, 0.0, 0.0], dtype=np.float64)
    quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    normals = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    arms = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    areas = np.array([2.0], dtype=np.float32)
    optical = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

    force = dynamics_backend.projected_area_srp_force(
        position, sun, quaternion, normals, areas, optical
    )
    half_force = dynamics_backend.projected_area_srp_force(
        position, sun, quaternion, normals, areas, optical, 0.5
    )
    torque = dynamics_backend.projected_area_srp_torque(
        position, sun, quaternion, normals, arms, areas, optical
    )

    assert force.shape == half_force.shape == torque.shape == (3,)
    assert force.dtype == half_force.dtype == torque.dtype == np.float64
    assert np.isfinite(force).all() and np.linalg.norm(force) > 0.0
    np.testing.assert_allclose(half_force, 0.5 * force)
    assert np.isfinite(torque).all() and np.linalg.norm(torque) > 0.0


def test_native_srp_binding_maps_coincident_positions_to_value_error(dynamics_backend: Any) -> None:
    with pytest.raises(ValueError, match="positions coincide"):
        dynamics_backend.projected_area_srp_force(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [[1.0, 0.0, 0.0]],
            [1.0],
            [[0.0, 0.0, 1.0]],
        )
