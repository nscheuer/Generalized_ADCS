from typing import Any

import numpy as np
import pytest


def test_numpy_matrix_conversion_and_array_is_owned(rotations: Any) -> None:
    raw = np.eye(3, dtype=np.float32)
    d = rotations.DCM(raw)
    raw[0, 0] = 7
    assert np.asarray(d).dtype == np.float64
    np.testing.assert_allclose(np.asarray(d), np.eye(3))
    result = d.as_matrix()
    result[0, 0] = 8
    np.testing.assert_allclose(d.as_matrix(), np.eye(3))


def test_binding_accepts_native_rotation_classes_and_array_inputs(rotations: Any) -> None:
    q = rotations.Quaternion.from_axis_angle([0.0, 0.0, 1.0], np.pi / 2)
    d = rotations.DCM.from_quaternion(q)
    np.testing.assert_allclose(
        d.rotate(np.array([1.0, 0.0, 0.0])), [0.0, 1.0, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(d.to_quaternion().to_dcm(), d.to_dcm(), atol=1e-12)


def test_invalid_dcm_shapes_values_and_rotations_raise_value_error(rotations: Any) -> None:
    for value in (
        [[1.0, 0.0], [0.0, 1.0]],
        np.full((3, 3), np.nan),
        np.diag([-1.0, 1.0, 1.0]),
        2 * np.eye(3),
    ):
        with pytest.raises((ValueError, TypeError)):
            rotations.DCM(value)
    with pytest.raises(ValueError):
        rotations.DCM.from_axis_angle([0.0, 0.0, 0.0], 1.0)
    with pytest.raises(ValueError):
        rotations.normalize_dcm(np.zeros((3, 3)))
