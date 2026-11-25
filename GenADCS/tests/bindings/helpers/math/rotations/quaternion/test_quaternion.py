import gc
import numpy as np
import pytest


def test_numpy_eigen_or_nalgebra_conversion_and_output_ownership(rotations):
    values = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    original = values.copy()
    q = rotations.Quaternion(values)
    assert q.norm == pytest.approx(1.0)
    np.testing.assert_array_equal(values, original)
    out = np.asarray(q)
    assert out.dtype == np.float64 and out.shape == (4,)
    out[0] = 88
    assert np.asarray(q)[0] == pytest.approx(1.0)


def test_slerp_binding_accepts_lists_numpy_views_and_extrapolation(rotations):
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    backing = np.array([9.0, 0.0, 0.0, 1.0, 8.0])
    q1 = backing[1:4:2]
    # Use a contiguous quaternion for conversion and also ensure caller buffers remain untouched.
    end = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    got = rotations.slerp(q0, end, 0.5)
    np.testing.assert_allclose(
        got, [np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)], atol=1e-12
    )
    assert np.asarray(rotations.slerp(q0, end, 1.5)).shape == (4,)
    np.testing.assert_array_equal(backing, [9.0, 0.0, 0.0, 1.0, 8.0])


def test_quaternion_objects_are_safe_after_temporaries_are_destroyed(rotations):
    q = rotations.Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    expected = q.to_dcm().copy()
    gc.collect()
    np.testing.assert_allclose(q.to_dcm(), expected, atol=1e-12)
    arr = np.asarray(q)
    del q
    gc.collect()
    np.testing.assert_allclose(arr, [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], atol=1e-12)


def test_quaternion_binding_errors_map_to_value_error(rotations):
    for value in ([0.0, 0.0, 0.0, 0.0], [1.0, 2.0], [1.0, np.nan, 0.0, 0.0]):
        with pytest.raises((ValueError, TypeError)):
            rotations.Quaternion(value)
    with pytest.raises(ValueError):
        rotations.slerp([0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 0.5)
    with pytest.raises((ValueError, TypeError)):
        rotations.Quaternion([1.0, 0.0, 0.0, 0.0]).rotate([1.0, 2.0])
