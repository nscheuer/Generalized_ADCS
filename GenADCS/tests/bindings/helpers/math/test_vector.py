"""Vector binding conversion, ownership, and validation checks."""

import gc

import numpy as np
import pytest


def test_vector_bindings_accept_sequences_and_numpy_dtypes(math_backend):
    expected = np.array([-11.0, -14.0, 13.0])
    for left, right in (
        ([1.0, 2.0, 3.0], ( -4.0, 5.0, 2.0)),
        (np.array([1, 2, 3], dtype=np.float32), np.array([-4, 5, 2], dtype=np.float32)),
    ):
        result = math_backend.cross(left, right)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, expected)


def test_noncontiguous_numpy_input_is_converted_without_mutation(math_backend):
    left_storage = np.array([1.0, 80.0, -2.0, 90.0, 3.0, 100.0])
    right_storage = np.array([4.0, 70.0, 5.0, 60.0, -6.0, 50.0])
    left, right = left_storage[::2], right_storage[::2]
    assert not left.flags.c_contiguous and not right.flags.c_contiguous
    original_left, original_right = left.copy(), right.copy()

    np.testing.assert_allclose(math_backend.cross(left, right), np.cross(left, right))
    np.testing.assert_array_equal(left, original_left)
    np.testing.assert_array_equal(right, original_right)


def test_returned_arrays_are_independent_owned_results(math_backend):
    source = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    normalized = math_backend.normalize(source)
    assert normalized.dtype == np.float64
    normalized[0] = 123.0
    np.testing.assert_array_equal(source, [3.0, 4.0, 0.0])
    np.testing.assert_allclose(math_backend.normalize(source), [0.6, 0.8, 0.0])

    matrix = math_backend.skew([1.5, -2.0, 4.0])
    assert matrix.shape == (3, 3) and matrix.dtype == np.float64
    matrix[0, 0] = 99.0
    assert math_backend.skew([1.5, -2.0, 4.0])[0, 0] == 0.0


def test_norm_normalize_and_unit_zero_vector_contracts_cross_binding(math_backend):
    zero = np.zeros(3)
    assert math_backend.norm(zero) == 0.0
    np.testing.assert_array_equal(math_backend.normalize(zero), zero)
    with pytest.raises(ValueError):
        math_backend.unit(zero)


@pytest.mark.parametrize("bad", ([1.0, 2.0], [1.0, 2.0, 3.0, 4.0]))
def test_wrong_vector_shapes_raise_python_exception(math_backend, backend_name, bad):
    expected_exception = TypeError if backend_name == "cpp" else ValueError
    with pytest.raises(expected_exception):
        math_backend.cross(bad, [1.0, 0.0, 0.0])


def test_nonfinite_vector_boundary_behavior_is_explicit(math_backend, backend_name):
    bad = [1.0, np.nan, 3.0]
    if backend_name == "rust":
        with pytest.raises(ValueError, match="finite"):
            math_backend.cross(bad, [0.0, 1.0, 0.0])
    else:
        assert np.isnan(math_backend.cross(bad, [0.0, 1.0, 0.0])).any()


def test_native_quaternion_constructor_owns_values_and_survives_gc(math_backend):
    components = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    quaternion = math_backend.Quaternion(components)
    components[0] = 99.0
    gc.collect()

    owned = quaternion.components
    assert owned.dtype == np.float64
    np.testing.assert_array_equal(owned, [1.0, 0.0, 0.0, 0.0])
    owned[0] = -5.0
    np.testing.assert_array_equal(quaternion.components, [1.0, 0.0, 0.0, 0.0])
    del quaternion
    gc.collect()
    np.testing.assert_array_equal(owned, [-5.0, 0.0, 0.0, 0.0])
