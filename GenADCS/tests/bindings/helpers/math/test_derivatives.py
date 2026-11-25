"""Derivative binding input conversion, results, errors, and ownership."""

import gc
from typing import Any

import numpy as np
import pytest


def test_quaternion_derivative_sequence_array_and_dtype_parity(math_backend: Any) -> None:
    q = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    omega = np.array([0.4, -0.7, 0.2], dtype=np.float32)
    q_before, omega_before = q.copy(), omega.copy()
    result = math_backend.quaternion_derivative(q, omega)
    assert result.shape == (4,) and result.dtype == np.float64
    np.testing.assert_allclose(result, [0.0, -0.4, 0.7, -0.2])
    np.testing.assert_array_equal(q, q_before)
    np.testing.assert_array_equal(omega, omega_before)
    np.testing.assert_allclose(
        math_backend.quaternion_derivative(tuple(q), tuple(omega)), result
    )


def test_derivative_outputs_own_their_memory_and_survive_argument_destruction(
    math_backend: Any,
) -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([1.0, 2.0, 3.0])
    result = math_backend.quaternion_derivative(q, omega)
    del q, omega
    gc.collect()
    np.testing.assert_allclose(result, [0.0, -0.5, -1.0, -1.5])
    result[0] = 7.0
    assert result[0] == 7.0


def test_jacobian_results_are_owned_and_have_expected_shapes(math_backend: Any) -> None:
    q = np.array([0.4, -1.2, 0.7, 2.0])
    norm_jacobian = math_backend.quaternion_normalization_jacobian(q)
    rotation_jacobian = math_backend.rotated_vector_quaternion_jacobian(
        np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, -2.0, 3.0])
    )
    assert norm_jacobian.shape == (4, 4)
    assert rotation_jacobian.shape == (3, 4)
    assert norm_jacobian.dtype == rotation_jacobian.dtype == np.float64
    norm_jacobian[0, 0] = 100.0
    assert math_backend.quaternion_normalization_jacobian(q)[0, 0] != 100.0


def test_normalization_jacobian_zero_quaternion_maps_to_value_error(math_backend: Any) -> None:
    with pytest.raises(ValueError, match="nonzero norm"):
        math_backend.quaternion_normalization_jacobian(np.zeros(4))


@pytest.mark.parametrize(
    "bad_q,bad_w", [([1.0, 0.0], [0.0, 0.0, 0.0]), ([1.0, 0.0, 0.0, 0.0], [1.0, 2.0])]
)
def test_wrong_derivative_argument_shapes_raise_python_exception(
    math_backend: Any, backend_name: str, bad_q: Any, bad_w: Any
) -> None:
    expected_exception = TypeError if backend_name == "cpp" else ValueError
    with pytest.raises(expected_exception):
        math_backend.quaternion_derivative(bad_q, bad_w)


def test_quaternion_object_constructor_and_destructor_keep_native_data_valid(
    math_backend: Any,
) -> None:
    source = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    quaternion = math_backend.Quaternion(source)
    source[:] = 0.0
    components = quaternion.components
    del quaternion
    gc.collect()
    np.testing.assert_array_equal(components, [1.0, 0.0, 0.0, 0.0])
