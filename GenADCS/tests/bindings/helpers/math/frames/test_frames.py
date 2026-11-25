"""Python integration checks for the C++ and Rust frame extensions."""
from importlib import import_module
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture(params=("cpp", "rust"), ids=("cpp", "rust"))
def frames(request: Any) -> Any:
    return import_module(f"GenADCS.backends.{request.param}.helpers.math.frames")


def test_numpy_vectors_convert_to_native_and_back(frames: Any) -> Any:
    q = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)
    vector = np.array([1, 0, 0], dtype=np.float32)
    original_q = q.copy()
    original_vector = vector.copy()
    body: NDArray[np.float64] = frames.inertial_to_body(vector, q)
    assert isinstance(body, np.ndarray)
    assert body.dtype == np.float64
    np.testing.assert_allclose(body, [0, 1, 0], atol=1e-7)
    np.testing.assert_array_equal(q, original_q)
    np.testing.assert_array_equal(vector, original_vector)
    np.testing.assert_allclose(
        frames.body_to_inertial(body, q), original_vector, atol=1e-7
    )


def test_numpy_matrix_conversion_and_result_ownership(frames: Any) -> Any:
    q = np.array([1.0, 0, 0, 0])
    matrix: NDArray[np.float64] = frames.inertial_to_body_matrix(q)
    assert matrix.shape == (3, 3) and matrix.dtype == np.float64
    matrix[0, 0] = 99
    np.testing.assert_array_equal(frames.inertial_to_body_matrix(q), np.eye(3))


def test_covariance_binding_uses_native_transform_and_does_not_alias_input(frames: Any) -> Any:
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    covariance = np.array([[4.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    result = frames.transform_covariance(covariance, rotation)
    assert result.shape == (3, 3) and result.dtype == np.float64
    np.testing.assert_allclose(result, rotation @ covariance @ rotation.T)
    result[0, 0] = -100
    assert covariance[0, 0] == 4


@pytest.mark.parametrize("vector", [np.array([1.0, 2.0]), np.array([1.0, np.nan, 2.0])])
def test_invalid_numpy_vector_raises_python_value_error(frames: Any, vector: Any) -> Any:
    with pytest.raises(ValueError):
        frames.transform_vector(vector, np.eye(3))


def test_invalid_numpy_matrix_and_quaternion_raise_python_value_error(frames: Any) -> Any:
    with pytest.raises(ValueError):
        frames.transform_vector([1.0, 2.0, 3.0], np.eye(2))
    with pytest.raises(ValueError):
        frames.inertial_to_body_matrix([1.0, np.inf, 0.0, 0.0])
