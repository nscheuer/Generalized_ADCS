import numpy as np
import pytest

from GenADCS.backends.python.helpers.math.frames import (
    body_to_inertial,
    body_to_inertial_matrix,
    inertial_to_body,
    inertial_to_body_matrix,
    transform_covariance,
    transform_vector,
)


def test_identity_quaternion_gives_identity_matrices():
    q = np.array([1.0, 0, 0, 0])
    np.testing.assert_allclose(inertial_to_body_matrix(q), np.eye(3), atol=1e-15)
    np.testing.assert_allclose(body_to_inertial_matrix(q), np.eye(3), atol=1e-15)


def test_quarter_turn_and_inverse_vector_transforms():
    # Scalar-first Hamilton quaternion for +90 degrees about z.
    q = np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
    vector_i = np.array([1.0, 0, 0])
    body = inertial_to_body(vector_i, q)
    np.testing.assert_allclose(body, [0, 1, 0], atol=1e-14)
    np.testing.assert_allclose(body_to_inertial(body, q), vector_i, atol=1e-14)
    np.testing.assert_allclose(
        body_to_inertial_matrix(q), inertial_to_body_matrix(q).T, atol=1e-15
    )


def test_rotation_matrix_is_orthogonal_and_proper():
    q = np.array([0.3, -0.4, 0.1, 0.5], dtype=np.float64)
    q /= np.linalg.norm(q)
    matrix = inertial_to_body_matrix(q)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-14)


def test_vector_transform_matches_matrix_multiplication():
    matrix = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    vector = np.array([2.0, 3, 4])
    np.testing.assert_array_equal(transform_vector(vector, matrix), matrix @ vector)


def test_covariance_transform_preserves_symmetry_and_psd():
    matrix = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    covariance = np.array([[4.0, 1, 0], [1, 2, 0], [0, 0, 3]])
    transformed = transform_covariance(covariance, matrix)
    np.testing.assert_allclose(transformed, matrix @ covariance @ matrix.T)
    np.testing.assert_allclose(transformed, transformed.T, atol=1e-14)
    assert np.linalg.eigvalsh(transformed).min() >= -1e-14


def test_low_level_quaternion_kernel_requires_normalized_input_for_rotation():
    # This kernel applies the quaternion polynomial directly and deliberately
    # does not normalize or validate; callers own that precondition.
    zero_result = inertial_to_body_matrix(np.zeros(4, dtype=np.float64))
    np.testing.assert_array_equal(zero_result, np.zeros((3, 3)))
    q = np.array([0.3, -0.4, 0.1, 0.5], dtype=np.float64)
    q /= np.linalg.norm(q)
    matrix = inertial_to_body_matrix(q)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-14)
    with pytest.raises((ValueError, TypeError)):
        inertial_to_body_matrix(np.array([1.0, 0.0, 0.0]))


def test_kernels_accept_noncontiguous_inputs_and_return_finite_values():
    q = np.array([1.0, 8, 0, 8, 0, 8, 0, 8])[::2]
    v = np.array([2.0, 8, 3, 8, 4, 8])[::2]
    result = inertial_to_body(v, q)
    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_covariance_transform_handles_diagonal_and_dense_inputs():
    dcm = np.eye(3)
    diagonal = np.diag([1.0, 2, 3])
    np.testing.assert_array_equal(transform_covariance(diagonal, dcm), diagonal)
    dense = np.arange(9.0).reshape(3, 3)
    np.testing.assert_array_equal(transform_covariance(dense, dcm), dense)
