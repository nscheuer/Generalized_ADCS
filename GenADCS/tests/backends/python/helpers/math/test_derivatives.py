from collections.abc import Callable

import numpy as np
import pytest

from GenADCS.backends.python.helpers.math.derivatives import (
    angular_velocity_from_quaternion_derivative,
    quaternion_derivative,
    quaternion_normalization_jacobian,
    quaternion_rate_matrix,
    quaternion_to_euler321,
    rotated_vector_quaternion_jacobian,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import (
    _rotate_vector,
)


def _finite_difference_jacobian(
    function: Callable[[np.ndarray], np.ndarray], x: np.ndarray, step: float = 1e-7
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    baseline = np.asarray(function(x))
    jacobian = np.empty((baseline.size, x.size))
    for column in range(x.size):
        delta = np.zeros_like(x)
        delta[column] = step
        jacobian[:, column] = (function(x + delta) - function(x - delta)) / (2.0 * step)
    return jacobian


def test_quaternion_derivative_matches_rate_matrix_and_inverse_recovers_angular_velocity() -> None:
    q = np.array([0.72, -0.2, 0.31, 0.57])
    q /= np.linalg.norm(q)
    omega = np.array([0.4, -0.7, 0.2])
    qdot = quaternion_derivative(q, omega)
    np.testing.assert_allclose(qdot, quaternion_rate_matrix(q) @ omega)
    np.testing.assert_allclose(
        angular_velocity_from_quaternion_derivative(q, qdot), omega
    )


def test_quaternion_rate_matrix_is_linear_and_has_expected_shape() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    matrix = quaternion_rate_matrix(q)
    assert matrix.shape == (4, 3)
    np.testing.assert_allclose(
        matrix, [[0, 0, 0], [-0.5, 0, 0], [0, -0.5, 0], [0, 0, -0.5]]
    )
    omega = np.array([1.0, 2.0, -3.0])
    np.testing.assert_allclose(
        quaternion_derivative(q, 2 * omega), 2 * quaternion_derivative(q, omega)
    )


def test_quaternion_derivative_preserves_unit_norm_to_first_order() -> None:
    q = np.array([0.8, 0.2, -0.3, 0.45])
    q /= np.linalg.norm(q)
    qdot = quaternion_derivative(q, np.array([0.5, -0.1, 0.9]))
    assert np.dot(q, qdot) == pytest.approx(0.0, abs=1e-14)


def test_quaternion_normalization_jacobian_matches_finite_difference() -> None:
    q = np.array([0.4, -1.2, 0.7, 2.0])
    normalize = lambda value: value / np.linalg.norm(value)
    actual = quaternion_normalization_jacobian(q)
    np.testing.assert_allclose(
        actual, _finite_difference_jacobian(normalize, q), rtol=2e-8, atol=2e-9
    )
    np.testing.assert_allclose(actual @ q, np.zeros(4), atol=1e-14)
    np.testing.assert_allclose(actual, actual.T, atol=1e-14)


def test_quaternion_normalization_jacobian_is_scale_invariant_with_inverse_scaling() -> None:
    q = np.array([0.2, 0.3, -0.4, 0.5])
    np.testing.assert_allclose(
        quaternion_normalization_jacobian(3.0 * q),
        quaternion_normalization_jacobian(q) / 3.0,
    )


def test_quaternion_normalization_jacobian_rejects_zero_norm() -> None:
    with pytest.raises(ValueError, match="nonzero norm"):
        quaternion_normalization_jacobian(np.zeros(4))


@pytest.mark.parametrize(
    "q", [np.array([0.8, -0.1, 0.3, 0.5]), np.array([1.0, 0.0, 0.0, 0.0])]
)
def test_rotated_vector_quaternion_jacobian_matches_vector_rotation_finite_difference(
    q: np.ndarray,
) -> None:
    vector = np.array([1.3, -0.7, 2.1])
    # The rotation function expects a unit quaternion; choose a unit point for
    # the derivative check so the finite difference matches the helper's model.
    q = q / np.linalg.norm(q)
    function = lambda value: _rotate_vector(value, vector)
    actual = rotated_vector_quaternion_jacobian(q, vector)
    np.testing.assert_allclose(
        actual, _finite_difference_jacobian(function, q), rtol=3e-7, atol=3e-8
    )


@pytest.mark.parametrize(
    "q,expected",
    [
        ([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], [0.0, 0.0, np.pi / 2]),
        ([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], [np.pi / 2, 0.0, 0.0]),
    ],
)
def test_quaternion_to_euler321_identity_and_principal_axis_rotations(
    q: object, expected: object
) -> None:
    np.testing.assert_allclose(
        quaternion_to_euler321(np.asarray(q)), expected, atol=1e-14
    )


def test_quaternion_to_euler321_clamps_roundoff_at_pitch_singularity() -> None:
    q = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5) + 1e-15, 0.0])
    result = quaternion_to_euler321(q)
    assert np.all(np.isfinite(result))
    assert result[1] == pytest.approx(np.pi / 2)
    # At the singularity, roll and yaw are coupled and individually undefined;
    # assert the defined pitch and finite output only.
