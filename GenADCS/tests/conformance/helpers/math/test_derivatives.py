"""Mathematical and validation contract for derivatives across backends."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest


def _rotate_raw(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    scalar, qv = q[0], q[1:]
    return vector + 2.0 * (
        scalar * np.cross(qv, vector) + np.cross(qv, np.cross(qv, vector))
    )


def _finite_difference(
    function: Callable[[np.ndarray], np.ndarray], x: np.ndarray, step: float = 1e-7
) -> np.ndarray:
    result = np.empty((np.asarray(function(x)).size, x.size))
    for column in range(x.size):
        delta = np.zeros_like(x)
        delta[column] = step
        result[:, column] = (function(x + delta) - function(x - delta)) / (2.0 * step)
    return result


def test_quaternion_rate_and_inverse_contract(math_api: Any) -> None:
    q = np.array([0.7, -0.2, 0.3, 0.6])
    q /= np.linalg.norm(q)
    omega = np.array([0.4, -0.7, 0.2])
    qdot = math_api.quaternion_derivative(q, omega)
    rate = math_api.quaternion_rate_matrix(q)
    assert rate.shape == (4, 3)
    np.testing.assert_allclose(qdot, rate @ omega, atol=2e-14)
    np.testing.assert_allclose(
        math_api.angular_velocity_from_quaternion_derivative(q, qdot), omega, atol=2e-14
    )
    assert np.dot(q, qdot) == pytest.approx(0.0, abs=2e-14)


def test_quaternion_normalization_jacobian_is_the_derivative_of_normalize(math_api: Any) -> None:
    q = np.array([0.4, -1.2, 0.7, 2.0])
    actual = math_api.quaternion_normalization_jacobian(q)
    expected = _finite_difference(lambda x: x / np.linalg.norm(x), q)
    np.testing.assert_allclose(actual, expected, rtol=3e-8, atol=3e-9)
    np.testing.assert_allclose(actual @ q, np.zeros(4), atol=2e-14)


def test_rotated_vector_jacobian_matches_finite_difference(math_api: Any) -> None:
    q = np.array([0.8, -0.1, 0.3, 0.5])
    q /= np.linalg.norm(q)
    vector = np.array([1.3, -0.7, 2.1])
    actual = math_api.rotated_vector_quaternion_jacobian(q, vector)
    expected = _finite_difference(lambda x: _rotate_raw(x, vector), q)
    np.testing.assert_allclose(actual, expected, rtol=4e-7, atol=4e-8)


def test_euler321_reference_rotations_and_zero_quaternion_validation(math_api: Any) -> None:
    half = np.sqrt(0.5)
    np.testing.assert_allclose(
        math_api.quaternion_to_euler321([1.0, 0.0, 0.0, 0.0]),
        [0.0, 0.0, 0.0],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        math_api.quaternion_to_euler321([half, half, 0.0, 0.0]),
        [np.pi / 2, 0, 0],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        math_api.quaternion_to_euler321([half, 0.0, 0.0, half]),
        [0, 0, np.pi / 2],
        atol=1e-14,
    )
    with pytest.raises(ValueError, match="nonzero norm"):
        math_api.quaternion_normalization_jacobian(np.zeros(4))
