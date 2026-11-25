import numpy as np
import pytest

from GenADCS.backends.python.helpers.math.dynamics import (
    rigid_body_angular_acceleration,
    wheel_coupled_angular_acceleration,
    wheel_momentum_derivative,
)


def test_rigid_body_angular_acceleration_matches_eulers_equation() -> None:
    omega = np.array([0.2, -0.4, 0.7])
    torque = np.array([0.03, -0.01, 0.02])
    inertia = np.array([[4.0, 0.2, -0.1], [0.2, 5.0, 0.3], [-0.1, 0.3, 6.0]])
    momentum = inertia @ omega
    expected = np.linalg.solve(inertia, torque - np.cross(omega, momentum))
    np.testing.assert_allclose(
        rigid_body_angular_acceleration(omega, torque, inertia), expected
    )


@pytest.mark.parametrize(
    "omega", [np.array([0.0, 0.0, 0.0]), np.array([0.1, -0.3, 0.8])]
)
def test_rigid_body_acceleration_for_spherical_inertia(omega: np.ndarray) -> None:
    inertia = 3.0 * np.eye(3)
    torque = np.array([0.3, -0.6, 0.9])
    # For spherical inertia, angular momentum is parallel to omega, so gyroscopic
    # coupling vanishes.
    np.testing.assert_allclose(
        rigid_body_angular_acceleration(omega, torque, inertia), torque / 3.0
    )


def test_rigid_body_acceleration_is_zero_without_torque_at_principal_axis_spin() -> None:
    inertia = np.diag([2.0, 3.0, 5.0])
    omega = np.array([0.0, 0.0, 0.7])
    np.testing.assert_allclose(
        rigid_body_angular_acceleration(omega, np.zeros(3), inertia), np.zeros(3)
    )


def test_wheel_coupled_acceleration_includes_wheel_momentum_gyroscopic_term() -> None:
    omega = np.array([0.3, -0.2, 0.5])
    wheel_momentum = np.array([0.04, -0.03])
    wheel_axes = np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])
    torque = np.array([0.01, 0.02, -0.04])
    inertia = np.array([[3.0, 0.1, 0.0], [0.1, 4.0, 0.2], [0.0, 0.2, 5.0]])
    total_momentum = inertia @ omega + wheel_axes.T @ wheel_momentum
    expected = np.linalg.solve(inertia, torque - np.cross(omega, total_momentum))
    actual = wheel_coupled_angular_acceleration(
        omega, wheel_momentum, wheel_axes, torque, inertia
    )
    np.testing.assert_allclose(actual, expected)
    without_wheel_momentum = wheel_coupled_angular_acceleration(
        omega, np.zeros_like(wheel_momentum), wheel_axes, torque, inertia
    )
    assert not np.allclose(actual, without_wheel_momentum)


def test_wheel_coupled_acceleration_reduces_to_rigid_body_without_wheels() -> None:
    omega = np.array([-0.2, 0.4, 0.1])
    inertia = np.diag([2.0, 4.0, 7.0])
    torque = np.array([0.03, -0.02, 0.01])
    actual = wheel_coupled_angular_acceleration(
        omega, np.zeros(0), np.zeros((0, 3)), torque, inertia
    )
    np.testing.assert_allclose(
        actual, rigid_body_angular_acceleration(omega, torque, inertia)
    )


def test_wheel_momentum_derivative_applies_axis_projection_of_bus_acceleration() -> None:
    wheel_torque = np.array([0.2, -0.1, 0.05])
    angular_acceleration = np.array([0.3, -0.4, 0.2])
    axes = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1 / np.sqrt(2), 0.0, 1 / np.sqrt(2)]]
    )
    np.testing.assert_allclose(
        wheel_momentum_derivative(wheel_torque, angular_acceleration, axes),
        wheel_torque - axes @ angular_acceleration,
    )


def test_wheel_momentum_derivative_supports_zero_wheels() -> None:
    result = wheel_momentum_derivative(
        np.zeros(0), np.array([1.0, 2.0, 3.0]), np.zeros((0, 3))
    )
    assert result.shape == (0,)
