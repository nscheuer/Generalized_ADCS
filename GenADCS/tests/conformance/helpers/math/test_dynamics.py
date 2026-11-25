"""Rigid body and wheel dynamics contract across backends."""

import numpy as np


def test_rigid_body_acceleration_satisfies_eulers_equation(math_api):
    omega = np.array([0.2, -0.4, 0.7])
    torque = np.array([0.03, -0.01, 0.02])
    inertia = np.array([[4.0, 0.2, -0.1], [0.2, 5.0, 0.3], [-0.1, 0.3, 6.0]])
    angular_momentum = inertia @ omega
    expected = np.linalg.solve(inertia, torque - np.cross(omega, angular_momentum))
    actual = math_api.rigid_body_angular_acceleration(omega, torque, inertia)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)


def test_spherical_inertia_has_no_gyroscopic_coupling(math_api):
    omega = np.array([0.2, -0.4, 0.7])
    torque = np.array([0.3, -0.6, 0.9])
    np.testing.assert_allclose(
        math_api.rigid_body_angular_acceleration(omega, torque, 3.0 * np.eye(3)),
        torque / 3.0,
        atol=2e-14,
    )


def test_wheel_coupled_acceleration_uses_total_momentum(math_api):
    omega = np.array([0.3, -0.2, 0.5])
    momentum = np.array([0.04, -0.03])
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])
    torque = np.array([0.01, 0.02, -0.04])
    inertia = np.array([[3.0, 0.1, 0.0], [0.1, 4.0, 0.2], [0.0, 0.2, 5.0]])
    total_momentum = inertia @ omega + axes.T @ momentum
    expected = np.linalg.solve(inertia, torque - np.cross(omega, total_momentum))
    np.testing.assert_allclose(
        math_api.wheel_coupled_angular_acceleration(
            omega, momentum, axes, torque, inertia
        ),
        expected,
        rtol=2e-13,
        atol=2e-14,
    )


def test_wheel_momentum_derivative_and_empty_wheel_contract(math_api):
    wheel_torque = np.array([0.2, -0.1, 0.05])
    acceleration = np.array([0.3, -0.4, 0.2])
    axes = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1 / np.sqrt(2), 0.0, 1 / np.sqrt(2)]]
    )
    np.testing.assert_allclose(
        math_api.wheel_momentum_derivative(wheel_torque, acceleration, axes),
        wheel_torque - axes @ acceleration,
        atol=2e-14,
    )
    empty = math_api.wheel_momentum_derivative(
        np.empty(0), acceleration, np.empty((0, 3))
    )
    assert np.asarray(empty).shape == (0,)
