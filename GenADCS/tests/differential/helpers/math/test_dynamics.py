"""Pairwise comparisons for rigid body and wheel dynamics."""

from typing import Any

import numpy as np


def test_rigid_body_and_wheel_dynamics_match_pairwise(math_backends: Any) -> None:
    omega = np.array([0.2, -0.4, 0.7])
    torque = np.array([0.03, -0.01, 0.02])
    inertia = np.array([[4.0, 0.2, -0.1], [0.2, 5.0, 0.3], [-0.1, 0.3, 6.0]])
    wheel_axes = np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])
    wheel_momentum = np.array([0.04, -0.03])
    wheel_torque = np.array([0.2, -0.1])
    acceleration = np.array([0.3, -0.4, 0.2])
    momentum_axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    results = {
        name: (
            api.rigid_body_angular_acceleration(omega, torque, inertia),
            api.wheel_coupled_angular_acceleration(
                omega, wheel_momentum, wheel_axes, torque, inertia
            ),
            api.wheel_momentum_derivative(wheel_torque, acceleration, momentum_axes),
        )
        for name, api in math_backends.items()
    }
    reference = results["python"]
    for backend, values in results.items():
        for actual, expected in zip(values, reference):
            np.testing.assert_allclose(
                actual, expected, rtol=3e-13, atol=3e-14, err_msg=backend
            )


def test_empty_wheel_results_match_across_backends(math_backends: Any) -> None:
    omega = np.array([0.1, -0.2, 0.3])
    torque = np.array([0.01, 0.02, -0.03])
    inertia = np.diag([2.0, 3.0, 4.0])
    alpha = np.array([0.3, 0.2, -0.1])
    results = []
    for api in math_backends.values():
        results.append(
            (
                api.wheel_coupled_angular_acceleration(
                    omega, np.empty(0), np.empty((0, 3)), torque, inertia
                ),
                api.wheel_momentum_derivative(np.empty(0), alpha, np.empty((0, 3))),
            )
        )
    for rigid, derivative in results[1:]:
        np.testing.assert_allclose(rigid, results[0][0], rtol=3e-13, atol=3e-14)
        assert np.asarray(derivative).shape == results[0][1].shape == (0,)
