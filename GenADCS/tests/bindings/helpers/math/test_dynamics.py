"""Dynamics binding conversion, dynamic shapes, errors, and ownership."""

import gc

import numpy as np
import pytest


def test_rigid_body_dynamics_accept_lists_and_numpy_inputs(math_backend):
    omega = np.zeros(3, dtype=np.float32)
    torque = np.array([0.03, -0.01, 0.02], dtype=np.float32)
    inertia = np.diag([2.0, 3.0, 5.0]).astype(np.float32)
    expected = torque.astype(float) / np.diag(inertia)
    result = math_backend.rigid_body_angular_acceleration(omega, torque, inertia)
    assert result.shape == (3,) and result.dtype == np.float64
    np.testing.assert_allclose(result, expected, atol=1e-14)
    np.testing.assert_allclose(
        math_backend.rigid_body_angular_acceleration(
            omega.tolist(), torque.tolist(), inertia.tolist()
        ),
        result,
    )


def test_wheel_coupled_dynamic_inputs_and_zero_wheel_shape(math_backend):
    omega = np.array([0.3, -0.2, 0.5])
    bus_inertia = np.diag([3.0, 4.0, 5.0])
    torque = np.array([0.01, 0.02, -0.04])
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]], dtype=np.float32)
    momentum = np.array([0.04, -0.03], dtype=np.float32)
    original = (omega.copy(), torque.copy(), axes.copy(), momentum.copy())
    result = math_backend.wheel_coupled_angular_acceleration(
        omega, momentum, axes, torque, bus_inertia
    )
    total_momentum = bus_inertia @ omega + axes.T @ momentum
    expected = np.linalg.solve(bus_inertia, torque - np.cross(omega, total_momentum))
    assert result.shape == (3,) and result.dtype == np.float64
    np.testing.assert_allclose(result, expected, atol=1e-14)
    for actual, saved in zip((omega, torque, axes, momentum), original):
        np.testing.assert_array_equal(actual, saved)

    no_wheel = math_backend.wheel_coupled_angular_acceleration(
        omega, np.empty(0), np.empty((0, 3)), torque, bus_inertia
    )
    np.testing.assert_allclose(
        no_wheel,
        math_backend.rigid_body_angular_acceleration(omega, torque, bus_inertia),
    )


def test_wheel_momentum_derivative_dynamic_result_ownership(math_backend):
    wheel_torque = np.array([0.2, -0.1, 0.05], dtype=np.float32)
    acceleration = np.array([0.3, -0.4, 0.2])
    axes = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1 / np.sqrt(2), 0.0, 1 / np.sqrt(2)]]
    )
    result = math_backend.wheel_momentum_derivative(wheel_torque, acceleration, axes)
    assert result.shape == (3,) and result.dtype == np.float64
    np.testing.assert_allclose(result, wheel_torque - axes @ acceleration)
    del wheel_torque, acceleration, axes
    gc.collect()
    np.testing.assert_allclose(result, [-0.1, 0.3, 0.05 - 0.5 / np.sqrt(2)])
    result[0] = 9.0
    assert result[0] == 9.0


def test_mismatched_wheel_axis_shapes_raise_value_error(math_backend):
    with pytest.raises(ValueError):
        math_backend.wheel_momentum_derivative([0.1], [0.0, 0.0, 0.0], np.zeros((2, 3)))
    with pytest.raises(ValueError):
        math_backend.wheel_coupled_angular_acceleration(
            [0.0, 0.0, 0.0], [0.1], np.zeros((2, 3)), [0.0, 0.0, 0.0], np.eye(3)
        )


def test_singular_inertia_maps_to_python_value_error(math_backend):
    with pytest.raises(ValueError):
        math_backend.rigid_body_angular_acceleration(
            [0.1, 0.2, 0.3], [0.0, 0.0, 0.0], np.zeros((3, 3))
        )


def test_dynamics_results_survive_temporary_inputs(math_backend):
    result = math_backend.rigid_body_angular_acceleration(
        np.array([0.0, 0.0, 0.5]), np.array([0.2, -0.4, 0.6]), np.diag([2.0, 4.0, 6.0])
    )
    gc.collect()
    np.testing.assert_allclose(result, [0.1, -0.1, 0.1])
