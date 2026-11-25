"""Tests for the Numba three-axis specific-force accelerometer."""

import numpy as np

from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import Accelerometer
from GenADCS.backends.python.helpers.math.rotations.quaternion.algebra import _multiply
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import (
    _axis_angle_to_quaternion,
    _quaternion_to_dcm,
)


def make_sensor(
    noise=None, bias=None, limit=100.0, orientation=None, misalignment=None
):
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    return Accelerometer(
        np.zeros(3),
        identity if orientation is None else orientation,
        identity if misalignment is None else misalignment,
        limit,
        np.eye(3) * 1e-6 if noise is None else noise,
        np.zeros(3) if bias is None else bias,
    )


def test_specific_force_is_acceleration_minus_gravity_and_rotated_to_sensor():
    orientation = _axis_angle_to_quaternion(np.array([0.0, 0.0, 1.0]), 0.5)
    misalignment = _axis_angle_to_quaternion(np.array([1.0, 0.0, 0.0]), -0.2)
    attitude = _axis_angle_to_quaternion(np.array([0.0, 1.0, 0.0]), 0.3)
    acceleration, gravity = np.array([2.0, 3.0, 4.0]), np.array([0.5, 1.0, -1.0])
    sensor = make_sensor(orientation=orientation, misalignment=misalignment)
    mounting = _quaternion_to_dcm(_multiply(misalignment, orientation))
    expected = mounting @ (_quaternion_to_dcm(attitude) @ (acceleration - gravity))
    np.testing.assert_allclose(
        sensor.noiseless_biasfree_measurement(acceleration, gravity, attitude), expected
    )


def test_bias_and_noise_measurement_variants():
    bias = np.array([0.1, -0.2, 0.3])
    sensor = make_sensor(bias=bias)
    a, g, q = (
        np.array([2.0, 3.0, 4.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    base = sensor.noiseless_biasfree_measurement(a, g, q)
    np.testing.assert_allclose(
        sensor.noiseless_biased_measurement(a, g, q), base + bias
    )
    np.testing.assert_array_equal(sensor.jacobian_bias(), np.eye(3))
    for method in (sensor.noisy_biasfree_measurement, sensor.noisy_biased_measurement):
        sample = method(a, g, q)
        assert sample.shape == (3,) and np.all(np.isfinite(sample))


def test_attitude_jacobians_have_correct_zero_error_scaling():
    sensor = make_sensor()
    a, g, q = (
        np.array([2.0, 3.0, 4.0]),
        np.array([0.5, 1.0, -1.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    jq = sensor.jacobian_relative_quaternion(a, g, q, q)
    np.testing.assert_allclose(
        sensor.jacobian_relative_mrp(a, g, q, np.zeros(3)), 2 * jq
    )
    np.testing.assert_allclose(
        sensor.jacobian_relative_angle_vector(a, g, q, np.zeros(3)), 0.5 * jq
    )


def test_nonzero_attitude_jacobians_are_finite():
    sensor = make_sensor()
    a, g, q = np.ones(3), np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.0, 0.0, 0.0])
    dq = _axis_angle_to_quaternion(np.array([1.0, 2.0, 0.0]), 0.1)
    values = (
        sensor.jacobian_relative_quaternion(a, g, q, dq),
        sensor.jacobian_relative_mrp(a, g, q, np.array([0.01, -0.02, 0.03])),
        sensor.jacobian_relative_angle_vector(a, g, q, np.array([0.01, -0.02, 0.03])),
    )
    assert all(x.shape == (3, 3) and np.all(np.isfinite(x)) for x in values)


def test_sensor_saturation_and_availability():
    sensor = make_sensor(limit=0.5)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        sensor.noiseless_biasfree_measurement(np.ones(3), np.zeros(3), q),
        [0.5, 0.5, 0.5],
    )
    assert sensor.availability()
    sensor.enabled = False
    assert not sensor.availability()
