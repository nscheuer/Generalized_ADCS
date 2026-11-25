"""Tests for the two-axis fine Sun sensor measurement model."""

import numpy as np

from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import FineSunSensor
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import (
    _axis_angle_to_quaternion,
    _quaternion_to_dcm,
)
from GenADCS.backends.python.helpers.math.rotations.sensor_kernels import mounting_dcm


def make_sensor(noise=None, bias=None, fov=None, orientation=None, misalignment=None):
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    return FineSunSensor(
        np.zeros(3),
        identity if orientation is None else orientation,
        identity if misalignment is None else misalignment,
        np.diag([1e-6, 2e-6]) if noise is None else noise,
        np.zeros(2) if bias is None else bias,
        np.array([0.6, 0.7]) if fov is None else fov,
    )


def test_angles_use_mounted_sensor_frame_vector():
    om = _axis_angle_to_quaternion(np.array([0.0, 1.0, 0.0]), 0.2)
    mm = _axis_angle_to_quaternion(np.array([1.0, 0.0, 0.0]), -0.1)
    att = _axis_angle_to_quaternion(np.array([0.0, 0.0, 1.0]), 0.3)
    sun = np.array([0.2, -0.1, 0.97])
    sun /= np.linalg.norm(sun)
    sensor = make_sensor(orientation=om, misalignment=mm)
    sv = mounting_dcm(om, mm) @ (_quaternion_to_dcm(att) @ sun)
    expected = np.array([np.arctan2(sv[0], sv[2]), np.arctan2(sv[1], sv[2])])
    np.testing.assert_allclose(
        sensor.noiseless_biasfree_measurement(sun, att), expected
    )


def test_bias_noise_and_bias_jacobian():
    sensor = make_sensor(noise=np.diag([1e-8, 2e-8]), bias=np.array([0.01, -0.02]))
    sun, q = np.array([0.1, 0.2, 0.97]), np.array([1.0, 0.0, 0.0, 0.0])
    base = sensor.noiseless_biasfree_measurement(sun, q)
    np.testing.assert_allclose(
        sensor.noiseless_biased_measurement(sun, q), base + sensor.bias
    )
    np.testing.assert_array_equal(sensor.jacobian_bias(), np.eye(2))
    for method in (sensor.noisy_biasfree_measurement, sensor.noisy_biased_measurement):
        sample = method(sun, q)
        assert sample.shape == (2,) and np.all(np.isfinite(sample))


def test_attitude_jacobian_zero_limits_and_nonzero_shapes():
    sensor = make_sensor()
    sun, q = np.array([0.1, 0.2, 0.97]), np.array([1.0, 0.0, 0.0, 0.0])
    jq = sensor.jacobian_relative_quaternion(sun, q, q)
    np.testing.assert_allclose(
        sensor.jacobian_relative_mrp(sun, q, np.zeros(3)), 2 * jq
    )
    np.testing.assert_allclose(
        sensor.jacobian_relative_angle_vector(sun, q, np.zeros(3)), 0.5 * jq
    )
    dq = _axis_angle_to_quaternion(np.array([1.0, 2.0, 0.0]), 0.1)
    for result in (
        sensor.jacobian_relative_quaternion(sun, q, dq),
        sensor.jacobian_relative_mrp(sun, q, np.array([0.01, -0.02, 0.03])),
        sensor.jacobian_relative_angle_vector(sun, q, np.array([0.01, -0.02, 0.03])),
    ):
        assert result.shape == (2, 3) and np.all(np.isfinite(result))


def test_nonzero_angle_jacobian_matches_finite_difference():
    sensor = make_sensor()
    sun, attitude = np.array([0.1, -0.2, 0.97]), np.array([1.0, 0.0, 0.0, 0.0])
    delta = np.array([0.04, -0.03, 0.02])
    step = 1e-6
    numeric = np.empty((2, 3))
    for column in range(3):
        offset = np.zeros(3)
        offset[column] = step

        def angles(error):
            q = _axis_angle_to_quaternion(
                error / np.linalg.norm(error), np.linalg.norm(error)
            )
            vector = _quaternion_to_dcm(q) @ sun
            return sensor.noiseless_biasfree_measurement(vector, attitude)

        numeric[:, column] = (angles(delta + offset) - angles(delta - offset)) / (
            2 * step
        )
    np.testing.assert_allclose(
        sensor.jacobian_relative_angle_vector(sun, attitude, delta),
        numeric,
        rtol=1e-3,
        atol=1e-5,
    )


def test_availability_respects_visibility_and_each_fov_axis():
    sensor = make_sensor(fov=np.array([0.2, 0.3]))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert sensor.availability(np.array([0.0, 0.0, 1.0]), q)
    assert not sensor.availability(np.array([0.0, 0.0, 1.0]), q, False)
    assert not sensor.availability(np.array([np.sin(0.25), 0.0, np.cos(0.25)]), q)
    assert not sensor.availability(np.array([0.0, np.sin(0.4), np.cos(0.4)]), q)


def test_fisher_information_uses_k_matrix_and_mounting():
    covariance = np.diag([0.01, 0.02])
    om = _axis_angle_to_quaternion(np.array([0.0, 0.0, 1.0]), 0.3)
    sensor = make_sensor(noise=covariance, orientation=om)
    sun, q = np.array([0.1, 0.2, 0.97]), np.array([1.0, 0.0, 0.0, 0.0])
    alpha, beta = sensor.noiseless_biasfree_measurement(sun, q)
    u, v = np.tan(alpha), np.tan(beta)
    k = np.array(
        [
            [-u * v / (1 + u * u), 1.0, -v / (1 + u * u)],
            [-1.0, u * v / (1 + v * v), u / (1 + v * v)],
        ]
    )
    rot = mounting_dcm(om, np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(
        sensor.fisher_information_matrix(sun, q),
        rot.T @ k.T @ np.linalg.solve(covariance, k) @ rot,
    )
    np.testing.assert_array_equal(
        sensor.fisher_information_matrix(sun, q, False), np.zeros((3, 3))
    )
