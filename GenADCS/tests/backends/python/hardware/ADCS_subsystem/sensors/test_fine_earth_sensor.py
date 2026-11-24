"""Tests for the two-axis fine Earth sensor measurement model."""
import numpy as np
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import FineEarthSensor
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import _axis_angle_to_quaternion, _quaternion_to_dcm
from GenADCS.backends.python.helpers.math.rotations.sensor_kernels import mounting_dcm


def make_sensor(noise=None, bias=None, fov=None, orientation=None, misalignment=None):
    identity = np.array([1., 0., 0., 0.])
    return FineEarthSensor(np.zeros(3), identity if orientation is None else orientation,
                           identity if misalignment is None else misalignment,
                           np.diag([1e-6, 2e-6]) if noise is None else noise,
                           np.zeros(2) if bias is None else bias,
                           np.array([.6, .7]) if fov is None else fov)


def test_angles_use_mounted_sensor_frame_vector():
    om = _axis_angle_to_quaternion(np.array([0., 1., 0.]), .2)
    mm = _axis_angle_to_quaternion(np.array([1., 0., 0.]), -.1)
    att = _axis_angle_to_quaternion(np.array([0., 0., 1.]), .3)
    earth = np.array([.2, -.1, .97]); earth /= np.linalg.norm(earth)
    sensor = make_sensor(orientation=om, misalignment=mm)
    sv = mounting_dcm(om, mm) @ (_quaternion_to_dcm(att) @ earth)
    expected = np.array([np.arctan2(sv[0], sv[2]), np.arctan2(sv[1], sv[2])])
    np.testing.assert_allclose(sensor.noiseless_biasfree_measurement(earth, att), expected)


def test_bias_noise_and_bias_jacobian():
    sensor = make_sensor(noise=np.diag([1e-8, 2e-8]), bias=np.array([-.01, .02]))
    earth, q = np.array([.1, .2, .97]), np.array([1., 0., 0., 0.])
    base = sensor.noiseless_biasfree_measurement(earth, q)
    np.testing.assert_allclose(sensor.noiseless_biased_measurement(earth, q), base + sensor.bias)
    np.testing.assert_array_equal(sensor.jacobian_bias(), np.eye(2))
    for method in (sensor.noisy_biasfree_measurement, sensor.noisy_biased_measurement):
        sample = method(earth, q)
        assert sample.shape == (2,) and np.all(np.isfinite(sample))


def test_attitude_jacobians_zero_limits_and_nonzero_shapes():
    sensor = make_sensor(); earth, q = np.array([.1, .2, .97]), np.array([1., 0., 0., 0.])
    jq = sensor.jacobian_relative_quaternion(earth, q, q)
    np.testing.assert_allclose(sensor.jacobian_relative_mrp(earth, q, np.zeros(3)), 2 * jq)
    np.testing.assert_allclose(sensor.jacobian_relative_angle_vector(earth, q, np.zeros(3)), .5 * jq)
    dq = _axis_angle_to_quaternion(np.array([1., 2., 0.]), .1)
    for result in (sensor.jacobian_relative_quaternion(earth, q, dq),
                   sensor.jacobian_relative_mrp(earth, q, np.array([.01, -.02, .03])),
                   sensor.jacobian_relative_angle_vector(earth, q, np.array([.01, -.02, .03]))):
        assert result.shape == (2, 3) and np.all(np.isfinite(result))


def test_availability_respects_earth_visibility_and_fov():
    sensor = make_sensor(fov=np.array([.2, .3])); q = np.array([1., 0., 0., 0.])
    assert sensor.availability(np.array([0., 0., 1.]), q)
    assert not sensor.availability(np.array([0., 0., 1.]), q, False)
    assert not sensor.availability(np.array([np.sin(.25), 0., np.cos(.25)]), q)
    assert not sensor.availability(np.array([0., np.sin(.4), np.cos(.4)]), q)


def test_fisher_information_uses_k_matrix_and_mounting():
    covariance = np.diag([.01, .02]); mm = _axis_angle_to_quaternion(np.array([1., 0., 0.]), .1)
    sensor = make_sensor(noise=covariance, misalignment=mm)
    earth, q = np.array([.1, .2, .97]), np.array([1., 0., 0., 0.])
    alpha, beta = sensor.noiseless_biasfree_measurement(earth, q); u, v = np.tan(alpha), np.tan(beta)
    k = np.array([[-u*v/(1+u*u), 1., -v/(1+u*u)], [-1., u*v/(1+v*v), u/(1+v*v)]])
    rot = mounting_dcm(np.array([1., 0., 0., 0.]), mm)
    np.testing.assert_allclose(sensor.fisher_information_matrix(earth, q), rot.T @ k.T @ np.linalg.solve(covariance, k) @ rot)
    np.testing.assert_array_equal(sensor.fisher_information_matrix(earth, q, False), np.zeros((3, 3)))
