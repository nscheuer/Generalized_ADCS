"""Tests for the six-component inertial GNSS sensor."""

import numpy as np
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import GNSS


def make_sensor(noise=None, bias=None):
    return GNSS(
        np.eye(6) if noise is None else noise, np.zeros(6) if bias is None else bias
    )


def test_measurement_concatenates_inertial_position_then_velocity():
    sensor = make_sensor()
    r = np.array([1.0, 2.0, 3.0])
    v = np.array([-4.0, 5.0, 6.0])
    np.testing.assert_array_equal(
        sensor.noiseless_biasfree_measurement(r, v), [1.0, 2.0, 3.0, -4.0, 5.0, 6.0]
    )


def test_bias_is_additive_over_position_and_velocity_channels():
    bias = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    sensor = make_sensor(bias=bias)
    r, v = np.arange(3.0, dtype=float), np.arange(3.0, 6.0, dtype=float)
    np.testing.assert_allclose(
        sensor.noiseless_biased_measurement(r, v), np.r_[r, v] + bias
    )
    np.testing.assert_array_equal(sensor.jacobian_bias(), np.eye(6))


def test_noise_variants_return_finite_six_vector():
    sensor = make_sensor(noise=np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    r, v = np.zeros(3), np.ones(3)
    for method in (sensor.noisy_biasfree_measurement, sensor.noisy_biased_measurement):
        result = method(r, v)
        assert result.shape == (6,) and np.all(np.isfinite(result))


def test_position_velocity_jacobians_have_block_structure():
    sensor = make_sensor()
    jp, jv = sensor.jacobian_position(), sensor.jacobian_velocity()
    assert jp.shape == jv.shape == (6, 3)
    np.testing.assert_array_equal(jp[:3], np.eye(3))
    np.testing.assert_array_equal(jp[3:], np.zeros((3, 3)))
    np.testing.assert_array_equal(jv[:3], np.zeros((3, 3)))
    np.testing.assert_array_equal(jv[3:], np.eye(3))


def test_attitude_jacobians_are_zero_and_availability_is_live():
    sensor = make_sensor()
    for jac in (
        sensor.jacobian_relative_quaternion(),
        sensor.jacobian_relative_mrp(),
        sensor.jacobian_relative_angle_vector(),
    ):
        np.testing.assert_array_equal(jac, np.zeros((6, 3)))
    assert sensor.availability()
    sensor.enabled = False
    assert not sensor.availability()
