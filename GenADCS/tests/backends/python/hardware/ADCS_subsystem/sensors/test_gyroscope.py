"""Behavior and Jacobian tests for the compiled three-axis gyroscope."""

import numpy as np
import pytest

from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import Gyroscope
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import _axis_angle_to_quaternion, _quaternion_to_dcm
from GenADCS.backends.python.helpers.math.rotations.quaternion.algebra import _multiply


def make_sensor(*, limit=100., noise=None, bias=None, orientation=None, misalignment=None):
    return Gyroscope(
        np.zeros(3),
        np.array([1., 0., 0., 0.]) if orientation is None else orientation,
        np.array([1., 0., 0., 0.]) if misalignment is None else misalignment,
        limit,
        np.eye(3) * 1e-8 if noise is None else noise,
        np.zeros(3) if bias is None else bias,
    )


def test_noiseless_measurements_apply_mounting_and_misalignment_rotations():
    orientation = _axis_angle_to_quaternion(np.array([0., 0., 1.]), np.pi / 2)
    misalignment = _axis_angle_to_quaternion(np.array([1., 0., 0.]), np.pi / 2)
    omega = np.array([.1, -.2, .3])
    sensor = make_sensor(orientation=orientation, misalignment=misalignment)
    expected = _quaternion_to_dcm(_multiply(misalignment, orientation)) @ omega
    np.testing.assert_allclose(sensor.noiseless_biasfree_measurement(omega), expected)
    np.testing.assert_allclose(sensor.noiseless_biased_measurement(omega), expected)


def test_bias_is_additive_and_bias_jacobian_is_identity():
    bias = np.array([.01, -.02, .03])
    sensor = make_sensor(bias=bias)
    omega = np.array([.1, .2, -.3])
    np.testing.assert_allclose(sensor.noiseless_biased_measurement(omega), omega + bias)
    np.testing.assert_allclose(sensor.jacobian_bias(), np.eye(3))


def test_measurement_saturates_each_axis():
    sensor = make_sensor(limit=.25)
    np.testing.assert_allclose(sensor.noiseless_biasfree_measurement(np.array([.5, -.4, .1])), [.25, -.25, .1])


@pytest.mark.parametrize("method", ["noisy_biasfree_measurement", "noisy_biased_measurement"])
def test_noise_measurements_are_finite_and_shaped(method):
    sensor = make_sensor(noise=np.diag([1e-6, 4e-6, 9e-6]), bias=np.array([.01, 0., -.01]))
    result = getattr(sensor, method)(np.array([.1, .2, .3]))
    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_rate_jacobian_and_availability_contract():
    sensor = make_sensor()
    np.testing.assert_array_equal(sensor.jacobian_angular_velocity(), np.eye(3))
    assert sensor.availability()
