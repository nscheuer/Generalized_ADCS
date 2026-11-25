"""Detailed tests for the Numba coarse photodiode Sun sensor."""

import numpy as np

from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import CoarseSunSensor
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import (
    _axis_angle_to_quaternion,
)
from GenADCS.backends.python.helpers.math.rotations.sensor_kernels import mounting_dcm


def make_sensor(
    *,
    normal=None,
    sensitivity=2.0,
    limit=10.0,
    variance=0.01,
    bias=0.0,
    fov=np.pi / 3.0,
    orientation=None,
    misalignment=None,
):
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    return CoarseSunSensor(
        np.zeros(3),
        identity if orientation is None else orientation,
        identity if misalignment is None else misalignment,
        np.array([1.0, 0.0, 0.0]) if normal is None else normal,
        sensitivity,
        limit,
        variance,
        bias,
        fov,
    )


def test_measurement_is_cosine_projection_in_sensor_frame():
    orientation = _axis_angle_to_quaternion(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)
    misalignment = _axis_angle_to_quaternion(np.array([1.0, 0.0, 0.0]), 0.2)
    sensor = make_sensor(orientation=orientation, misalignment=misalignment)
    sun_i = np.array([0.0, 1.0, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    sun_body = sun_i
    normal_body = mounting_dcm(orientation, misalignment).T @ np.array([1.0, 0.0, 0.0])
    expected = 2.0 * max(0.0, np.dot(normal_body, sun_body))
    np.testing.assert_allclose(
        sensor.noiseless_biasfree_measurement(sun_i, q), expected
    )


def test_backside_illumination_reads_zero_and_bias_clamps_nonnegative():
    sensor = make_sensor(bias=-0.5)
    sun = np.array([-1.0, 0.0, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert sensor.noiseless_biasfree_measurement(sun, q) == 0.0
    assert sensor.noiseless_biased_measurement(sun, q) == 0.0


def test_sensitivity_bias_and_saturation_are_applied():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    sun = np.array([1.0, 0.0, 0.0])
    sensor = make_sensor(sensitivity=3.0, limit=2.0, bias=0.25)
    assert sensor.noiseless_biasfree_measurement(sun, q) == 2.0
    assert sensor.noiseless_biased_measurement(sun, q) == 2.0
    assert sensor.jacobian_bias() == 1.0


def test_noise_variants_return_finite_scalar_measurements_in_range():
    sensor = make_sensor(variance=1e-8, limit=3.0)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    sun = np.array([0.5, 0.0, np.sqrt(0.75)])
    for method in (sensor.noisy_biasfree_measurement, sensor.noisy_biased_measurement):
        result = method(sun, q)
        assert np.isfinite(result)
        assert 0.0 <= result <= sensor.saturation_value


def test_quaternion_mrp_and_angle_jacobians_have_expected_zero_limits():
    sensor = make_sensor(normal=np.array([0.0, 1.0, 0.0]))
    sun = np.array([0.0, 0.0, 1.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    jac_q = sensor.jacobian_relative_quaternion(sun, q, q)
    jac_mrp = sensor.jacobian_relative_mrp(sun, q, np.zeros(3))
    jac_angle = sensor.jacobian_relative_angle_vector(sun, q, np.zeros(3))
    assert jac_q.shape == jac_mrp.shape == jac_angle.shape == (3,)
    np.testing.assert_allclose(jac_mrp, 2.0 * jac_q)
    np.testing.assert_allclose(jac_angle, 0.5 * jac_q)


def test_nonzero_attitude_jacobians_are_finite():
    sensor = make_sensor()
    sun = np.array([0.2, -0.1, 0.97])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    values = (
        sensor.jacobian_relative_quaternion(
            sun, q, _axis_angle_to_quaternion(np.array([1.0, 0.0, 0.0]), 0.1)
        ),
        sensor.jacobian_relative_mrp(sun, q, np.array([0.02, -0.03, 0.01])),
        sensor.jacobian_relative_angle_vector(sun, q, np.array([0.02, -0.03, 0.01])),
    )
    assert all(value.shape == (3,) and np.all(np.isfinite(value)) for value in values)


def test_availability_respects_visibility_frontside_and_fov():
    sensor = make_sensor(normal=np.array([0.0, 0.0, 1.0]), fov=0.3)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert sensor.availability(np.array([0.0, 0.0, 1.0]), q)
    assert not sensor.availability(np.array([0.0, 0.0, 1.0]), q, False)
    assert not sensor.availability(np.array([0.0, 0.0, -1.0]), q)
    outside = np.array([np.sin(0.5), 0.0, np.cos(0.5)])
    assert not sensor.availability(outside, q)


def test_fisher_information_is_rank_one_and_zero_when_unavailable():
    sensor = make_sensor(
        normal=np.array([0.0, 0.0, 1.0]), sensitivity=2.0, variance=0.5
    )
    q = np.array([1.0, 0.0, 0.0, 0.0])
    sun = np.array([0.0, np.sin(0.2), np.cos(0.2)])
    normal_body = np.array([0.0, 0.0, 1.0])
    direction = np.cross(normal_body, sun)
    expected = (2.0**2 / 0.5) * np.outer(direction, direction)
    np.testing.assert_allclose(sensor.fisher_information_matrix(sun, q), expected)
    np.testing.assert_array_equal(
        sensor.fisher_information_matrix(sun, q, False), np.zeros((3, 3))
    )
