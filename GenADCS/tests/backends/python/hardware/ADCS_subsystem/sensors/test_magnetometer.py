"""Behavior and Jacobian tests for the compiled three-axis magnetometer."""

from typing import Any

import numpy as np
import pytest

from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import Magnetometer
from GenADCS.backends.python.helpers.math.rotations.quaternion.algebra import _multiply
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import (
    _axis_angle_to_quaternion,
    _quaternion_to_dcm,
)


def make_sensor(
    *,
    limit: float = 1.0,
    noise: Any = None,
    bias: Any = None,
    orientation: Any = None,
    misalignment: Any = None,
) -> Magnetometer:
    return Magnetometer(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation,
        np.array([1.0, 0.0, 0.0, 0.0]) if misalignment is None else misalignment,
        limit,
        np.eye(3) * 1e-12 if noise is None else noise,
        np.zeros(3) if bias is None else bias,
    )


def test_noiseless_measurements_rotate_inertial_field_to_sensor_frame() -> None:
    q_att = _axis_angle_to_quaternion(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    q_mount = _axis_angle_to_quaternion(np.array([1.0, 0.0, 0.0]), np.pi / 2)
    q_misalign = _axis_angle_to_quaternion(np.array([0.0, 1.0, 0.0]), np.pi / 2)
    field_i = np.array([2e-5, -3e-5, 5e-5])
    sensor = make_sensor(limit=1.0, orientation=q_mount, misalignment=q_misalign)
    expected = _quaternion_to_dcm(_multiply(q_misalign, q_mount)) @ (
        _quaternion_to_dcm(q_att) @ field_i
    )
    np.testing.assert_allclose(
        sensor.noiseless_biasfree_measurement(field_i, q_att), expected
    )
    np.testing.assert_allclose(
        sensor.noiseless_biased_measurement(field_i, q_att), expected
    )


def test_bias_is_additive_before_componentwise_saturation() -> None:
    sensor = make_sensor(limit=4e-5, bias=np.array([1e-5, -2e-5, 0.0]))
    field = np.array([4e-5, -4e-5, 2e-5])
    np.testing.assert_allclose(
        sensor.noiseless_biased_measurement(field, np.array([1.0, 0.0, 0.0, 0.0])),
        [4e-5, -4e-5, 2e-5],
    )
    np.testing.assert_allclose(
        sensor.noiseless_biasfree_measurement(field, np.array([1.0, 0.0, 0.0, 0.0])),
        [4e-5, -4e-5, 2e-5],
    )
    np.testing.assert_allclose(sensor.jacobian_bias(), np.eye(3))


def test_measurement_saturates_each_axis() -> None:
    sensor = make_sensor(limit=3e-5)
    out = sensor.noiseless_biasfree_measurement(
        np.array([5e-5, -4e-5, 1e-5]), np.array([1.0, 0.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(out, [3e-5, -3e-5, 1e-5])


@pytest.mark.parametrize(
    "method", ["noisy_biasfree_measurement", "noisy_biased_measurement"]
)
def test_noise_is_finite_and_measurement_shaped(method: str) -> None:
    sensor = make_sensor(
        noise=np.diag([1e-10, 4e-10, 9e-10]), bias=np.array([1e-6, 2e-6, 3e-6])
    )
    result = getattr(sensor, method)(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_attitude_jacobians_match_small_angle_finite_difference() -> None:
    sensor = make_sensor()
    q = np.array([1.0, 0.0, 0.0, 0.0])
    field = np.array([2e-5, -3e-5, 4e-5])
    expected = -2.0 * np.array(
        [[0.0, -4e-5, -3e-5], [4e-5, 0.0, -2e-5], [3e-5, 2e-5, 0.0]]
    )
    np.testing.assert_allclose(
        sensor.jacobian_relative_quaternion(field, q, q), expected, atol=1e-14
    )
    np.testing.assert_allclose(
        sensor.jacobian_relative_mrp(field, q, np.zeros(3)), 2.0 * expected, atol=1e-14
    )
    np.testing.assert_allclose(
        sensor.jacobian_relative_angle_vector(field, q, np.zeros(3)),
        0.5 * expected,
        atol=1e-14,
    )


def test_nonzero_relative_parameter_jacobians_are_finite() -> None:
    sensor = make_sensor()
    field, q = np.array([2e-5, 1e-5, -3e-5]), np.array([1.0, 0.0, 0.0, 0.0])
    delta_q = _axis_angle_to_quaternion(np.array([1.0, 2.0, -1.0]), 0.2)
    for jac in (
        sensor.jacobian_relative_quaternion(field, q, delta_q),
        sensor.jacobian_relative_mrp(field, q, np.array([0.02, -0.01, 0.03])),
        sensor.jacobian_relative_angle_vector(field, q, np.array([0.02, -0.01, 0.03])),
    ):
        assert jac.shape == (3, 3)
        assert np.all(np.isfinite(jac))


def test_availability_and_fisher_information() -> None:
    sensor = make_sensor(noise=np.diag([2.0, 3.0, 4.0]))
    field, q = np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0, 0.0])
    assert sensor.availability()
    covariance = np.diag([0.1, 0.2, 0.3])
    rotation = np.eye(3)
    cross = np.array([[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]])
    expected = (
        rotation.T
        @ cross.T
        @ np.linalg.solve(sensor.noise + covariance, cross)
        @ rotation
    )
    np.testing.assert_allclose(
        sensor.fisher_information_matrix(field, q, covariance), expected
    )
