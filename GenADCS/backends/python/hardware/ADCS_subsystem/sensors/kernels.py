"""Cached Numba measurement kernels for use from sensor wrappers or a solver."""

import numpy as np
from numba import njit

from GenADCS.backends.python.bias.bias import add_bias
from GenADCS.backends.python.noise.noise import add_gaussian_noise
from GenADCS.backends.python.helpers.random.philox import standard_normals
from GenADCS.backends.python.helpers.math.rotations.sensor_kernels import (
    coarse_sun_measurement,
    fine_sun_angles,
    mounting_dcm,
    rotate_body_to_sensor,
    rotate_inertial_to_sensor,
    sun_in_body,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.algebra import (
    _conjugate,
    _multiply,
    _normalize_quaternion,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import (
    _quaternion_to_dcm,
)
from GenADCS.backends.python.helpers.math.vector import saturate_vector
from GenADCS.backends.python.helpers.math.vector import skew


@njit(cache=True)
def vector_measurement(
    vector,
    attitude,
    orientation,
    misalignment,
    bias,
    noise_cholesky,
    saturation,
    with_bias,
    with_noise,
    noise_draw=None,
):
    """Magnetometer or accelerometer measurement in sensor coordinates."""
    result = rotate_inertial_to_sensor(vector, attitude, orientation, misalignment)
    if with_bias:
        result = add_bias(result, bias)
    if with_noise:
        result = (
            add_gaussian_noise(result, noise_cholesky)
            if noise_draw is None
            else result + noise_cholesky @ noise_draw
        )
    return saturate_vector(result, saturation)


@njit(cache=True)
def rate_measurement(
    angular_velocity_body,
    orientation,
    misalignment,
    bias,
    noise_cholesky,
    saturation,
    with_bias,
    with_noise,
    noise_draw=None,
):
    result = rotate_body_to_sensor(angular_velocity_body, orientation, misalignment)
    if with_bias:
        result = add_bias(result, bias)
    if with_noise:
        result = (
            add_gaussian_noise(result, noise_cholesky)
            if noise_draw is None
            else result + noise_cholesky @ noise_draw
        )
    return saturate_vector(result, saturation)


@njit(cache=True)
def scalar_sun_measurement(
    sun_inertial,
    attitude,
    orientation,
    misalignment,
    normal,
    sensitivity,
    bias,
    noise_std,
    saturation,
    with_bias,
    with_noise,
    noise_draw=None,
):
    result = coarse_sun_measurement(
        sun_inertial, attitude, orientation, misalignment, normal, sensitivity
    )
    if with_bias:
        result += bias
    if with_noise:
        result += noise_std * (
            standard_normals(np.uint64(0), 0, 0, 1)[0]
            if noise_draw is None
            else noise_draw[0]
        )
    return min(max(result, 0.0), saturation)


@njit(cache=True)
def fine_direction_measurement(
    direction_inertial,
    attitude,
    orientation,
    misalignment,
    bias,
    noise_cholesky,
    with_bias,
    with_noise,
    noise_draw=None,
):
    result = fine_sun_angles(direction_inertial, attitude, orientation, misalignment)
    if with_bias:
        result = add_bias(result, bias)
    if with_noise:
        result = (
            add_gaussian_noise(result, noise_cholesky)
            if noise_draw is None
            else result + noise_cholesky @ noise_draw
        )
    return result


@njit(cache=True)
def gnss_measurement(
    position_inertial,
    velocity_inertial,
    bias,
    noise_cholesky,
    with_bias,
    with_noise,
    noise_draw=None,
):
    result = np.concatenate((position_inertial, velocity_inertial))
    if with_bias:
        result = add_bias(result, bias)
    if with_noise:
        result = (
            add_gaussian_noise(result, noise_cholesky)
            if noise_draw is None
            else result + noise_cholesky @ noise_draw
        )
    return result


@njit(cache=True)
def star_tracker_measurement(
    attitude, orientation, misalignment, noise_cholesky, with_noise, noise_draw=None
):
    result = _normalize_quaternion(
        _multiply(misalignment, _multiply(orientation, attitude))
    )
    if with_noise:
        noise_quaternion = small_rotation_noise_quaternion_from_cholesky(
            noise_cholesky, noise_draw
        )
        result = _normalize_quaternion(_multiply(noise_quaternion, result))
    return result


@njit(cache=True)
def small_rotation_noise_quaternion_from_cholesky(noise_cholesky, noise_draw=None):
    eta = noise_cholesky @ (
        standard_normals(np.uint64(0), 0, 0, 3) if noise_draw is None else noise_draw
    )
    q = np.empty(4)
    q[0] = 1.0
    q[1:4] = 0.5 * eta
    return _normalize_quaternion(q)


@njit(cache=True)
def star_tracker_residual(measured_attitude, estimated_attitude):
    relative = _normalize_quaternion(
        _multiply(measured_attitude, _conjugate(estimated_attitude))
    )
    if relative[0] < 0.0:
        relative = -relative
    return 2.0 * relative[1:4]


@njit(cache=True)
def star_tracker_mrp_jacobian(delta_mrp):
    square = np.dot(delta_mrp, delta_mrp)
    return (
        4.0
        / (1.0 + square) ** 2
        * ((1.0 + square) * np.eye(3) - 2.0 * np.outer(delta_mrp, delta_mrp))
    )


@njit(cache=True)
def star_tracker_information(noise_covariance, orientation, misalignment):
    mounting = mounting_dcm(orientation, misalignment)
    return mounting.T @ np.linalg.solve(noise_covariance, mounting)


@njit(cache=True)
def magnetometer_information(
    field_inertial,
    attitude,
    orientation,
    misalignment,
    noise_covariance,
    field_covariance,
):
    mounting = mounting_dcm(orientation, misalignment)
    q = attitude / np.sqrt(np.dot(attitude, attitude))
    body_rotation = _quaternion_to_dcm(q)
    field = mounting @ (body_rotation @ field_inertial)
    covariance = (
        noise_covariance
        + mounting @ body_rotation @ field_covariance @ body_rotation.T @ mounting.T
    )
    cross = skew(field)
    return mounting.T @ cross.T @ np.linalg.solve(covariance, cross) @ mounting


@njit(cache=True)
def coarse_sun_available(
    sun_inertial,
    attitude,
    orientation,
    misalignment,
    normal,
    fov_half_angle,
    enabled,
    sun_visible,
):
    if not enabled or not sun_visible:
        return False
    sun_body = sun_in_body(sun_inertial, attitude)
    normal_body = mounting_dcm(orientation, misalignment).T @ normal
    return np.dot(normal_body, sun_body) >= np.cos(fov_half_angle)


@njit(cache=True)
def coarse_sun_information(
    sun_inertial,
    attitude,
    orientation,
    misalignment,
    normal,
    sensitivity,
    noise_variance,
):
    sun_body = sun_in_body(sun_inertial, attitude)
    normal_body = mounting_dcm(orientation, misalignment).T @ normal
    direction = np.cross(normal_body, sun_body)
    return (sensitivity**2 / noise_variance) * np.outer(direction, direction)


@njit(cache=True)
def fine_direction_available(
    direction_inertial,
    attitude,
    orientation,
    misalignment,
    fov_half_angles,
    enabled,
    visible,
):
    if not enabled or not visible:
        return False
    angles = fine_sun_angles(direction_inertial, attitude, orientation, misalignment)
    return abs(angles[0]) <= fov_half_angles[0] and abs(angles[1]) <= fov_half_angles[1]
