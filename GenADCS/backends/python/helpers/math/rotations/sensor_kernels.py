"""Compiled rotation and measurement geometry kernels shared by sensors."""

import numpy as np
from numba import njit

from GenADCS.backends.python.helpers.random.philox import standard_normals

from ..vector import skew
from .mrp.kernels import mrp_to_quaternion
from .quaternion.algebra import _multiply, _normalize_quaternion
from .quaternion.conversions import _quaternion_to_dcm
from .quaternion.interpolation import rotation_vector_to_quaternion


@njit(cache=True)
def mounting_dcm(orientation, misalignment):
    return _quaternion_to_dcm(_multiply(misalignment, orientation))


@njit(cache=True)
def rotate_inertial_to_sensor(vector_inertial, attitude, orientation, misalignment):
    q = attitude / np.sqrt(np.dot(attitude, attitude))
    return mounting_dcm(orientation, misalignment) @ (
        _quaternion_to_dcm(q) @ vector_inertial
    )


@njit(cache=True)
def rotate_body_to_sensor(vector_body, orientation, misalignment):
    return mounting_dcm(orientation, misalignment) @ vector_body


@njit(cache=True)
def attitude_vector_jacobian_quaternion(vector, delta_q):
    q = delta_q / np.sqrt(np.dot(delta_q, delta_q))
    rotated = _quaternion_to_dcm(q) @ vector
    v = q[1:4]
    return -2.0 * skew(rotated) @ (q[0] * np.eye(3) + skew(v) + np.outer(v, v) / q[0])


@njit(cache=True)
def attitude_vector_jacobian_mrp(vector, delta_mrp):
    square = np.dot(delta_mrp, delta_mrp)
    rotated = _quaternion_to_dcm(mrp_to_quaternion(delta_mrp)) @ vector
    matrix = (
        (1.0 - square) * np.eye(3)
        + 2.0 * skew(delta_mrp)
        + 2.0 * np.outer(delta_mrp, delta_mrp)
    )
    return -4.0 / (1.0 + square) ** 2 * skew(rotated) @ matrix


@njit(cache=True)
def so3_left_jacobian(vector):
    angle = np.sqrt(np.dot(vector, vector))
    cross = skew(vector)
    if angle < 1e-5:
        a = 0.5 - angle * angle / 24.0
        b = 1.0 / 6.0 - angle * angle / 120.0
    else:
        a = (1.0 - np.cos(angle)) / angle**2
        b = (angle - np.sin(angle)) / angle**3
    return np.eye(3) + a * cross + b * (cross @ cross)


@njit(cache=True)
def attitude_vector_jacobian_angle(vector, delta_angle):
    rotated = _quaternion_to_dcm(rotation_vector_to_quaternion(delta_angle)) @ vector
    return -skew(rotated) @ so3_left_jacobian(delta_angle)


@njit(cache=True)
def star_tracker_angle_jacobian(delta_angle):
    angle = np.sqrt(np.dot(delta_angle, delta_angle))
    if angle < 1e-7:
        a = 1.0 - angle * angle / 24.0
        b = -1.0 / 12.0 + angle * angle / 480.0
    else:
        a = 2.0 * np.sin(angle / 2.0) / angle
        b = np.cos(angle / 2.0) / angle**2 - 2.0 * np.sin(angle / 2.0) / angle**3
    return a * np.eye(3) + b * np.outer(delta_angle, delta_angle)


@njit(cache=True)
def small_rotation_noise_quaternion(covariance, seed=np.uint64(0), tick=0):
    """Sample a small rotation vector from covariance and return its quaternion."""
    eta = np.linalg.cholesky(covariance) @ standard_normals(seed, tick, 0, 3)
    return _normalize_quaternion(np.concatenate((np.ones(1), 0.5 * eta)))


@njit(cache=True)
def coarse_sun_measurement(
    sun_inertial, attitude, orientation, misalignment, normal, sensitivity
):
    sun_body = sun_in_body(sun_inertial, attitude)
    normal_body = mounting_dcm(orientation, misalignment).T @ normal
    return sensitivity * max(0.0, np.dot(normal_body, sun_body))


@njit(cache=True)
def sun_in_body(sun_inertial, attitude):
    """Rotate an inertial Sun vector into body coordinates."""
    q = attitude / np.sqrt(np.dot(attitude, attitude))
    return _quaternion_to_dcm(q) @ sun_inertial


@njit(cache=True)
def scalar_measurement_angle_jacobian(angle):
    """SO(3) angle-vector Jacobian used for scalar attitude measurements."""
    magnitude = np.sqrt(np.dot(angle, angle))
    cross = skew(angle)
    if magnitude < 1e-5:
        a = 0.5 - magnitude * magnitude / 24.0
        b = 1.0 / 6.0 - magnitude * magnitude / 120.0
    else:
        a = (1.0 - np.cos(magnitude)) / magnitude**2
        b = (magnitude - np.sin(magnitude)) / magnitude**3
    return np.eye(3) + a * cross + b * (cross @ cross)


@njit(cache=True)
def coarse_sun_attitude_jacobian(
    sun_inertial,
    attitude,
    orientation,
    misalignment,
    normal,
    sensitivity,
    parameter,
    kind,
):
    """Scalar sun-angle Jacobian; kind: 0 quaternion, 1 MRP, 2 angle vector."""
    q = attitude / np.sqrt(np.dot(attitude, attitude))
    sun_body = sun_in_body(sun_inertial, q)
    normal_body = mounting_dcm(orientation, misalignment).T @ normal
    if kind == 0:
        qd = parameter / np.sqrt(np.dot(parameter, parameter))
        sun = _quaternion_to_dcm(qd) @ sun_body
        v = qd[1:4]
        right = (
            -2.0 * skew(sun) @ (qd[0] * np.eye(3) + skew(v) + np.outer(v, v) / qd[0])
        )
    elif kind == 1:
        square = np.dot(parameter, parameter)
        sun = _quaternion_to_dcm(mrp_to_quaternion(parameter)) @ sun_body
        matrix = (
            (1.0 - square) * np.eye(3)
            + 2.0 * skew(parameter)
            + 2.0 * np.outer(parameter, parameter)
        )
        right = -4.0 / (1.0 + square) ** 2 * skew(sun) @ matrix
    else:
        sun = _quaternion_to_dcm(rotation_vector_to_quaternion(parameter)) @ sun_body
        right = -skew(sun) @ so3_left_jacobian(parameter)
    return sensitivity * normal_body @ right


@njit(cache=True)
def fine_sun_angles(sun_inertial, attitude, orientation, misalignment):
    s = rotate_inertial_to_sensor(sun_inertial, attitude, orientation, misalignment)
    return np.array((np.arctan2(s[0], s[2]), np.arctan2(s[1], s[2])))


@njit(cache=True)
def fine_sun_direction_jacobian(sensor_vector):
    x, y, z = sensor_vector
    xden = x * x + z * z
    yden = y * y + z * z
    result = np.zeros((2, 3))
    result[0, 0], result[0, 2] = z / xden, -x / xden
    result[1, 1], result[1, 2] = z / yden, -y / yden
    return result


@njit(cache=True)
def fine_sun_attitude_jacobian(
    sun_inertial, attitude, orientation, misalignment, parameter, kind
):
    sun_body = sun_in_body(sun_inertial, attitude)
    mounting = mounting_dcm(orientation, misalignment)
    if kind == 0:
        q = parameter / np.sqrt(np.dot(parameter, parameter))
        perturbed_body = _quaternion_to_dcm(q) @ sun_body
        direction_jac = attitude_vector_jacobian_quaternion(sun_body, parameter)
    elif kind == 1:
        perturbed_body = _quaternion_to_dcm(mrp_to_quaternion(parameter)) @ sun_body
        direction_jac = attitude_vector_jacobian_mrp(sun_body, parameter)
    else:
        perturbed_body = (
            _quaternion_to_dcm(rotation_vector_to_quaternion(parameter)) @ sun_body
        )
        direction_jac = attitude_vector_jacobian_angle(sun_body, parameter)
    return (
        fine_sun_direction_jacobian(mounting @ perturbed_body)
        @ mounting
        @ direction_jac
    )


@njit(cache=True)
def fine_sun_fisher_information(angles, covariance, orientation, misalignment):
    alpha, beta = angles
    u, v = np.tan(alpha), np.tan(beta)
    k = np.array(
        (
            (-u * v / (1.0 + u * u), 1.0, -v / (1.0 + u * u)),
            (-1.0, u * v / (1.0 + v * v), u / (1.0 + v * v)),
        )
    )
    mounting = mounting_dcm(orientation, misalignment)
    return mounting.T @ k.T @ np.linalg.solve(covariance, k) @ mounting
