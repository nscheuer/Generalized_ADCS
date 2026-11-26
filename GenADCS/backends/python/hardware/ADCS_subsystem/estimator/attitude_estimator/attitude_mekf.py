"""Six-error-state multiplicative EKF. All operations are Numba nopython kernels."""

import numpy as np
from numba import njit

from GenADCS.backends.python.helpers.math.rotations.quaternion.algebra import (
    conjugate,
    multiply,
    normalize_quaternion,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import (
    quaternion_to_dcm,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.interpolation import (
    quaternion_to_rotation_vector,
    rotation_vector_to_quaternion,
)
from GenADCS.backends.python.helpers.math.rotations.sensor_kernels import (
    so3_left_jacobian,
)
from GenADCS.backends.python.helpers.math.vector import skew

from .base import AttitudeEstimator
from .direction_updates import update as update_directions


class AttitudeMEKF(AttitudeEstimator):
    """Constant-rate MEKF with gyro, magnetic, Sun, Earth and tracker corrections.

    ``acceleration_psd`` is continuous white angular-acceleration spectral
    density in rad²/s³. Tracker noise is a rotation-vector covariance in its
    sensor frame; gyro noise is a rate covariance in its sensor frame.
    """


@njit(cache=True)
def predict(
    q: np.ndarray, w: np.ndarray, p: np.ndarray, acceleration_psd: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate the nominal state and its left/body attitude error."""
    if dt < 0.0:
        raise ValueError("dt must be nonnegative")
    rotation = -w * dt
    increment = rotation_vector_to_quaternion(rotation)
    q_next = normalize_quaternion(multiply(increment, q))
    phi = np.eye(6)
    phi[:3, :3] = quaternion_to_dcm(increment)
    phi[:3, 3:6] = -so3_left_jacobian(rotation) * dt
    noise = np.zeros((6, 6))
    noise[:3, :3] = acceleration_psd * (dt**3 / 3.0)
    noise[:3, 3:6] = -acceleration_psd * (dt**2 / 2.0)
    noise[3:6, :3] = noise[:3, 3:6]
    noise[3:6, 3:6] = acceleration_psd * dt
    result = phi @ p @ phi.T + noise
    return q_next, w.copy(), 0.5 * (result + result.T)


@njit(cache=True)
def correct(
    q: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    residual: np.ndarray,
    h: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared Joseph-form correction and multiplicative covariance reset."""
    innovation = h @ p @ h.T + covariance
    gain = np.linalg.solve(innovation, h @ p).T
    delta = gain @ residual
    q_next = normalize_quaternion(
        multiply(rotation_vector_to_quaternion(delta[:3]), q)
    )
    w_next = w + delta[3:6]
    identity = np.eye(6)
    remainder = identity - gain @ h
    updated = remainder @ p @ remainder.T + gain @ covariance @ gain.T
    reset = np.eye(6)
    reset[:3, :3] += 0.5 * skew(delta[:3])
    updated = reset @ updated @ reset.T
    return q_next, w_next, 0.5 * (updated + updated.T)


@njit(cache=True)
def correct_gyro(
    q: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    measured_rate: np.ndarray,
    mounting: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update the rate from a body-to-sensor mounted gyroscope."""
    h = np.zeros((3, 6))
    h[:, 3:6] = mounting
    return correct(q, w, p, measured_rate - mounting @ w, h, covariance)


@njit(cache=True)
def correct_star_tracker(
    q: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    measured_quaternion: np.ndarray,
    mounting_quaternion: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update attitude from a scalar-first body-to-sensor mounted tracker."""
    predicted = multiply(mounting_quaternion, q)
    relative = multiply(measured_quaternion, conjugate(predicted))
    residual = quaternion_to_rotation_vector(relative)
    h = np.zeros((3, 6))
    h[:, :3] = quaternion_to_dcm(mounting_quaternion)
    return correct(q, w, p, residual, h, covariance)


@njit(cache=True)
def step(
    q: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    acceleration_psd: np.ndarray,
    dt: float,
    readings: np.ndarray,
    sensor_config: tuple[np.ndarray, ...],
    references=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict and consume valid attitude sensor readings at one sensor tick.

    References contain inertial magnetic, Sun and Earth vectors (three rows).
    Accelerometer and GNSS readings are excluded.
    NaN readings, including occulted trackers, are skipped.
    """
    q, w, p = predict(q, w, p, acceleration_psd, dt)
    q, w, p = update_directions(q, w, p, readings, sensor_config, references, False)
    (
        code,
        orientation,
        misalignment,
        bias,
        chol,
        _normal,
        _fov,
        _scalars,
        enabled,
        offsets,
        _counts,
        _starts,
    ) = sensor_config
    for i in range(code.size):
        if not enabled[i]:
            continue
        kind = code[i]
        if kind != 0 and kind != 2:
            continue
        offset = offsets[i]
        width = 3 if kind == 0 else 4
        value = readings[offset : offset + width]
        if not np.all(np.isfinite(value)):
            continue
        mounting_q = normalize_quaternion(multiply(misalignment[i], orientation[i]))
        noise = np.empty((3, 3))
        for row in range(3):
            for column in range(3):
                noise[row, column] = 0.0
                for k in range(3):
                    noise[row, column] += chol[i, row, k] * chol[i, column, k]
        if kind == 0:
            q, w, p = correct_gyro(
                q, w, p, value - bias[i, :3], quaternion_to_dcm(mounting_q), noise
            )
        else:
            q, w, p = correct_star_tracker(q, w, p, value, mounting_q, noise)
    return q, w, p


@njit(cache=True)
def step_with_diagnostics(
    q: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    acceleration_psd: np.ndarray,
    dt: float,
    readings: np.ndarray,
    sensor_config: tuple[np.ndarray, ...],
    flags: int,
    truth_q: np.ndarray,
    truth_w: np.ndarray,
    diagnostics: np.ndarray,
    references=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compiled update; diagnostics describe the last valid measurement at a tick.

    A tick with no valid measurement leaves measurement fields as NaN.
    """
    q, w, p = predict(q, w, p, acceleration_psd, dt)
    q, w, p = update_directions(q, w, p, readings, sensor_config, references, False, flags, diagnostics)
    (
        code,
        orientation,
        misalignment,
        bias,
        chol,
        _normal,
        _fov,
        _scalars,
        enabled,
        offsets,
        _counts,
        _starts,
    ) = sensor_config
    for i in range(code.size):
        kind = code[i]
        if not enabled[i] or (kind != 0 and kind != 2):
            continue
        value = readings[offsets[i] : offsets[i] + (3 if kind == 0 else 4)]
        if not np.all(np.isfinite(value)):
            continue
        mount_q = normalize_quaternion(multiply(misalignment[i], orientation[i]))
        mount = quaternion_to_dcm(mount_q)
        r = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                for k in range(3):
                    r[a, b] += chol[i, a, k] * chol[i, b, k]
        h = np.zeros((3, 6))
        if kind == 0:
            h[:, 3:] = mount
            residual = value - bias[i, :3] - mount @ w
        else:
            h[:, :3] = mount
            predicted = multiply(mount_q, q)
            residual = quaternion_to_rotation_vector(
                multiply(value, conjugate(predicted))
            )
        s = h @ p @ h.T + r
        gain = np.linalg.solve(s, h @ p).T
        delta = gain @ residual
        if flags & 16:
            diagnostics[0:3] = residual
        if flags & 32:
            diagnostics[3:12] = s.ravel()
        if flags & 64:
            diagnostics[12] = np.dot(residual, np.linalg.solve(s, residual))
        if flags & 256:
            diagnostics[14:17] = residual - h @ delta
        if flags & 512:
            diagnostics[17:35] = gain.ravel()
        q, w, p = correct(q, w, p, residual, h, r)
    if flags & 4:
        diagnostics[35:44] = p[:3, :3].ravel()
    if flags & 8:
        diagnostics[44:53] = p[3:, 3:].ravel()
    if flags & 128:
        error_q = multiply(truth_q, conjugate(q))
        error = np.empty(6)
        error[:3] = quaternion_to_rotation_vector(error_q)
        error[3:] = truth_w - w
        diagnostics[13] = np.dot(error, np.linalg.solve(p, error))
    return q, w, p
