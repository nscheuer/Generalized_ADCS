"""Seven-state quaternion EKF. Numerical work stays in Numba kernels."""

import numpy as np
from numba import njit

from GenADCS.backends.python.helpers.math.rotations.quaternion.algebra import (
    conjugate, multiply, normalize_quaternion,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import quaternion_to_dcm
from GenADCS.backends.python.helpers.math.rotations.quaternion.interpolation import (
    quaternion_to_rotation_vector, rotation_vector_to_quaternion,
)
from .base import AttitudeEstimator
from .direction_updates import update as update_directions


class AttitudeEKF(AttitudeEstimator):
    """Full quaternion state ``[q (scalar first), body rate]`` with 7x7 covariance."""

    COVARIANCE_SIZE = 7


@njit(cache=True)
def _propagate(q, w, dt):
    return normalize_quaternion(multiply(rotation_vector_to_quaternion(-w * dt), q))


@njit(cache=True)
def predict(q, w, p, acceleration_psd, dt):
    if dt < 0.0:
        raise ValueError("dt must be nonnegative")
    x = np.empty(7)
    x[:4], x[4:] = q, w
    xp = np.empty(7)
    xp[:4], xp[4:] = _propagate(q, w, dt), w
    # Differentiate the discrete quaternion/rate transition in the full state.
    f = np.empty((7, 7))
    eps = 1e-7
    for j in range(7):
        plus, minus = x.copy(), x.copy()
        plus[j] += eps
        minus[j] -= eps
        qp = _propagate(plus[:4], plus[4:], dt)
        qm = _propagate(minus[:4], minus[4:], dt)
        if np.dot(qp, xp[:4]) < 0.0:
            qp = -qp
        if np.dot(qm, xp[:4]) < 0.0:
            qm = -qm
        yp, ym = np.empty(7), np.empty(7)
        yp[:4], yp[4:] = qp, plus[4:]
        ym[:4], ym[4:] = qm, minus[4:]
        f[:, j] = (yp - ym) / (2.0 * eps)

    # Continuous white angular acceleration, integrated in attitude/rate error
    # coordinates then lifted to the four-component quaternion tangent space.
    q6 = np.zeros((6, 6))
    q6[:3, :3] = acceleration_psd * (dt**3 / 3.0)
    q6[:3, 3:] = -acceleration_psd * (dt**2 / 2.0)
    q6[3:, :3] = q6[:3, 3:]
    q6[3:, 3:] = acceleration_psd * dt
    lift = np.zeros((7, 6))
    lift[0, :3] = -0.5 * xp[:3]
    lift[1:4, :3] = 0.5 * (xp[0] * np.eye(3) - _skew(xp[1:4]))
    lift[4:, 3:] = np.eye(3)
    result = f @ p @ f.T + lift @ q6 @ lift.T
    result = _tangent_covariance(xp[:4], result)
    return xp[:4], xp[4:], 0.5 * (result + result.T)


@njit(cache=True)
def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


@njit(cache=True)
def _tangent_covariance(q, p):
    projection = np.eye(7)
    projection[:4, :4] -= np.outer(q, q)
    result = projection @ p @ projection.T
    return 0.5 * (result + result.T)


@njit(cache=True)
def correct(q, w, p, residual, h, covariance):
    s = h @ p @ h.T + covariance
    k = np.linalg.solve(s, h @ p).T
    delta = k @ residual
    qn = normalize_quaternion(q + delta[:4])
    wn = w + delta[4:]
    a = np.eye(7) - k @ h
    pn = a @ p @ a.T + k @ covariance @ k.T
    pn = _tangent_covariance(qn, pn)
    return qn, wn, 0.5 * (pn + pn.T)


@njit(cache=True)
def _tracker_residual(q, measured, mount_q):
    predicted = multiply(mount_q, q)
    observation = measured.copy()
    if np.dot(observation, predicted) < 0.0:
        observation = -observation
    return quaternion_to_rotation_vector(multiply(observation, conjugate(predicted)))


@njit(cache=True)
def _state_error_jacobian(q):
    jacobian = np.zeros((6, 7))
    jacobian[:3, 0] = -2.0 * q[1:]
    jacobian[:3, 1:4] = 2.0 * (q[0] * np.eye(3) + _skew(q[1:]))
    jacobian[3:, 4:] = np.eye(3)
    return jacobian


@njit(cache=True)
def step(q, w, p, acceleration_psd, dt, readings, sensor_config,
         flags=0, truth_q=None, truth_w=None, diagnostics=None, references=None):
    q, w, p = predict(q, w, p, acceleration_psd, dt)
    q, w, p = update_directions(q, w, p, readings, sensor_config, references, True, flags, diagnostics)
    (code, orientation, misalignment, bias, chol, _normal, _fov, _scalars,
     enabled, offsets, _counts, _starts) = sensor_config
    for i in range(code.size):
        if not enabled[i] or (code[i] != 0 and code[i] != 2):
            continue
        kind = code[i]
        value = readings[offsets[i]:offsets[i] + (3 if kind == 0 else 4)]
        if not np.all(np.isfinite(value)):
            continue
        mount_q = normalize_quaternion(multiply(misalignment[i], orientation[i]))
        mount = quaternion_to_dcm(mount_q)
        r = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                for k in range(3):
                    r[a, b] += chol[i, a, k] * chol[i, b, k]
        h = np.zeros((3, 7))
        if kind == 0:
            h[:, 4:] = mount
            residual = value - bias[i, :3] - mount @ w
        else:
            residual = _tracker_residual(q, value, mount_q)
            # Differentiate the exact rotation-vector observation residual.
            eps = 1e-7
            for j in range(4):
                qp, qm = q.copy(), q.copy()
                qp[j] += eps
                qm[j] -= eps
                rp = _tracker_residual(normalize_quaternion(qp), value, mount_q)
                rm = _tracker_residual(normalize_quaternion(qm), value, mount_q)
                h[:, j] = -(rp - rm) / (2.0 * eps)
        innovation = h @ p @ h.T + r
        gain = np.linalg.solve(innovation, h @ p).T
        delta = gain @ residual
        if diagnostics is not None:
            tangent_gain = _state_error_jacobian(q) @ gain
            if flags & 16:
                diagnostics[0:3] = residual
            if flags & 32:
                diagnostics[3:12] = innovation.ravel()
            if flags & 64:
                diagnostics[12] = np.dot(residual, np.linalg.solve(innovation, residual))
            if flags & 256:
                diagnostics[14:17] = residual - h @ delta
            if flags & 512:
                diagnostics[17:35] = tangent_gain.ravel()
        q, w, p = correct(q, w, p, residual, h, r)
    if diagnostics is not None:
        if flags & 4:
            jacobian = _state_error_jacobian(q)[:3]
            diagnostics[35:44] = (jacobian @ p @ jacobian.T).ravel()
        if flags & 8:
            diagnostics[44:53] = p[4:, 4:].ravel()
        if flags & 128 and truth_q is not None and truth_w is not None:
            error = np.empty(6)
            error[:3] = quaternion_to_rotation_vector(multiply(truth_q, conjugate(q)))
            error[3:] = truth_w - w
            jacobian = _state_error_jacobian(q)
            covariance6 = jacobian @ p @ jacobian.T
            diagnostics[13] = np.dot(error, np.linalg.solve(covariance6, error))
    return q, w, p
