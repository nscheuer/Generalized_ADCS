"""Six-error-state unscented Kalman filter. All hot-loop operations are Numba nopython kernels."""

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
    quaternion_mean,
    quaternion_to_rotation_vector,
    rotation_vector_to_quaternion,
)
from GenADCS.backends.python.helpers.math.vector import skew

from .base import AttitudeEstimator
from .direction_updates import measurement as direction_measurement

_CHART_LIMIT_ANGLE = 2.0 * np.arcsin(0.95)


class AttitudeSequentialUKF(AttitudeEstimator):
    """Right-state / left-error unscented Kalman attitude estimator.

    Uses a 6-dimensional tangent error state ``[delta_theta, delta_omega]``
    with 13 sigma points, continuous white angular-acceleration process noise
    ``acceleration_psd`` (rad^2/s^3), and unscented updates across all attitude
    sensors (gyro, magnetometer, star tracker, coarse/fine sun, fine earth).
    """

    def __init__(
        self,
        quaternion: np.ndarray,
        rate: np.ndarray,
        covariance: np.ndarray,
        acceleration_psd: float | np.ndarray,
        *,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
        record: tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            quaternion,
            rate,
            covariance,
            acceleration_psd,
            record=record,
        )
        alpha_f = float(alpha)
        beta_f = float(beta)
        kappa_f = float(kappa)
        if not np.isfinite(alpha_f) or alpha_f <= 0.0:
            raise ValueError("alpha must be finite and positive")
        if not np.isfinite(beta_f):
            raise ValueError("beta must be finite")
        if not np.isfinite(kappa_f):
            raise ValueError("kappa must be finite")
        if alpha_f * alpha_f * (self.COVARIANCE_SIZE + kappa_f) <= 0.0:
            raise ValueError("alpha and kappa must give a positive UKF scale")
        self.alpha = alpha_f
        self.beta = beta_f
        self.kappa = kappa_f
        self.ukf_params = np.array([alpha_f, beta_f, kappa_f], dtype=np.float64)


@njit(cache=True)
def _ukf_weights(
    alpha: float, beta: float, kappa: float, dimension: int = 6
) -> tuple[float, np.ndarray, np.ndarray]:
    scale = alpha * alpha * (dimension + kappa)
    lam = scale - dimension
    gamma = np.sqrt(scale)
    count = 2 * dimension + 1
    w_m = np.full(count, 0.5 / scale)
    w_c = np.full(count, 0.5 / scale)
    w_m[0] = lam / scale
    w_c[0] = lam / scale + (1.0 - alpha * alpha + beta)
    return gamma, w_m, w_c


@njit(cache=True)
def _cholesky_psd(p: np.ndarray) -> np.ndarray:
    p_sym = 0.5 * (p + p.T)
    n = p_sym.shape[0]
    vals, vecs = np.linalg.eigh(p_sym)
    if vals[0] > 1.0e-15:
        return np.linalg.cholesky(p_sym)
    for i in range(n):
        vals[i] = max(vals[i], 0.0)
    reconstructed = (vecs * vals) @ vecs.T + 1.0e-14 * np.eye(n)
    return np.linalg.cholesky(0.5 * (reconstructed + reconstructed.T))


@njit(cache=True)
def _sigma_points(
    q: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = 6
    gamma, w_m, w_c = _ukf_weights(alpha, beta, kappa, n)
    chol = _cholesky_psd(p)
    q_pts = np.empty((2 * n + 1, 4))
    w_pts = np.empty((2 * n + 1, 3))
    q_pts[0] = normalize_quaternion(q)
    w_pts[0] = w
    for i in range(n):
        offset = gamma * chol[:, i]
        d_theta = offset[:3]
        theta_norm = np.sqrt(np.dot(d_theta, d_theta))
        if theta_norm > _CHART_LIMIT_ANGLE:
            d_theta = d_theta * (_CHART_LIMIT_ANGLE / theta_norm)
        dq = rotation_vector_to_quaternion(d_theta)
        q_pts[1 + i] = normalize_quaternion(multiply(dq, q_pts[0]))
        w_pts[1 + i] = w + offset[3:]
        q_pts[1 + n + i] = normalize_quaternion(multiply(conjugate(dq), q_pts[0]))
        w_pts[1 + n + i] = w - offset[3:]
    return q_pts, w_pts, w_m, w_c


@njit(cache=True)
def _state_deviations(
    q_pts: np.ndarray, w_pts: np.ndarray, q_ref: np.ndarray, w_ref: np.ndarray
) -> np.ndarray:
    count = q_pts.shape[0]
    deviations = np.empty((count, 6))
    q_ref_conj = conjugate(q_ref)
    for k in range(count):
        deviations[k, :3] = quaternion_to_rotation_vector(
            multiply(q_pts[k], q_ref_conj)
        )
        deviations[k, 3:] = w_pts[k] - w_ref
    return deviations


@njit(cache=True)
def predict(
    q: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    acceleration_psd: np.ndarray,
    dt: float,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate sigma points and add continuous white angular-acceleration noise."""
    if dt < 0.0:
        raise ValueError("dt must be nonnegative")
    q_pts, w_pts, w_m, w_c = _sigma_points(q, w, p, alpha, beta, kappa)
    count = q_pts.shape[0]
    q_prop = np.empty((count, 4))
    w_prop = np.empty((count, 3))
    for k in range(count):
        increment = rotation_vector_to_quaternion(-w_pts[k] * dt)
        q_prop[k] = normalize_quaternion(multiply(increment, q_pts[k]))
        w_prop[k] = w_pts[k]
    q_mean = quaternion_mean(q_prop, w_m)
    w_mean = w_m @ w_prop
    deviations = _state_deviations(q_prop, w_prop, q_mean, w_mean)
    noise = np.zeros((6, 6))
    noise[:3, :3] = acceleration_psd * (dt**3 / 3.0)
    noise[:3, 3:6] = -acceleration_psd * (dt**2 / 2.0)
    noise[3:6, :3] = noise[:3, 3:6]
    noise[3:6, 3:6] = acceleration_psd * dt
    p_pred = (deviations.T * w_c) @ deviations + noise
    return q_mean, w_mean, 0.5 * (p_pred + p_pred.T)


@njit(cache=True)
def _correct_unscented(
    q: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    dx_pts: np.ndarray,
    dz_pts: np.ndarray,
    w_c: np.ndarray,
    r: np.ndarray,
    residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p_zz = (dz_pts.T * w_c) @ dz_pts + r
    p_zz = 0.5 * (p_zz + p_zz.T)
    p_xz = (dx_pts.T * w_c) @ dz_pts
    gain = np.linalg.solve(p_zz, p_xz.T).T
    delta = gain @ residual
    q_next = normalize_quaternion(
        multiply(rotation_vector_to_quaternion(delta[:3]), q)
    )
    w_next = w + delta[3:]
    p_raw = p - gain @ p_zz @ gain.T
    reset = np.eye(6)
    reset[:3, :3] += 0.5 * skew(delta[:3])
    p_next = reset @ p_raw @ reset.T
    return q_next, w_next, 0.5 * (p_next + p_next.T), gain, p_zz, delta


@njit(cache=True)
def _sensor_noise_matrix(
    kind: int, width: int, chol_i: np.ndarray, scalars_i: np.ndarray
) -> np.ndarray:
    r = np.zeros((width, width))
    if kind == 3:
        r[0, 0] = scalars_i[2] ** 2
        return r
    for a in range(width):
        for b in range(width):
            for k in range(width):
                r[a, b] += chol_i[a, k] * chol_i[b, k]
    return r


@njit(cache=True)
def _wrap_angles(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


@njit(cache=True)
def _record_measurement_diagnostics(
    diagnostics: np.ndarray,
    flags: int,
    width: int,
    residual: np.ndarray,
    innovation: np.ndarray,
    post_residual: np.ndarray,
    gain: np.ndarray,
) -> None:
    if flags & 16:
        diagnostics[:3] = np.nan
        diagnostics[:width] = residual
    if flags & 32:
        block = np.full((3, 3), np.nan)
        block[:width, :width] = innovation
        diagnostics[3:12] = block.ravel()
    if flags & 64:
        diagnostics[12] = np.dot(residual, np.linalg.solve(innovation, residual))
    if flags & 256:
        diagnostics[14:17] = np.nan
        diagnostics[14 : 14 + width] = post_residual
    if flags & 512:
        block = np.full((6, 3), np.nan)
        block[:, :width] = gain
        diagnostics[17:35] = block.ravel()


@njit(cache=True)
def step(
    q: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    acceleration_psd: np.ndarray,
    dt: float,
    readings: np.ndarray,
    sensor_config: tuple[np.ndarray, ...],
    flags: int = 0,
    truth_q: np.ndarray | None = None,
    truth_w: np.ndarray | None = None,
    diagnostics: np.ndarray | None = None,
    references: np.ndarray | None = None,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict and consume all valid attitude sensor readings at one sensor tick."""
    q, w, p = predict(q, w, p, acceleration_psd, dt, alpha, beta, kappa)
    (
        code,
        orientation,
        misalignment,
        bias,
        chol,
        _normal,
        _fov,
        scalars,
        enabled,
        offsets,
        _counts,
        _starts,
    ) = sensor_config

    for i in range(code.size):
        kind = code[i]
        if not enabled[i] or kind not in (0, 1, 2, 3, 4, 5):
            continue
        obs_width = 4 if kind == 2 else (3 if kind in (0, 1) else (1 if kind == 3 else 2))
        observed = readings[offsets[i] : offsets[i] + obs_width]
        if not np.all(np.isfinite(observed)):
            continue

        q_pts, w_pts, w_m, w_c = _sigma_points(q, w, p, alpha, beta, kappa)
        count = q_pts.shape[0]
        dx_pts = _state_deviations(q_pts, w_pts, q, w)
        meas_width = 3 if kind in (0, 1, 2) else obs_width
        r = _sensor_noise_matrix(kind, meas_width, chol[i], scalars[i])

        if kind == 0:
            mount_q = normalize_quaternion(multiply(misalignment[i], orientation[i]))
            mount = quaternion_to_dcm(mount_q)
            z_pts = np.empty((count, 3))
            for k in range(count):
                z_pts[k] = mount @ w_pts[k] + bias[i, :3]
            z_mean = w_m @ z_pts
            dz_pts = z_pts - z_mean
            residual = observed - z_mean
            q_next, w_next, p_next, gain, innovation, _ = _correct_unscented(
                q, w, p, dx_pts, dz_pts, w_c, r, residual
            )
            if diagnostics is not None:
                post_res = observed - (mount @ w_next + bias[i, :3])
                _record_measurement_diagnostics(
                    diagnostics, flags, 3, residual, innovation, post_res, gain
                )
            q, w, p = q_next, w_next, p_next
        elif kind == 2:
            mount_q = normalize_quaternion(multiply(misalignment[i], orientation[i]))
            z_pts = np.empty((count, 4))
            for k in range(count):
                z_pts[k] = normalize_quaternion(multiply(mount_q, q_pts[k]))
            z_mean = quaternion_mean(z_pts, w_m)
            z_mean_conj = conjugate(z_mean)
            dz_pts = np.empty((count, 3))
            for k in range(count):
                dz_pts[k] = quaternion_to_rotation_vector(
                    multiply(z_pts[k], z_mean_conj)
                )
            residual = quaternion_to_rotation_vector(
                multiply(normalize_quaternion(observed), z_mean_conj)
            )
            q_next, w_next, p_next, gain, innovation, _ = _correct_unscented(
                q, w, p, dx_pts, dz_pts, w_c, r, residual
            )
            if diagnostics is not None:
                post_pred = normalize_quaternion(multiply(mount_q, q_next))
                post_res = quaternion_to_rotation_vector(
                    multiply(normalize_quaternion(observed), conjugate(post_pred))
                )
                _record_measurement_diagnostics(
                    diagnostics, flags, 3, residual, innovation, post_res, gain
                )
            q, w, p = q_next, w_next, p_next
        else:
            if references is None:
                raise ValueError(
                    "magnetic and optical corrections require inertial references"
                )
            z_pts = np.empty((count, meas_width))
            all_finite = True
            for k in range(count):
                z_k = direction_measurement(q_pts[k], i, sensor_config, references)
                if not np.all(np.isfinite(z_k)):
                    all_finite = False
                    break
                z_pts[k] = z_k
            if not all_finite:
                continue
            if kind == 4 or kind == 5:
                wrapped_deltas = _wrap_angles(z_pts - z_pts[0])
                z_mean = _wrap_angles(z_pts[0] + w_m @ wrapped_deltas)
                dz_pts = _wrap_angles(z_pts - z_mean)
                residual = _wrap_angles(observed - z_mean)
            else:
                z_mean = w_m @ z_pts
                dz_pts = z_pts - z_mean
                residual = observed - z_mean
            q_next, w_next, p_next, gain, innovation, _ = _correct_unscented(
                q, w, p, dx_pts, dz_pts, w_c, r, residual
            )
            if diagnostics is not None:
                post_pred = direction_measurement(q_next, i, sensor_config, references)
                if np.all(np.isfinite(post_pred)):
                    post_res = observed - post_pred
                    if kind == 4 or kind == 5:
                        post_res = _wrap_angles(post_res)
                else:
                    post_res = np.full(meas_width, np.nan)
                _record_measurement_diagnostics(
                    diagnostics, flags, meas_width, residual, innovation, post_res, gain
                )
            q, w, p = q_next, w_next, p_next

    if diagnostics is not None:
        if flags & 4:
            diagnostics[35:44] = p[:3, :3].ravel()
        if flags & 8:
            diagnostics[44:53] = p[3:, 3:].ravel()
        if flags & 128 and truth_q is not None and truth_w is not None:
            error = np.empty(6)
            error[:3] = quaternion_to_rotation_vector(multiply(truth_q, conjugate(q)))
            error[3:] = truth_w - w
            diagnostics[13] = np.dot(error, np.linalg.solve(p, error))

    return q, w, p
