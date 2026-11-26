"""Six-error-state square-root unscented Kalman filter. All hot-loop operations are Numba nopython kernels."""

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

from .attitude_sequential_ukf import (
    _CHART_LIMIT_ANGLE,
    _cholesky_psd,
    _record_measurement_diagnostics,
    _state_deviations,
    _ukf_weights,
    _wrap_angles,
)
from .base import AttitudeEstimator
from .direction_updates import measurement as direction_measurement


class AttitudeSequentialSRUKF(AttitudeEstimator):
    """Right-state / left-error square-root unscented Kalman attitude estimator.

    Uses a 6-dimensional tangent error state ``[delta_theta, delta_omega]``
    with 13 sigma points, propagating and updating the lower-triangular
    Cholesky factor ``S`` (``P = S @ S.T``) via QR decompositions and rank-1
    Cholesky updates/downdates across all attitude sensors.
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
def _cholupdate(s_mat: np.ndarray, vec: np.ndarray, sign: float) -> np.ndarray:
    """Rank-1 Cholesky update (sign=+1) or downdate (sign=-1) on lower-triangular S."""
    n = s_mat.shape[0]
    out = s_mat.copy()
    x = vec.copy()
    for k in range(n):
        diag = out[k, k]
        if abs(diag) < 1.0e-15:
            p_fallback = s_mat @ s_mat.T + sign * np.outer(vec, vec)
            return _cholesky_psd(p_fallback)
        r2 = diag * diag + sign * x[k] * x[k]
        if r2 <= 1.0e-30:
            p_fallback = s_mat @ s_mat.T + sign * np.outer(vec, vec)
            return _cholesky_psd(p_fallback)
        r = np.sqrt(r2)
        c = r / diag
        s = x[k] / diag
        out[k, k] = r
        if k + 1 < n:
            for j in range(k + 1, n):
                out[j, k] = (out[j, k] + sign * s * x[j]) / c
                x[j] = c * x[j] - s * out[j, k]
    return out


@njit(cache=True)
def _sigma_points_from_sqrt(
    q: np.ndarray,
    w: np.ndarray,
    s_mat: np.ndarray,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = 6
    gamma, w_m, w_c = _ukf_weights(alpha, beta, kappa, n)
    q_pts = np.empty((2 * n + 1, 4))
    w_pts = np.empty((2 * n + 1, 3))
    q_pts[0] = normalize_quaternion(q)
    w_pts[0] = w
    for i in range(n):
        offset = gamma * s_mat[:, i]
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
def _predict_sqrt(
    q: np.ndarray,
    w: np.ndarray,
    s_mat: np.ndarray,
    acceleration_psd: np.ndarray,
    dt: float,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dt < 0.0:
        raise ValueError("dt must be nonnegative")
    q_pts, w_pts, w_m, w_c = _sigma_points_from_sqrt(q, w, s_mat, alpha, beta, kappa)
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

    # Build compound matrix A = [sqrt(w_c[1:]) * deviations[1:]; S_Q.T]
    has_noise = dt > 0.0 and np.max(np.abs(noise)) > 0.0
    rows = (count - 1) + (6 if has_noise else 0)
    compound = np.empty((rows, 6))
    for k in range(1, count):
        compound[k - 1] = np.sqrt(w_c[k]) * deviations[k]
    if has_noise:
        s_q = _cholesky_psd(noise)
        compound[count - 1 :] = s_q.T
    _, r_qr = np.linalg.qr(compound)
    s_pred = r_qr.T
    # Enforce positive diagonal convention on lower-triangular S_pred
    for i in range(6):
        if s_pred[i, i] < 0.0:
            s_pred[:, i] = -s_pred[:, i]

    sign0 = 1.0 if w_c[0] >= 0.0 else -1.0
    s_pred = _cholupdate(s_pred, np.sqrt(abs(w_c[0])) * deviations[0], sign0)
    return q_mean, w_mean, s_pred


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
    """Propagate sigma points in square-root form and return covariance P = S @ S.T."""
    s_in = _cholesky_psd(p)
    q_mean, w_mean, s_pred = _predict_sqrt(
        q, w, s_in, acceleration_psd, dt, alpha, beta, kappa
    )
    p_pred = s_pred @ s_pred.T
    return q_mean, w_mean, 0.5 * (p_pred + p_pred.T)


@njit(cache=True)
def _sensor_noise_sqrt(
    kind: int, width: int, chol_i: np.ndarray, scalars_i: np.ndarray
) -> np.ndarray:
    if kind == 3:
        s_r = np.zeros((1, 1))
        s_r[0, 0] = abs(scalars_i[2])
        return s_r
    return chol_i[:width, :width].copy()


@njit(cache=True)
def _correct_srukf(
    q: np.ndarray,
    w: np.ndarray,
    s_mat: np.ndarray,
    dx_pts: np.ndarray,
    dz_pts: np.ndarray,
    w_c: np.ndarray,
    s_r: np.ndarray,
    residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = dz_pts.shape[0]
    m = dz_pts.shape[1]
    compound = np.empty((count - 1 + m, m))
    for k in range(1, count):
        compound[k - 1] = np.sqrt(w_c[k]) * dz_pts[k]
    compound[count - 1 :] = s_r.T
    _, r_z = np.linalg.qr(compound)
    s_z = r_z.T
    for i in range(m):
        if s_z[i, i] < 0.0:
            s_z[:, i] = -s_z[:, i]

    sign0 = 1.0 if w_c[0] >= 0.0 else -1.0
    s_z = _cholupdate(s_z, np.sqrt(abs(w_c[0])) * dz_pts[0], sign0)

    p_xz = (dx_pts.T * w_c) @ dz_pts
    # Solve S_z @ Y = P_xz.T, then S_z.T @ K.T = Y
    y_solve = np.linalg.solve(s_z, p_xz.T)
    gain = np.linalg.solve(s_z.T, y_solve).T
    delta = gain @ residual

    q_next = normalize_quaternion(
        multiply(rotation_vector_to_quaternion(delta[:3]), q)
    )
    w_next = w + delta[3:]

    u_mat = gain @ s_z
    s_raw = s_mat.copy()
    for j in range(m):
        s_raw = _cholupdate(s_raw, u_mat[:, j], -1.0)

    reset = np.eye(6)
    reset[:3, :3] += 0.5 * skew(delta[:3])
    _, r_post = np.linalg.qr((reset @ s_raw).T)
    s_next = r_post.T
    for i in range(6):
        if s_next[i, i] < 0.0:
            s_next[:, i] = -s_next[:, i]

    innovation = s_z @ s_z.T
    return q_next, w_next, s_next, gain, 0.5 * (innovation + innovation.T)


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
    """Predict and consume all valid attitude sensor readings in square-root form."""
    s_mat = _cholesky_psd(p)
    q, w, s_mat = _predict_sqrt(
        q, w, s_mat, acceleration_psd, dt, alpha, beta, kappa
    )
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

        q_pts, w_pts, w_m, w_c = _sigma_points_from_sqrt(
            q, w, s_mat, alpha, beta, kappa
        )
        count = q_pts.shape[0]
        dx_pts = _state_deviations(q_pts, w_pts, q, w)
        meas_width = 3 if kind in (0, 1, 2) else obs_width
        s_r = _sensor_noise_sqrt(kind, meas_width, chol[i], scalars[i])

        if kind == 0:
            mount_q = normalize_quaternion(multiply(misalignment[i], orientation[i]))
            mount = quaternion_to_dcm(mount_q)
            z_pts = np.empty((count, 3))
            for k in range(count):
                z_pts[k] = mount @ w_pts[k] + bias[i, :3]
            z_mean = w_m @ z_pts
            dz_pts = z_pts - z_mean
            residual = observed - z_mean
            q_next, w_next, s_next, gain, innovation = _correct_srukf(
                q, w, s_mat, dx_pts, dz_pts, w_c, s_r, residual
            )
            if diagnostics is not None:
                post_res = observed - (mount @ w_next + bias[i, :3])
                _record_measurement_diagnostics(
                    diagnostics, flags, 3, residual, innovation, post_res, gain
                )
            q, w, s_mat = q_next, w_next, s_next
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
            q_next, w_next, s_next, gain, innovation = _correct_srukf(
                q, w, s_mat, dx_pts, dz_pts, w_c, s_r, residual
            )
            if diagnostics is not None:
                post_pred = normalize_quaternion(multiply(mount_q, q_next))
                post_res = quaternion_to_rotation_vector(
                    multiply(normalize_quaternion(observed), conjugate(post_pred))
                )
                _record_measurement_diagnostics(
                    diagnostics, flags, 3, residual, innovation, post_res, gain
                )
            q, w, s_mat = q_next, w_next, s_next
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
            q_next, w_next, s_next, gain, innovation = _correct_srukf(
                q, w, s_mat, dx_pts, dz_pts, w_c, s_r, residual
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
            q, w, s_mat = q_next, w_next, s_next

    p_out = s_mat @ s_mat.T
    p_out = 0.5 * (p_out + p_out.T)
    if diagnostics is not None:
        if flags & 4:
            diagnostics[35:44] = p_out[:3, :3].ravel()
        if flags & 8:
            diagnostics[44:53] = p_out[3:, 3:].ravel()
        if flags & 128 and truth_q is not None and truth_w is not None:
            error = np.empty(6)
            error[:3] = quaternion_to_rotation_vector(multiply(truth_q, conjugate(q)))
            error[3:] = truth_w - w
            diagnostics[13] = np.dot(error, np.linalg.solve(p_out, error))

    return q, w, p_out
