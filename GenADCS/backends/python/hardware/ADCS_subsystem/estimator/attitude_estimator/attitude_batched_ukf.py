"""Six-error-state batched unscented Kalman filter. All hot-loop operations are Numba nopython kernels."""

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
    _sensor_noise_matrix,
    _sigma_points,
    _state_deviations,
    _wrap_angles,
    predict,
)
from .base import AttitudeEstimator
from .direction_updates import measurement as direction_measurement


class AttitudeBatchedUKF(AttitudeEstimator):
    """Right-state / left-error batched unscented Kalman attitude estimator.

    Uses a 6-dimensional tangent error state ``[delta_theta, delta_omega]``
    with 13 sigma points, stacking all active finite-sigma-point attitude
    sensors into a single order-invariant measurement update per tick.
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
    """Predict and batch-consume all valid attitude sensor readings at one tick."""
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

    q_pts, w_pts, w_m, w_c = _sigma_points(q, w, p, alpha, beta, kappa)
    count = q_pts.shape[0]
    dx_pts = _state_deviations(q_pts, w_pts, q, w)

    max_meas = 3 * code.size
    dz_buf = np.empty((max_meas, count))
    res_buf = np.empty(max_meas)
    r_buf = np.zeros((max_meas, max_meas))
    m_total = 0

    for i in range(code.size):
        kind = code[i]
        if not enabled[i] or kind not in (0, 1, 2, 3, 4, 5):
            continue
        obs_width = 4 if kind == 2 else (3 if kind in (0, 1) else (1 if kind == 3 else 2))
        observed = readings[offsets[i] : offsets[i] + obs_width]
        if not np.all(np.isfinite(observed)):
            continue

        meas_width = 3 if kind in (0, 1, 2) else obs_width
        if kind == 0:
            mount_q = normalize_quaternion(multiply(misalignment[i], orientation[i]))
            mount = quaternion_to_dcm(mount_q)
            z_pts = np.empty((count, 3))
            for k in range(count):
                z_pts[k] = mount @ w_pts[k] + bias[i, :3]
            z_mean = w_m @ z_pts
            dz_buf[m_total : m_total + 3, :] = (z_pts - z_mean).T
            res_buf[m_total : m_total + 3] = observed - z_mean
        elif kind == 2:
            mount_q = normalize_quaternion(multiply(misalignment[i], orientation[i]))
            z_pts = np.empty((count, 4))
            for k in range(count):
                z_pts[k] = normalize_quaternion(multiply(mount_q, q_pts[k]))
            z_mean = quaternion_mean(z_pts, w_m)
            z_mean_conj = conjugate(z_mean)
            for k in range(count):
                dz_buf[m_total : m_total + 3, k] = quaternion_to_rotation_vector(
                    multiply(z_pts[k], z_mean_conj)
                )
            res_buf[m_total : m_total + 3] = quaternion_to_rotation_vector(
                multiply(normalize_quaternion(observed), z_mean_conj)
            )
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
                dz_buf[m_total : m_total + meas_width, :] = _wrap_angles(z_pts - z_mean).T
                res_buf[m_total : m_total + meas_width] = _wrap_angles(observed - z_mean)
            else:
                z_mean = w_m @ z_pts
                dz_buf[m_total : m_total + meas_width, :] = (z_pts - z_mean).T
                res_buf[m_total : m_total + meas_width] = observed - z_mean

        r_buf[m_total : m_total + meas_width, m_total : m_total + meas_width] = (
            _sensor_noise_matrix(kind, meas_width, chol[i], scalars[i])
        )
        m_total += meas_width

    if m_total > 0:
        dz_t = dz_buf[:m_total, :]
        res_all = res_buf[:m_total]
        r_all = r_buf[:m_total, :m_total]

        p_zz = (dz_t * w_c) @ dz_t.T + r_all
        p_zz = 0.5 * (p_zz + p_zz.T)
        p_xz = (dx_pts.T * w_c) @ dz_t.T
        gain = np.linalg.solve(p_zz, p_xz.T).T
        delta = gain @ res_all
        p_prior = p

        q = normalize_quaternion(
            multiply(rotation_vector_to_quaternion(delta[:3]), q)
        )
        w = w + delta[3:]
        p_raw = p - gain @ p_zz @ gain.T
        reset = np.eye(6)
        reset[:3, :3] += 0.5 * skew(delta[:3])
        p = reset @ p_raw @ reset.T
        p = 0.5 * (p + p.T)

        if diagnostics is not None:
            diag_w = min(3, m_total)
            if flags & 16:
                diagnostics[:3] = np.nan
                diagnostics[:diag_w] = res_all[:diag_w]
            if flags & 32:
                block = np.full((3, 3), np.nan)
                block[:diag_w, :diag_w] = p_zz[:diag_w, :diag_w]
                diagnostics[3:12] = block.ravel()
            if flags & 64:
                diagnostics[12] = np.dot(res_all, np.linalg.solve(p_zz, res_all))
            if flags & 256:
                h_eff = np.linalg.solve(p_prior, p_xz).T
                post_res = res_all - h_eff @ delta
                diagnostics[14:17] = np.nan
                diagnostics[14 : 14 + diag_w] = post_res[:diag_w]
            if flags & 512:
                block = np.full((6, 3), np.nan)
                block[:, :diag_w] = gain[:, :diag_w]
                diagnostics[17:35] = block.ravel()

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
