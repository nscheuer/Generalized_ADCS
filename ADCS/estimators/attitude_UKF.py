__all__ = ["UKF"]

import numpy as np
from typing import List

from ADCS.estimators.estimator import Estimator
from ADCS.estimators.estimator_helpers.estimator_helpers import EstimatedArray
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import quat_to_vec3, vec3_to_quat


class UKF(Estimator):
    def __init__(
        self,
        est_sat: EstimatedSatellite,
        J2000: float,
        x_hat: np.ndarray,
        P_hat: np.ndarray,
        Q_hat: np.ndarray,
        dt: float = 1.0,
        cross_term: bool = False,
        quat_as_vec: bool = False
    ) -> None:
        super().__init__(
            est_sat=est_sat,
            J2000=J2000,
            x_hat=x_hat,
            P_hat=P_hat,
            Q_hat=Q_hat,
            dt=dt,
            cross_term=cross_term,
            quat_as_vec=quat_as_vec
        )

        self.dt = dt
        self.al = 1.0
        self.kap = 1.0
        self.bet = 2.0
        self.vec_mode = 6

    @staticmethod
    def _chol_from_cov(P_hat: np.ndarray) -> np.ndarray:
        """
        Robust Cholesky-like factorization for possibly PSD covariances.
        Returns a matrix L such that P ≈ L L^T.
        """
        if P_hat.size == 0:
            return P_hat.reshape(0, 0)

        try:
            return np.linalg.cholesky(P_hat)
        except np.linalg.LinAlgError:
            # Fallback: eigen-decomposition, clamp small negatives
            w, v = np.linalg.eig(P_hat)
            w = np.maximum(np.real(w), 0.0)
            srw = np.diag(np.sqrt(w))
            v = np.real(v)
            L = v @ srw @ v.T
            return L

    def _x_to_y(self, x: np.ndarray) -> np.ndarray:
        """
        Map full state x = [w(3), q(4), rest] to reduced state
        y = [w(3), v3(3), rest], where v3 is a 3-parameter attitude
        representation (log map).

        NEW: enforce a quaternion hemisphere convention before conversion
        to avoid q / -q ambiguity causing ~180 deg flips.
        """
        w = x[0:3]
        q = x[3:7].copy()
        rest = x[7:]

        # Normalize quaternion and enforce scalar >= 0 (hemisphere)
        nq = np.linalg.norm(q)
        if nq > 0.0:
            q = q / nq
        if q[0] < 0.0:
            q = -q

        v3 = quat_to_vec3(q, self.vec_mode)
        return np.concatenate([w, v3, rest])

    def _y_to_x(self, y: np.ndarray) -> np.ndarray:
        """
        Map reduced state y = [w(3), v3(3), rest] back to full state
        x = [w(3), q(4), rest].

        NEW: normalize and enforce hemisphere convention on q.
        """
        w = y[0:3]
        v3 = y[3:6]
        rest = y[6:]

        q = vec3_to_quat(v3, self.vec_mode)

        # Normalize and enforce scalar >= 0
        nq = np.linalg.norm(q)
        if nq > 0.0:
            q = q / nq
        if q[0] < 0.0:
            q = -q

        return np.concatenate([w, q, rest])

    @staticmethod
    def _flatten_sensors(sensors) -> np.ndarray:
        """
        Stack a list of sensor arrays into one 1D measurement vector.
        """
        if isinstance(sensors, np.ndarray):
            return sensors.ravel()
        return np.concatenate([np.asarray(s).ravel() for s in sensors])

    def make_pts_and_wts(self, y0: np.ndarray):
        """
        Standard (unscented) sigma-point generation around y0 using the
        current covariance self.x_hat.cov in reduced space.
        """
        P = self.x_hat.cov
        n = P.shape[0]

        if n == 0:
            pts = y0.reshape(1, -1)
            w_m = np.array([1.0])
            w_c = np.array([1.0])
            return pts, w_m, w_c

        L = n
        lam = self.al ** 2 * (L + self.kap) - L
        c = L + lam
        scale = np.sqrt(c)

        # Cholesky factor: P ≈ Lmat Lmatᵀ (L lower-triangular-like)
        Lmat = self._chol_from_cov(P)

        pts = np.zeros((2 * L + 1, n))
        pts[0] = y0

        for i in range(L):
            col = Lmat[:, i]
            pts[1 + i] = y0 + scale * col
            pts[1 + L + i] = y0 - scale * col

        w_m = np.full(2 * L + 1, 0.5 / c)
        w_c = np.full(2 * L + 1, 0.5 / c)
        w_m[0] = lam / c
        w_c[0] = lam / c + (1.0 - self.al ** 2 + self.bet)

        return pts, w_m, w_c

    def _propagate_dynamics(self, x: np.ndarray, u: np.ndarray, os: Orbital_State) -> np.ndarray:
        """
        One-step propagation of the full augmented state using the
        noiseless dynamics model on the EstimatedSatellite.
        """
        mid_os = self.prev_os.average(orbital_state_2=os)
        return self.est_sat.noiseless_rk4(
            x=x,
            u=u,
            dt=self.dt,
            orbital_state0=self.prev_os,
            orbital_state1=os,
            mid_orbital_state=mid_os,
        )

    def _predict_measurement(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        """
        Noiseless measurement prediction for the full augmented state.
        """
        noiseless = self.est_sat.noiseless_sensor_readings(x=x, os=os)
        return noiseless

    def update_core(self, u: np.ndarray, sensors: List[np.ndarray], os: Orbital_State) -> EstimatedArray:
        """
        Core UKF step in reduced state space (y), with quaternion handled
        via a 3-parameter mapping and hemisphere enforcement.

        Returns an EstimatedArray corresponding to the FULL augmented state
        (not just the physical state). Estimator.update() will handle
        insertion into self.x_hat and cross-term zeroing.
        """
        # Current full state and its reduced representation
        x0 = self.x_hat.val.copy()
        y0 = self._x_to_y(x0)

        # Sigma points in reduced space
        sigma_y, w_m, w_c = self.make_pts_and_wts(y0)
        num_pts, n = sigma_y.shape

        sigma_y_pred = np.zeros_like(sigma_y)
        sigma_z = None

        # Propagate each sigma point through dynamics and measurement models
        for i in range(num_pts):
            x_i = self._y_to_x(sigma_y[i])
            x_pred = self._propagate_dynamics(x_i, u, os)
            y_pred_i = self._x_to_y(x_pred)      # re-map with hemisphere enforcement
            sigma_y_pred[i] = y_pred_i

            z_i = self._predict_measurement(x_pred, os).ravel()
            if sigma_z is None:
                m_meas = z_i.size
                sigma_z = np.zeros((num_pts, m_meas))
            sigma_z[i, :] = z_i

        # Predicted mean in reduced space and measurement space
        y_mean = np.sum(w_m[:, None] * sigma_y_pred, axis=0)
        z_mean = np.sum(w_m[:, None] * sigma_z, axis=0)

        Dy = sigma_y_pred - y_mean
        Dz = sigma_z - z_mean

        # Predicted covariance in reduced state space
        if n == 0:
            P_pred = self.x_hat.cov
        else:
            P_pred = np.zeros((n, n))
            for i in range(num_pts):
                P_pred += w_c[i] * np.outer(Dy[i], Dy[i])
            P_pred += self.x_hat.int_cov

        # Form measurement covariance and cross-covariance
        z_meas = self._flatten_sensors(sensors)
        if z_meas.size != sigma_z.shape[1]:
            raise ValueError(
                f"Measurement dimension mismatch: got {z_meas.size}, "
                f"expected {sigma_z.shape[1]}"
            )

        sens_srcov = self.est_sat.sensor_srcov()
        m_meas = sens_srcov.shape[0]

        if m_meas != sigma_z.shape[1]:
            raise ValueError(
                f"sensor_srcov() dimension {m_meas} does not match measurement dim {sigma_z.shape[1]}"
            )

        # R = L_R L_R^T
        R_meas = sens_srcov.T @ sens_srcov

        P_zz = np.zeros((m_meas, m_meas))
        for i in range(num_pts):
            P_zz += w_c[i] * np.outer(Dz[i], Dz[i])
        P_zz += R_meas

        P_yz = np.zeros((n, m_meas))
        for i in range(num_pts):
            P_yz += w_c[i] * np.outer(Dy[i], Dz[i])

        # Kalman gain
        try:
            K = P_yz @ np.linalg.inv(P_zz)
        except np.linalg.LinAlgError:
            # Fall back if P_zz is near singular
            K = np.linalg.lstsq(P_zz, P_yz.T, rcond=None)[0].T

        # Innovation and posterior reduced-state
        innov = z_meas - z_mean
        y_post = y_mean + K @ innov

        # Posterior covariance
        P_post = P_pred - K @ P_zz @ K.T
        P_post = 0.5 * (P_post + P_post.T)  # symmetrize

        # Map reduced-state posterior back to full state, with normalized, hemisphere-consistent quaternion
        x_post = self._y_to_x(y_post)

        # Optionally enforce continuity w.r.t previous quaternion
        q_prev = x0[3:7]
        q_new = x_post[3:7].copy()
        if np.dot(q_prev, q_new) < 0.0:
            q_new = -q_new
            x_post[3:7] = q_new

        # Update the internal array; Estimator.update will push into est_sat
        # and handle cross-term zeroing on the full covariance.
        return EstimatedArray(val=x_post, cov=P_post, int_cov=self.x_hat.int_cov)
