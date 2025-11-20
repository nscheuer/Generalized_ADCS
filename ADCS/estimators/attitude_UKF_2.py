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
        quat_as_vec: bool = False,
    ) -> None:
        """
        Unscented Kalman Filter with an error-state attitude representation.

        - self.x_hat.val is always the FULL state: [w(3), q(4), rest].
        - self.x_hat.cov is the covariance of the REDUCED error state:
            * quat_as_vec == False: [dw(3), dtheta(3), drest]  (dim = N-1)
            * quat_as_vec == True : [dw(3), dq(4),     drest]  (dim = N)
        """
        super().__init__(
            est_sat=est_sat,
            J2000=J2000,
            x_hat=x_hat,
            P_hat=P_hat,
            Q_hat=Q_hat,
            dt=dt,
            cross_term=cross_term,
            quat_as_vec=quat_as_vec,
        )

        self.dt = dt
        self.al = 1.0
        self.kap = 1.0
        self.bet = 2.0
        self.vec_mode = 6

    # ------------------------------------------------------------------
    # Quaternion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_quat(
        q: np.ndarray,
        hemisphere: bool = False,
        ref: np.ndarray | None = None,
    ) -> np.ndarray:
        """Normalize quaternion, optionally enforcing a hemisphere convention."""
        q = np.asarray(q, dtype=float).copy()
        n = np.linalg.norm(q)
        if n > 0.0:
            q /= n
        if hemisphere:
            if ref is not None:
                if np.dot(q, ref) < 0.0:
                    q = -q
            else:
                if q[0] < 0.0:
                    q = -q
        return q

    @staticmethod
    def _quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamilton product q = q1 * q2."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    @staticmethod
    def _quat_inv(q: np.ndarray) -> np.ndarray:
        """Inverse of a unit quaternion."""
        q = np.asarray(q, dtype=float)
        return np.array([q[0], -q[1], -q[2], -q[3]])

    # ------------------------------------------------------------------
    # State layout helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _split_state(x: np.ndarray):
        """
        Split full state x = [w(3), q(4), rest] into (w, q, rest).
        """
        w = x[0:3]
        q = x[3:7]
        rest = x[7:]
        return w, q, rest

    @staticmethod
    def _compose_state(w: np.ndarray, q: np.ndarray, rest: np.ndarray) -> np.ndarray:
        return np.concatenate([w, q, rest])

    def _state_error_dim(self, x: np.ndarray) -> int:
        """
        Given a full state vector x = [w(3), q(4), rest],
        return the expected dimension of the reduced error state.
        """
        N = x.size
        if N < 7:
            raise ValueError(f"State dimension {N} is too small; expected at least 7.")
        rest_len = N - 7
        if self.quat_as_vec:
            # [dw(3), dq(4), drest(rest_len)] -> dim = N
            return 3 + 4 + rest_len
        else:
            # [dw(3), dtheta(3), drest(rest_len)] -> dim = N-1
            return 3 + 3 + rest_len

    # ------------------------------------------------------------------
    # Boxplus / boxminus: full state <-> reduced error state
    # ------------------------------------------------------------------
    def _boxplus(self, x_ref: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """
        Apply an error-state increment dx to a reference full state x_ref.

        quat_as_vec == False:
            dx = [dw(3), dtheta(3), drest]
            q = q_ref ⊗ exp(dtheta)

        quat_as_vec == True:
            dx = [dw(3), dq(4), drest]
            q = normalize(q_ref + dq)
        """
        x_ref = np.asarray(x_ref, dtype=float)
        dx = np.asarray(dx, dtype=float)

        w_ref, q_ref, rest_ref = self._split_state(x_ref)
        rest_len = rest_ref.size

        if self.quat_as_vec:
            if dx.size != 3 + 4 + rest_len:
                raise ValueError("dx has inconsistent size for quat_as_vec=True.")
            dw = dx[0:3]
            dq = dx[3:7]
            drest = dx[7:]
            w = w_ref + dw
            q = q_ref + dq
            q = self._normalize_quat(q, hemisphere=False)
            rest = rest_ref + drest
        else:
            if dx.size != 3 + 3 + rest_len:
                raise ValueError("dx has inconsistent size for quat_as_vec=False.")
            dw = dx[0:3]
            dtheta = dx[3:6]
            drest = dx[6:]
            w = w_ref + dw

            dq = vec3_to_quat(dtheta, self.vec_mode)
            dq = self._normalize_quat(dq, hemisphere=False)
            q_ref = self._normalize_quat(q_ref, hemisphere=False)
            q = self._quat_mult(q_ref, dq)        # q = q_ref ⊗ dq
            q = self._normalize_quat(q, hemisphere=True, ref=q_ref)

            rest = rest_ref + drest

        return self._compose_state(w, q, rest)

    def _boxminus(self, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """
        Error-state difference dx that "adds" to x_ref to produce x.

        For quat_as_vec == False, attitude error lives in R^3 via the
        log map of the quaternion difference q_ref^{-1} ⊗ q.
        """
        x = np.asarray(x, dtype=float)
        x_ref = np.asarray(x_ref, dtype=float)

        w, q, rest = self._split_state(x)
        w_ref, q_ref, rest_ref = self._split_state(x_ref)

        dw = w - w_ref
        drest = rest - rest_ref
        rest_len = rest_ref.size

        if self.quat_as_vec:
            # Direct quaternion component-wise difference (small-angle assumption)
            dq = q - q_ref
            if dq.size != 4:
                raise ValueError("Quaternion part must have size 4 for quat_as_vec=True.")
            return np.concatenate([dw, dq, drest])
        else:
            q_ref_n = self._normalize_quat(q_ref, hemisphere=False)
            q_n = self._normalize_quat(q, hemisphere=True, ref=q_ref_n)
            dq = self._quat_mult(self._quat_inv(q_ref_n), q_n)  # q = q_ref ⊗ dq
            dq = self._normalize_quat(dq, hemisphere=False)
            dtheta = quat_to_vec3(dq, self.vec_mode)

            if dtheta.size != 3:
                raise ValueError("quat_to_vec3 must return a 3-vector.")

            if drest.size != rest_len:
                raise ValueError("Rest-state dimension mismatch in _boxminus.")

            return np.concatenate([dw, dtheta, drest])

    # ------------------------------------------------------------------
    # Sigma-point generation in error space
    # ------------------------------------------------------------------
    @staticmethod
    def _chol_from_cov(P_hat: np.ndarray) -> np.ndarray:
        """
        Robust Cholesky-like factorization for possibly PSD covariances.
        Returns a matrix L such that P ≈ L L^T.
        """
        P_hat = np.asarray(P_hat, dtype=float)
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

    def _make_error_sigma_points(self, P: np.ndarray):
        """
        Generate sigma points in the reduced error space δx ~ N(0, P).
        """
        P = np.asarray(P, dtype=float)
        n = P.shape[0]

        if n == 0:
            pts = np.zeros((1, 0))
            w_m = np.array([1.0])
            w_c = np.array([1.0])
            return pts, w_m, w_c

        Lmat = self._chol_from_cov(P)  # n x n

        lam = self.al**2 * (n + self.kap) - n
        c = n + lam
        scale = np.sqrt(c)

        pts = np.zeros((2 * n + 1, n))
        # pts[0] is the zero vector (mean error)

        for i in range(n):
            col = Lmat[:, i]
            pts[1 + i] = scale * col
            pts[1 + n + i] = -scale * col

        w_m = np.full(2 * n + 1, 0.5 / c)
        w_c = np.full(2 * n + 1, 0.5 / c)
        w_m[0] = lam / c
        w_c[0] = lam / c + (1.0 - self.al**2 + self.bet)

        return pts, w_m, w_c

    def make_pts_and_wts(self, y0: np.ndarray):
        """
        Backwards-compatible wrapper that ignores y0 and returns
        error-space sigma points based on the current covariance.
        """
        del y0  # unused
        return self._make_error_sigma_points(self.x_hat.cov)

    # ------------------------------------------------------------------
    # Dynamics / measurement models
    # ------------------------------------------------------------------
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

    @staticmethod
    def _flatten_sensors(sensors) -> np.ndarray:
        """
        Stack a list of sensor arrays into one 1D measurement vector.
        """
        if isinstance(sensors, np.ndarray):
            return sensors.ravel()
        return np.concatenate([np.asarray(s).ravel() for s in sensors])

    # ------------------------------------------------------------------
    # Main UKF step
    # ------------------------------------------------------------------
    def update_core(self, u: np.ndarray, sensors: List[np.ndarray], os: Orbital_State) -> EstimatedArray:
        """
        Core UKF step using an error-state formulation.

        - The nominal state x_hat.val is kept in full coordinates [w, q, rest].
        - The covariance x_hat.cov is over the reduced error state δx.
        """
        # Nominal prior state and covariance
        x0 = self.x_hat.val.copy()
        P0 = self.x_hat.cov.copy()
        Q = self.x_hat.int_cov.copy()

        # Check consistency between state and covariance dimensions
        n_expected = self._state_error_dim(x0)
        n = P0.shape[0]
        if n != n_expected:
            raise ValueError(
                f"Covariance dim {n} inconsistent with state dim {x0.size} "
                f"and quat_as_vec={self.quat_as_vec}. Expected {n_expected}."
            )

        # 1) Sigma points in error space around zero
        sigma_dx, w_m, w_c = self._make_error_sigma_points(P0)
        num_pts = sigma_dx.shape[0]

        # 2) Map error sigma points to full state space around x0
        sigma_x = np.zeros((num_pts, x0.size))
        for i in range(num_pts):
            sigma_x[i] = self._boxplus(x0, sigma_dx[i])

        # 3) Propagate each sigma point through the dynamics and measurement models
        sigma_x_pred = np.zeros_like(sigma_x)
        sigma_z = None

        for i in range(num_pts):
            x_i = sigma_x[i]
            x_pred_i = self._propagate_dynamics(x_i, u, os)
            sigma_x_pred[i] = x_pred_i

            z_i = self._predict_measurement(x_pred_i, os).ravel()
            if sigma_z is None:
                m_meas = z_i.size
                sigma_z = np.zeros((num_pts, m_meas))
            sigma_z[i, :] = z_i

        # 4) Choose reference predicted state as propagation of central sigma
        x_ref_pred = sigma_x_pred[0].copy()
        w_ref, q_ref, rest_ref = self._split_state(x_ref_pred)
        # Normalize reference quaternion to keep it on S^3
        if self.quat_as_vec:
            q_ref_n = self._normalize_quat(q_ref, hemisphere=False)
        else:
            q_ref_n = self._normalize_quat(q_ref, hemisphere=True)
        x_ref_pred = self._compose_state(w_ref, q_ref_n, rest_ref)

        # 5) Express all predicted sigmas as error-states around x_ref_pred
        sigma_dx_pred = np.zeros_like(sigma_dx)
        for i in range(num_pts):
            sigma_dx_pred[i] = self._boxminus(sigma_x_pred[i], x_ref_pred)

        # 6) Predicted mean error and nominal predicted state
        dx_mean = np.sum(w_m[:, None] * sigma_dx_pred, axis=0)
        x_pred_mean = self._boxplus(x_ref_pred, dx_mean)

        # 7) Predicted covariance in error space
        Dx = sigma_dx_pred - dx_mean
        P_pred = np.zeros_like(P0)
        for i in range(num_pts):
            P_pred += w_c[i] * np.outer(Dx[i], Dx[i])
        P_pred += Q

        # 8) Measurement mean and covariances
        z_mean = np.sum(w_m[:, None] * sigma_z, axis=0)

        Dz = sigma_z - z_mean
        P_zz = np.zeros((m_meas, m_meas))
        P_xz = np.zeros((n, m_meas))
        for i in range(num_pts):
            P_zz += w_c[i] * np.outer(Dz[i], Dz[i])
            P_xz += w_c[i] * np.outer(Dx[i], Dz[i])

        # Add measurement noise covariance R = L_R L_R^T
        sens_srcov = self.est_sat.sensor_srcov()
        if sens_srcov.shape[0] != m_meas:
            raise ValueError(
                f"sensor_srcov() dimension {sens_srcov.shape[0]} "
                f"does not match measurement dim {m_meas}"
            )
        R_meas = sens_srcov.T @ sens_srcov
        P_zz += R_meas

        # 9) Kalman gain
        try:
            K = P_xz @ np.linalg.inv(P_zz)
        except np.linalg.LinAlgError:
            # Fall back if P_zz is near singular
            K = np.linalg.lstsq(P_zz, P_xz.T, rcond=None)[0].T

        # 10) Innovation and posterior update in error space
        z_meas = self._flatten_sensors(sensors)
        if z_meas.size != m_meas:
            raise ValueError(
                f"Measurement dimension mismatch: got {z_meas.size}, "
                f"expected {m_meas}"
            )

        innov = z_meas - z_mean
        dx_update = K @ innov

        x_post = self._boxplus(x_pred_mean, dx_update)

        # Posterior covariance
        P_post = P_pred - K @ P_zz @ K.T
        P_post = 0.5 * (P_post + P_post.T)  # symmetrize

        # Optional continuity enforcement w.r.t previous quaternion
        q_prev = x0[3:7].copy()
        w_post, q_post, rest_post = self._split_state(x_post)
        if not self.quat_as_vec:
            q_post = self._normalize_quat(q_post, hemisphere=True, ref=q_prev)
        else:
            q_post = self._normalize_quat(q_post, hemisphere=False)
        x_post = self._compose_state(w_post, q_post, rest_post)

        # For quat_as_vec=True we simply renormalize quaternion; a more exact
        # match to the old code would also transform P_post with the Jacobian
        # of the normalization. That can be added if needed.

        return EstimatedArray(val=x_post, cov=P_post, int_cov=self.x_hat.int_cov)