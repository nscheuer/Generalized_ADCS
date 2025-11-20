__all__ = ["UKF"]

import numpy as np
from choldate import cholupdate, choldowndate
from typing import List

from scipy.linalg import solve_triangular

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
    ) -> None:
        super().__init__(
            est_sat=est_sat,
            J2000=J2000,
            x_hat=x_hat,
            P_hat=P_hat,
            Q_hat=Q_hat,
            dt=dt,
            cross_term=cross_term,
        )

        self.dt = dt
        self.al = 1.0
        self.kap = 0.0
        self.bet = 2.0
        self.vec_mode = 6

    @staticmethod
    def _chol_from_cov(P_hat: np.ndarray) -> np.ndarray:
        if P_hat.size == 0:
            return P_hat.reshape(0, 0)

        try:
            return np.linalg.cholesky(P_hat)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eig(P_hat)
            w = np.maximum(np.real(w), 0.0)
            srw = np.diag(np.sqrt(w))
            v = np.real(v)
            L = v @ srw @ v.T
            return L

    def _x_to_y(self, x: np.ndarray) -> np.ndarray:
        w = x[0:3]
        q = x[3:7]
        rest = x[7:]
        v3 = quat_to_vec3(q, self.vec_mode)
        return np.concatenate([w, v3, rest])

    def _y_to_x(self, y: np.ndarray) -> np.ndarray:
        w = y[0:3]
        v3 = y[3:6]
        rest = y[6:]
        q = vec3_to_quat(v3, self.vec_mode)
        return np.concatenate([w, q, rest])

    @staticmethod
    def _flatten_sensors(sensors) -> np.ndarray:
        if isinstance(sensors, np.ndarray):
            return sensors.ravel()
        return np.concatenate([np.asarray(s).ravel() for s in sensors])

    def make_pts_and_wts(self, y0: np.ndarray):
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

        # Cholesky factor: P = L Lᵀ (L lower-triangular)
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
        noiseless = self.est_sat.noiseless_sensor_readings(x=x, os=os)
        return noiseless

    def update_core(self, u: np.ndarray, sensors: List[np.ndarray], os: Orbital_State) -> EstimatedArray:
        x0 = self.x_hat.val
        y0 = self._x_to_y(x0)
        sigma_y, w_m, w_c = self.make_pts_and_wts(y0)
        num_pts, n = sigma_y.shape

        sigma_y_pred = np.zeros_like(sigma_y)
        sigma_z = None

        for i in range(num_pts):
            x_i = self._y_to_x(sigma_y[i])
            x_pred = self._propagate_dynamics(x_i, u, os)
            y_pred_i = self._x_to_y(x_pred)
            sigma_y_pred[i] = y_pred_i

            z_i = self._predict_measurement(x_pred, os).ravel()
            if sigma_z is None:
                m_meas = z_i.size
                sigma_z = np.zeros((num_pts, m_meas))
            sigma_z[i, :] = z_i

        y_mean = np.sum(w_m[:, None] * sigma_y_pred, axis=0)
        z_mean = np.sum(w_m[:, None] * sigma_z, axis=0)

        Dy = sigma_y_pred - y_mean
        Dz = sigma_z - z_mean

        if n == 0:
            P_pred = self.x_hat.cov
        else:
            P_pred = np.zeros((n, n))
            for i in range(num_pts):
                P_pred += w_c[i] * np.outer(Dy[i], Dy[i])
            P_pred += self.x_hat.int_cov

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

        R_meas = sens_srcov.T @ sens_srcov

        P_zz = np.zeros((m_meas, m_meas))
        for i in range(num_pts):
            P_zz += w_c[i] * np.outer(Dz[i], Dz[i])
        P_zz += R_meas

        P_yz = np.zeros((n, m_meas))
        for i in range(num_pts):
            P_yz += w_c[i] * np.outer(Dy[i], Dz[i])

        try:
            K = P_yz @ np.linalg.inv(P_zz)
        except np.linalg.LinAlgError:
            # Fall back to solve if P_zz is singular-ish
            K = np.linalg.lstsq(P_zz, P_yz.T, rcond=None)[0].T

        innov = z_meas - z_mean
        y_post = y_mean + K @ innov

        P_post = P_pred - K @ P_zz @ K.T
        P_post = 0.5 * (P_post + P_post.T)

        if not self.cross_term and P_post.size:
            n_rw = self.est_sat.number_RW
            ab0 = 6 + n_rw
            ab1 = ab0 + self.est_sat.act_bias_len
            sb0 = ab1
            sb1 = sb0 + self.est_sat.att_sens_bias_len
            d0 = sb1

            P_post[ab0:ab1, sb0:sb1] = 0.0
            P_post[sb0:sb1, ab0:ab1] = 0.0
            P_post[ab0:ab1, d0:] = 0.0
            P_post[d0:, ab0:ab1] = 0.0
            P_post[sb0:sb1, d0:] = 0.0
            P_post[d0:, sb0:sb1] = 0.0

        x_post = self._y_to_x(y_post)
        self.x_hat.val = x_post
        self.x_hat.cov = P_post

        self.est_sat.match_estimate(est_state=self.x_hat, dt=self.dt)

        phys_len = self.est_sat.state_len
        phys_val = x_post[:phys_len]
        phys_cov = P_post[:phys_len, :phys_len]

        return EstimatedArray(val=phys_val, cov=phys_cov)