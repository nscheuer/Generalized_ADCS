__all__ = ["Orbit_EKF"]

import numpy as np
from typing import List
from scipy.linalg import block_diag

from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.estimators.estimator_helpers.estimator_helpers import EstimatedOrbital_State
from ADCS.estimators.orbit_estimators import Orbit_Estimator

class Orbit_EKF(Orbit_Estimator):
    r"""
    Extended Kalman Filter (EKF) for Satellite Orbit Determination.

    This class implements a continuous-discrete Extended Kalman Filter to estimate 
    the satellite's state vector :math:`\mathbf{x} \in \mathbb{R}^6` (ECI Position and Velocity).

    The filter linearizes the non-linear orbital dynamics :math:`f(\mathbf{x}, t)` 
    (including J2 perturbations) to propagate the error covariance matrix :math:`\mathbf{P}`.

    Attributes:
        est_sat (EstimatedSatellite): Satellite model containing sensor specifications.
        R (np.ndarray): Measurement noise covariance matrix :math:`\mathbf{R}`.
        os_hat (EstimatedOrbital_State): Current state estimate :math:`\hat{\mathbf{x}}` 
            and covariance :math:`\mathbf{P}`.
    """
    def __init__(
        self,
        est_sat: EstimatedSatellite,
        J2000: float,
        os_hat: Orbital_State,
        P_hat: np.ndarray,
        Q_hat: np.ndarray,
        dt: float = 1.0
    ) -> None:
        r"""
        Initialize the Orbit EKF.

        :param est_sat: Satellite object with GPS sensors.
        :param J2000: Initial epoch time [s].
        :param os_hat: Initial orbital state estimate :math:`\hat{\mathbf{x}}_0`.
        :param P_hat: Initial state error covariance matrix :math:`\mathbf{P}_0` (6x6).
        :param Q_hat: Process noise covariance matrix :math:`\mathbf{Q}` (6x6).
        :param dt: Propagation time step [s].
        :raises ValueError: If the satellite has no GPS sensors.
        """
        super().__init__(est_sat=est_sat, dt=dt)
        if not self.est_sat.GPS_sensors:
            return ValueError("Satellite must have at least one GPS sensor!")

        self.reset(est_sat=est_sat, J2000=J2000, os_hat=os_hat, P_hat=P_hat, Q_hat=Q_hat, dt=dt)

    def reset(
        self, 
        est_sat: EstimatedSatellite,
        J2000: float,
        os_hat: Orbital_State,
        P_hat: np.ndarray,
        Q_hat: np.ndarray,
        dt: float = 1.0
    ) -> None:
        r"""
        Reset the filter state and matrices.

        Initializes the estimate :math:`\hat{\mathbf{x}}` and constructs the measurement 
        noise matrix :math:`\mathbf{R}` based on the standard deviation of the 
        onboard GPS sensors.

        .. math::
            \mathbf{R} = \text{block\_diag}(\sigma_{GPS,1}^2 \mathbf{I}, \dots)

        :param est_sat: Satellite hardware model.
        :param J2000: Current J2000 epoch.
        :param os_hat: Initial state guess.
        :param P_hat: Initial covariance :math:`\mathbf{P}` (must be 6x6).
        :param Q_hat: Process noise :math:`\mathbf{Q}` (must be 6x6).
        :param dt: Time step.
        """
        if P_hat.shape != (6, 6):
            raise ValueError(f"P must be 6×6, got {P_hat.shape}")
        if Q_hat.shape != (6, 6):
            raise ValueError(f"Q must be 6×6, got {Q_hat.shape}")
        self.os_hat = EstimatedOrbital_State(os=os_hat, P=P_hat, Q=Q_hat)

        gps_sensors = self.est_sat.GPS_sensors
        blocks = []
        for gps in gps_sensors:
            std = gps.noise.std_noise
            R_i = np.diag(std**2)
            blocks.append(R_i)

        # Block diag for multiple sensors
        self.R = block_diag(*blocks)

    def update(
        self,
        GPS_measurements: List[np.ndarray],
        J2000: float
    ) -> EstimatedOrbital_State:
        r"""
        Perform the EKF Time Update and Measurement Update steps.

        1. Time Update (Prediction):
        ----------------------------
        Propagates the state using RK4 integration and the covariance using the 
        state transition matrix approximation:

        .. math::
            \hat{\mathbf{x}}_k^- &= f(\hat{\mathbf{x}}_{k-1}, u_{k-1}) \\
            \mathbf{P}_k^- &= \mathbf{F}_k \mathbf{P}_{k-1} \mathbf{F}_k^T + \mathbf{Q}

        2. Measurement Update (Correction):
        -----------------------------------
        Updates the estimate using the Kalman Gain :math:`\mathbf{K}` derived from 
        the measurement residual (innovation) :math:`\mathbf{y}`:

        .. math::
            \mathbf{K}_k &= \mathbf{P}_k^- \mathbf{H}^T (\mathbf{H} \mathbf{P}_k^- \mathbf{H}^T + \mathbf{R})^{-1} \\
            \hat{\mathbf{x}}_k &= \hat{\mathbf{x}}_k^- + \mathbf{K}_k (\mathbf{z}_k - \mathbf{H}\hat{\mathbf{x}}_k^-) \\
            \mathbf{P}_k &= (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_k^-

        :param GPS_measurements: List of sensor measurements (Position or PV) in ECEF frame.
        :param J2000: Current epoch time [s].
        :return: The updated :class:`~ADCS.estimators.estimator_helpers.estimator_helpers.EstimatedOrbital_State`.
        """

        # --- 1. Propagate the orbital state ---
        os0: Orbital_State = self.os_hat.os

        os_pred: Orbital_State = os0.propagate_orbit_rk4(
            dt=self.dt,
            zonal_J=2,
            fast=True,
        )
        r_pred, v_pred = os_pred.R, os_pred.V
        x_pred = np.hstack([r_pred, v_pred])   # (6,)

        # Dynamics Jacobian Fk
        dr_dr0, dr_dv0, dv_dr0, dv_dv0 = os0.orbit_dynamics_jacobians(
            zonal_J=2
        )
        Fk = np.block([
            [dr_dr0, dr_dv0],
            [dv_dr0, dv_dv0],
        ])  # (6×6)

        P0 = self.os_hat.P      # 6×6
        Q0 = self.os_hat.Q      # 6×6
        P_pred = Fk @ P0 @ Fk.T + Q0

        # --- 2. If no measurements, just return prediction ---
        if GPS_measurements is None or len(GPS_measurements) == 0:
            self.os_hat = EstimatedOrbital_State(os=os_pred, P=P_pred, Q=Q0)
            return self.os_hat

        # --- 3. Build z (measurement) in the SAME FRAME as x_pred (ECI) ---
        # GPS_measurements are likely in ECEF (depending on your sensor implementation),
        # so we convert them to ECI using os_pred.ecef_to_eci.

        z_list = []
        for m in GPS_measurements:
            m = np.asarray(m).reshape(-1)
            if m.size == 3:
                # Position-only GPS: ECEF position -> ECI
                r_ecef = m
                r_eci  = os_pred.ecef_to_eci(r_ecef)
                z_list.append(r_eci)
            elif m.size == 6:
                # Position + velocity:
                r_ecef = m[0:3]
                v_ecef = m[3:6]
                r_eci  = os_pred.ecef_to_eci(r_ecef)
                v_eci  = os_pred.ecef_to_eci(v_ecef)
                z_list.append(np.hstack([r_eci, v_eci]))
            else:
                raise ValueError(f"GPS measurement must have length 3 or 6, got {m.size}")

        z = np.concatenate(z_list)
        m_i = z_list[0].size        # 3 or 6
        n_sens = len(z_list)
        m_total = m_i * n_sens

        # --- 4. Predicted measurement h(x) in the same space ---
        # For this simple EKF, treat GPS as a direct measurement of [R,V] in ECI:
        #   if m_i == 6: z = [R; V]
        #   if m_i == 3: z = R
        if m_i == 3:
            # Only position measured → h is R part of x_pred
            h_single = x_pred[0:3]
            H_i = np.hstack([np.eye(3), np.zeros((3,3))])
        elif m_i == 6:
            h_single = x_pred
            H_i = np.eye(6)
        else:
            # should be unreachable because of the earlier check
            raise RuntimeError("Unexpected measurement dimension")

        h = np.concatenate([h_single for _ in range(n_sens)])  # (m_total,)
        H = np.vstack([H_i for _ in range(n_sens)])            # (m_total × 6)

        # --- 5. Measurement noise covariance R (already built in reset) ---
        if m_i == 3:
            n_gps = len(self.est_sat.GPS_sensors)
            if self.R.shape == (6 * n_gps, 6 * n_gps):
                blocks = []
                for i in range(n_sens):
                    start = 6 * i
                    blocks.append(self.R[start : start + 3, start : start + 3])
                R = block_diag(*blocks)
            else:
                R = self.R
        else:
            R = self.R
        if R.shape != (m_total, m_total):
            raise ValueError(f"R must be {m_total}×{m_total}, got {R.shape}")

        # --- 6. EKF update ---
        y = z - h                          # innovation
        S = H @ P_pred @ H.T + R           # innovation covariance
        K = P_pred @ H.T @ np.linalg.inv(S)

        x_upd = x_pred + K @ y
        P_upd = (np.eye(6) - K @ H) @ P_pred
        P_upd = 0.5 * (P_upd + P_upd.T)    # enforce symmetry

        # --- 7. Build updated Orbital_State ---
        os_upd = Orbital_State(
            ephem=os0.ephem,
            J2000=J2000,
            R=x_upd[0:3],
            V=x_upd[3:6],
            S=None,
            B=None,
            rho=None,
            density_model=os0.density_model,
            fast=False,
        )

        self.os_hat = EstimatedOrbital_State(
            os=os_upd,
            P=P_upd,
            Q=Q0,
        )

        self.innovation = y

        return self.os_hat
