__all__ = ["MTQ_Lovera"]

import numpy as np

from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.controller import Controller
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat

class MTQ_Lovera(Controller):
    def __init__(self, est_sat: EstimatedSatellite, p_gain: float, d_gain: float, eps: float) -> None:
        self.p_gain = p_gain
        self.d_gain = d_gain
        self.eps = eps

        self.M_read, self.mtm_indices = self.build_sensor_matrix_pinv(sensors=est_sat.attitude_sensors+est_sat.rw_actuators, sensor_type=MTM)


    def find_u(self, x_hat: np.ndarray, sens: np.ndarray, est_sat: EstimatedSatellite, os_hat: Orbital_State, goal: Goal | None = None) -> np.ndarray:
        if goal is None:
            goal = No_Goal()

        w = x_hat[0:3]
        q = x_hat[3:7]

        n_rw = len([a for a in est_sat.actuators if isinstance(a, RW)])
        if len(x_hat) >= 7 + n_rw:
            h_rw_states = x_hat[7 : 7 + n_rw]
        else:
            h_rw_states = np.array([rw.h for rw in est_sat.actuators if isinstance(rw, RW)])

        goal_vec_eci, w_ref_eci = goal.to_ref(os0=os_hat)
        R_b2i = rot_mat(q)
        w_ref_body = R_b2i.T @ w_ref_eci

        q_err = goal.error(q=q, body_boresight=est_sat.boresight, os0=os_hat)
        w_err = w - w_ref_body
        
        tau_pd = -(self.eps**2*self.p_gain*q_err + self.eps*self.d_gain*w_err)

        h_rw_body = np.zeros(3)
        rw_counter = 0
        for actuator in est_sat.actuators:
            if isinstance(actuator, RW):
                h_rw_body += np.asarray(actuator.axis).flatten() * h_rw_states[rw_counter]
                rw_counter += 1
        
        J = est_sat.J_0
        tau_gyro = np.cross(w, J @ w + h_rw_body)

        tau_des = tau_pd + tau_gyro

        y = np.asarray(sens).reshape(-1)
        B_curr = self.M_read @ y
        B_norm_sq = np.linalg.norm(B_curr)**2

        if B_norm_sq < 1e-11:
            u_mtq_cmd = np.zeros(3)
        else:
            u_mtq_cmd = np.cross(B_curr, tau_des) / B_norm_sq

        u_out = np.zeros(len(est_sat.actuators))
        mtq_indices = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, MTQ)]
        u_out[mtq_indices] = u_mtq_cmd

        return u_out