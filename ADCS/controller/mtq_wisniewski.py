__all__ = ["MTQ_Wisniewski"]

import numpy as np

from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.controller import Controller
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat, Wmat

class MTQ_Wisniewski(Controller):
    r"""
    MTQ_Wisniewski
    ==============

    Sliding Mode Magnetic Attitude Controller
    -----------------------------------------

    This controller implements the **Sliding Mode Control (SMC)** law for magnetic actuation proposed by Rafal Wisniewski (1998).

    Unlike linear controllers (PD/LQR) which may struggle with the highly nonlinear and time-varying nature of magnetic control, this SMC formulation drives the system states onto a predefined "sliding surface" in the phase plane. Once on this surface, the system dynamics are governed by the surface design itself, providing robustness against model uncertainties and disturbances.

    

    Reference
    ^^^^^^^^^
    **Wisniewski, R.** "Sliding Mode Attitude Control for Magnetic Actuated Satellite."
    *IFAC Proceedings Volumes*, 31(18), 1998.

    Control Law Derivation
    ----------------------

    **1. Error Definitions**

    Let the attitude error quaternion be :math:`\boldsymbol{q}_{\mathrm{err}}` and the angular velocity error be :math:`\boldsymbol{\omega}_{\mathrm{err}} = \boldsymbol{\omega} - \boldsymbol{\omega}_{\mathrm{ref}}`.

    **2. Sliding Surface**

    The sliding manifold :math:`\boldsymbol{s}` is defined as a linear combination of the angular momentum error and the attitude error:

    .. math::

        \boldsymbol{s} = J \boldsymbol{\omega}_{\mathrm{err}} + \Lambda_q \boldsymbol{q}_{\mathrm{err}}

    where :math:`\Lambda_q` is a positive definite gain matrix governing the convergence speed of the attitude error once the sliding mode is established (:math:`\boldsymbol{s} \approx 0`).

    **3. Lyapunov Stability & Control Torque**

    To ensure reachability of the sliding surface, we choose a Lyapunov candidate :math:`V = \frac{1}{2} \boldsymbol{s}^T \boldsymbol{s}`. The condition :math:`\dot{V} < 0` leads to the desired control torque definition.

    Differentiating the surface:

    .. math::

        \dot{\boldsymbol{s}} = J \dot{\boldsymbol{\omega}}_{\mathrm{err}} + \Lambda_q \dot{\boldsymbol{q}}_{\mathrm{err}}

    Substituting Euler's equations of motion and solving for the control torque :math:`\boldsymbol{\tau}_{\mathrm{ctrl}}` that cancels nonlinearities (feedback linearization) and imposes a decaying dynamics :math:`-\Lambda_s \boldsymbol{s}`:

    .. math::

        \boldsymbol{\tau}_{\mathrm{des}} =
        \boldsymbol{\omega} \times (J \boldsymbol{\omega} + \boldsymbol{h}_{\mathrm{rw}})
        + J (\boldsymbol{\omega} \times \boldsymbol{\omega}_{\mathrm{err}})
        - \Lambda_q \dot{\boldsymbol{q}}_{\mathrm{err}}
        - \Lambda_s \boldsymbol{s}

    **4. Magnetic Allocation**

    Since magnetic torque is constrained to be perpendicular to the local B-field (:math:`\boldsymbol{\tau} = \boldsymbol{m} \times \boldsymbol{B}`), the desired torque :math:`\boldsymbol{\tau}_{\mathrm{des}}` is projected onto the available plane using the standard cross-product law:

    .. math::

        \boldsymbol{m} = \frac{\boldsymbol{B} \times \boldsymbol{\tau}_{\mathrm{des}}}{\|\boldsymbol{B}\|^2}

    """
    def __init__(self, est_sat: EstimatedSatellite, lambda_s: np.ndarray, lambda_q: np.ndarray) -> None:
        self.lambda_s = lambda_s
        self.lambda_q = lambda_q

        self.M_read, self.mtm_indices = self.build_sensor_matrix_pinv(sensors=est_sat.attitude_sensors+est_sat.rw_actuators, sensor_type=MTM)

        self.mtq_umax = np.array([a.u_max for a in est_sat.actuators if isinstance(a, MTQ)], dtype=float)
        
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

        # Calculate sliding surface
        J = est_sat.J_0

        s = J @ w_err + self.lambda_q @ q_err 

        h_rw_body = np.zeros(3)
        rw_counter = 0
        for actuator in est_sat.actuators:
            if isinstance(actuator, RW):
                h_rw_body += np.asarray(actuator.axis).flatten() * h_rw_states[rw_counter]
                rw_counter += 1

        q_err_full = np.hstack(([np.sqrt(max(0.0, 1.0 - np.dot(q_err, q_err)))], q_err))
        q_err_dot = 0.5*w_err@Wmat(q_err_full).T

        tau_gyro = np.cross(w, J @ w + h_rw_body)
        tau_frame = J @ np.cross(w, w_err)
        tau_q_err_dot = self.lambda_q @ q_err_dot[1:4]
        tau_sliding = self.lambda_s @ s

        tau_des = tau_gyro + tau_frame - tau_q_err_dot - tau_sliding

        y = np.asarray(sens).reshape(-1)
        B_curr = self.M_read @ y
        B_norm_sq = np.linalg.norm(B_curr)**2

        if B_norm_sq < 1e-11:
            u_mtq_cmd = np.zeros(3)
        else:
            u_mtq_cmd = np.cross(B_curr, tau_des) / B_norm_sq

        scale = np.min(
            np.where(np.abs(u_mtq_cmd) > 0.0,
                    self.mtq_umax / np.abs(u_mtq_cmd),
                    np.inf)
        )
        u_mtq_cmd *= min(1.0, scale)

        u_out = np.zeros(len(est_sat.actuators))
        mtq_indices = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, MTQ)]
        u_out[mtq_indices] = u_mtq_cmd

        return u_out