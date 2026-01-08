__all__ = ["Combined_MTQ"]

import numpy as np

from ADCS.CONOPS.goals import Goal
from ADCS.controller import Controller
from ADCS.controller.helpers.quaternion_math import vector_alignment_error
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat, skewsym, limit

class Combined_MTQ(Controller):
    r"""
    Combined RW + MTQ attitude controller with MTQ-assisted control and MTQ desaturation.

    - RW + MTQ both contribute to attitude control torque.
    - MTQ also performs momentum dumping (desaturation).
    - MTQ torque is always projected into the plane orthogonal to B (physical constraint).

    Tuning knobs:
      - mtq_share in [0,1]: how much of the *achievable* control torque MTQ should attempt
        (0 -> RW-only control, 1 -> MTQ does as much control as physically possible).
      - c_gain: momentum dumping gain (same idea as your MTQ_w_RW).
    """

    def __init__(
        self,
        est_sat: EstimatedSatellite,
        p_gain: float,
        d_gain: float,
        c_gain: float,
        h_target: np.ndarray,
        mtq_share: float = 0.35,
        pinv_rcond: float = 1e-6,
    ) -> None:
        self.p_gain = float(p_gain)
        self.d_gain = float(d_gain)
        self.c_gain = float(c_gain)
        self.mtq_share = float(mtq_share)
        self.pinv_rcond = float(pinv_rcond)

        # Sensors: MTM readout mapping
        self.M_mtm_read, self.mtm_indices = self.build_sensor_matrix_pinv(
            sensors=est_sat.sensors + est_sat.rw_actuators, sensor_type=MTM
        )

        # Actuators: torque->u maps
        self.M_rw_act, self.rw_indices = self.build_torque_to_u_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=RW
        )
        self.M_mtq_act, self.mtq_indices = self.build_torque_to_u_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=MTQ
        )

        # MTQ geometry mapping used in tau = -[B]x A_mtq u_mtq_type
        self.A_mtq = self.build_u_to_torque_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=MTQ
        )

        A_rw_full = self.build_u_to_torque_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=RW
        )
        self.A_rw = A_rw_full[:, self.rw_indices]

        # RW momentum limits and generic saturation bound
        self.rw_max_h = np.asarray([rw.h_max for rw in est_sat.actuators if isinstance(rw, RW)])

        # Store body-frame target momentum
        self.h_target = np.asarray(h_target, dtype=float).reshape(3,)

        # Feasibility check: does there exist h_w such that A_rw h_w = h_target within limits?
        # (Least-norm solution; if infeasible, this will typically violate limits)
        h_w_star = np.linalg.pinv(self.A_rw) @ self.h_target  # (N_RW,)

        if np.any(np.abs(h_w_star) > self.rw_max_h):
            raise ValueError(
                "Target body-frame momentum h_target is infeasible "
                "given RW geometry and momentum limits."
            )

        # Saturation bound (all actuators, ordered like full u)
        self.max_torque = self.find_max_torque(actuators=est_sat.actuators)

        self.n_actuators = len(est_sat.actuators)

    @staticmethod
    def _proj_perp_to_b(tau: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Project a torque vector into the plane orthogonal to magnetic field b."""
        b = np.asarray(b, float).reshape(3,)
        bn = np.linalg.norm(b)
        if bn < 1e-12:
            return np.zeros(3)
        bh = b / bn
        return tau - (bh @ tau) * bh


    def find_u(
        self,
        x_hat: np.ndarray,
        sens: np.ndarray,
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal: Goal | None = None,
    ) -> np.ndarray:

        w = x_hat[0:3]
        q = x_hat[3:7]

        if goal is None:
            goal = Goal()
        goal_vector_eci, w_ref_eci = goal.to_ref(os0=os_hat)

        # Magnetic field in body frame from MTM(s)
        b_body = self.M_mtm_read @ sens

        # Attitude error (vector part) and rate error
        q_err_vec = vector_alignment_error(q=q, eci_goal=goal_vector_eci, body_boresight=est_sat.boresight)
        R_b2i = rot_mat(q)
        w_ref_body = R_b2i.T @ w_ref_eci
        w_err = w - w_ref_body

        # Baseline control torque (RW+MTQ will share this)
        tau_ctrl = -self.p_gain * q_err_vec - self.d_gain * w_err

        # Gyro coupling compensation
        # Reconstruct RW momentum in body frame
        rw_axes = np.vstack([
            np.asarray(rw.axis, float).reshape(3,)
            for rw in est_sat.actuators
            if isinstance(rw, RW)
        ])  # (N_RW, 3)

        h_vals = x_hat[7:]  # assumes order matches RW order in actuators
        h_rw_body = h_vals @ rw_axes  # (3,)

        J = est_sat.J_0
        tau_gyro = np.cross(w, J @ w + h_rw_body)
        tau_ctrl = tau_ctrl + tau_gyro

        # Momentum dumping torque request (body frame)
        delta_h = h_rw_body - self.h_target
        tau_dump = -self.c_gain * delta_h

        # MTQ effective torque mapping:
        # tau_mag = M_mag_eff @ u_mtq_type, where M_mag_eff = -[B]x A_mtq
        B_skew = skewsym(b_body)
        M_mag_eff = -B_skew @ self.A_mtq  # (3, N_MTQ)

        # MTQ can only produce torque orthogonal to B: enforce projection
        tau_ctrl_perp = self._proj_perp_to_b(tau_ctrl, b_body)
        tau_dump_perp = self._proj_perp_to_b(tau_dump, b_body)

        # Decide how much MTQ should help with control (in achievable subspace)
        mtq_share = np.clip(self.mtq_share, 0.0, 1.0)
        tau_mag_cmd = mtq_share * tau_ctrl_perp + tau_dump_perp

        # Allocate MTQ dipoles/currents (type-space)
        # Use rcond for stability (avoid exploding commands near singularities / low B)
        u_mtq_type = np.linalg.pinv(M_mag_eff, rcond=self.pinv_rcond) @ tau_mag_cmd
        u_mtq_type = limit(u=u_mtq_type, umax=self.max_torque)  # TODO: per-actuator bounds if available

        tau_mag_actual = M_mag_eff @ u_mtq_type  # actual torque produced by MTQ

        # Remaining torque requested from RW
        tau_rw_req = tau_ctrl - tau_mag_actual

        # Allocate RW commands into full actuator vector
        u_rw_full = self.M_rw_act @ tau_rw_req
        u_rw_full = limit(u=u_rw_full, umax=self.max_torque)  # TODO: per-RW bounds

        # Pack MTQ commands into full actuator vector and sum
        u_mtq_full = u_mtq_type

        u_total = u_rw_full + u_mtq_full
        return u_total
