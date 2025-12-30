__all__ = ["MTQ_w_RW_Projection_Split"]

import numpy as np

from ADCS.CONOPS.goals import Goal
from ADCS.controller import Controller
from ADCS.controller.helpers.quaternion_math import vector_alignment_error
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat, skewsym, limit


class MTQ_w_RW_Projection_Split(Controller):
    r"""
    MTQ + RW controller using a *projection split* with two practical tracking fixes:

    (A) Spin damping about the boresight (prevents "spinning around the target vector")
    (B) LOS rate feedforward (enables tracking of a moving target direction)

    Control structure:
      1) tau_des from PD on vector alignment + rate error (with optional gyro compensation)
      2) MTQ commands realize the achievable component of (tau_des + tau_dump) orthogonal to B
      3) RW commands realize the residual tau_des - tau_mtq_actual

    Notes / conventions in this codebase:
    - build_u_to_torque_matrix_pinv(..., actuator_type) returns (3, N_total_cmds) with zeros for non-target actuators.
    - build_torque_to_u_matrix_pinv(..., actuator_type) returns (N_total_cmds, 3) pseudoinverse allocation map.
    - MTQ allocation here is done in "full command vector" space, so u_mtq_full has length N_total_cmds.
    - RW momentum states are assumed to begin at x_hat[7]; for N_RW wheels we use x_hat[7:7+N_RW].
    """

    def __init__(
        self,
        est_sat: EstimatedSatellite,
        p_gain: float,
        d_gain: float,
        c_gain: float,
        h_target: np.ndarray,
        pinv_rcond: float = 1e-6,
        k_spin: float | None = None,
        los_rate_dt: float | None = None,
        los_rate_alpha: float = 0.5,
    ) -> None:
        self.p_gain = float(p_gain)
        self.d_gain = float(d_gain)
        self.c_gain = float(c_gain)
        self.pinv_rcond = float(pinv_rcond)

        # Spin damping gain (default: match derivative gain)
        self.k_spin = float(d_gain if k_spin is None else k_spin)

        # LOS-rate feedforward settings
        self.los_rate_dt = None if los_rate_dt is None else float(los_rate_dt)
        self.los_rate_alpha = float(los_rate_alpha)
        self._s_body_prev: np.ndarray | None = None
        self._w_los_filt = np.zeros(3, dtype=float)

        # MTM reconstruction
        self.M_mtm_read, self.mtm_indices = self.build_sensor_matrix_pinv(
            sensors=est_sat.sensors + est_sat.rw_actuators, sensor_type=MTM
        )

        # Full-length torque->u maps (aligned to full command vector)
        self.M_rw_act, self.rw_indices = self.build_torque_to_u_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=RW
        )
        self.M_mtq_act, self.mtq_indices = self.build_torque_to_u_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=MTQ
        )

        # Full-length forward axis matrices (aligned to full command vector)
        self.A_mtq = self.build_u_to_torque_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=MTQ
        )  # (3, N_total_cmds)

        A_rw_full = self.build_u_to_torque_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=RW
        )  # (3, N_total_cmds)
        self.A_rw = A_rw_full[:, self.rw_indices]  # (3, N_RW)

        # Wheel momentum limits in wheel space (N_RW,)
        self.rw_max_h = np.asarray([rw.h_max for rw in est_sat.actuators if isinstance(rw, RW)], dtype=float)

        # Body-frame momentum target (3,)
        self.h_target = np.asarray(h_target, dtype=float).reshape(3,)

        # Optional feasibility check (least-norm wheel momentum to realize body target)
        if self.A_rw.size != 0:
            h_w_star = np.linalg.pinv(self.A_rw) @ self.h_target  # (N_RW,)
            if h_w_star.shape != self.rw_max_h.shape:
                raise ValueError(
                    f"RW momentum shape mismatch: h_w_star {h_w_star.shape} vs rw_max_h {self.rw_max_h.shape}. "
                    "This suggests your RW 'axis' definition has multiple input channels per RW actuator."
                )
            if np.any(np.abs(h_w_star) > self.rw_max_h):
                raise ValueError(
                    "Target body-frame momentum h_target is infeasible "
                    "given RW geometry and momentum limits."
                )

        # Full per-command limits (must match full command vector length)
        self.max_u = self.find_max_torque(actuators=est_sat.actuators)

    @staticmethod
    def _proj_perp_to_b(vec3: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Project a 3-vector into the plane orthogonal to magnetic field b."""
        b = np.asarray(b, dtype=float).reshape(3,)
        bn = np.linalg.norm(b)
        if bn < 1e-12:
            return np.zeros(3, dtype=float)
        bh = b / bn
        return vec3 - (bh @ vec3) * bh

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float).reshape(3,)
        n = np.linalg.norm(v)
        if n < 1e-12:
            return np.zeros(3, dtype=float)
        return v / n

    def _los_rate_body(self, R_b2i: np.ndarray, s_eci: np.ndarray, dt: float) -> np.ndarray:
        """
        Estimate the angular rate needed to track the LOS direction, in body frame.
        Uses w_los ≈ s × s_dot in body coordinates, with optional first-order filtering.
        """
        s_eci_hat = self._unit(s_eci)
        R_i2b = R_b2i.T
        s_body = R_i2b @ s_eci_hat  # unit LOS in body

        if self._s_body_prev is None or dt <= 0.0:
            self._s_body_prev = s_body
            self._w_los_filt[:] = 0.0
            return np.zeros(3, dtype=float)

        s_dot = (s_body - self._s_body_prev) / dt
        w_los = np.cross(s_body, s_dot)

        # simple low-pass filter to reduce numerical noise
        a = np.clip(self.los_rate_alpha, 0.0, 1.0)
        self._w_los_filt = a * self._w_los_filt + (1.0 - a) * w_los

        self._s_body_prev = s_body
        return self._w_los_filt.copy()

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

        # Magnetic field in body frame
        b_body = self.M_mtm_read @ sens

        # Attitude error (vector alignment) + base rate error
        q_err_vec = vector_alignment_error(q=q, eci_goal=goal_vector_eci, body_boresight=est_sat.boresight)

        R_b2i = rot_mat(q)
        w_ref_body = R_b2i.T @ w_ref_eci  # existing reference (often 0)

        # --- LOS-rate feedforward (for moving target directions) ---
        if self.los_rate_dt is not None:
            w_los = self._los_rate_body(R_b2i=R_b2i, s_eci=goal_vector_eci, dt=self.los_rate_dt)
            w_ref_body = w_ref_body + w_los

        w_err = w - w_ref_body

        # Desired torque (PD on direction + rate tracking)
        tau_des = -self.p_gain * q_err_vec - self.d_gain * w_err

        # --- Spin damping about boresight (prevents "spin around target") ---
        b_hat = self._unit(est_sat.boresight)
        w_spin = (b_hat @ w) * b_hat
        tau_des = tau_des - self.k_spin * w_spin

        # --- RW momentum reconstruction (robust to N_RW != 3) ---
        rw_axes = np.vstack([
            np.asarray(rw.axis, float).reshape(3,)
            for rw in est_sat.actuators
            if isinstance(rw, RW)
        ])  # (N_RW, 3)
        N_RW = rw_axes.shape[0]

        if x_hat.shape[0] < 7 + N_RW:
            raise ValueError(
                f"x_hat too short for RW momentum states: need >= {7+N_RW}, got {x_hat.shape[0]}"
            )

        h_vals = x_hat[7:7 + N_RW]      # (N_RW,)
        h_rw_body = h_vals @ rw_axes    # (3,)

        # Gyroscopic coupling compensation
        J = est_sat.J_0
        tau_gyro = np.cross(w, J @ w + h_rw_body)
        tau_des = tau_des + tau_gyro

        # Momentum dumping (body-frame target)
        tau_dump = -self.c_gain * (h_rw_body - self.h_target)

        # --- MTQ allocation: only torque ⟂ B is achievable ---
        B_skew = skewsym(b_body)
        M_mag_eff = -B_skew @ self.A_mtq  # (3, N_total_cmds)

        # Command MTQs to realize achievable part of (tau_des + tau_dump)
        tau_mtq_cmd = self._proj_perp_to_b(tau_des + tau_dump, b_body)

        # Full-length MTQ command vector
        u_mtq_full = np.linalg.pinv(M_mag_eff, rcond=self.pinv_rcond) @ tau_mtq_cmd  # (N_total_cmds,)
        u_mtq_full = limit(u=u_mtq_full, umax=self.max_u)

        # Actual MTQ torque realized
        tau_mtq_actual = M_mag_eff @ u_mtq_full

        # Residual to be produced by RWs
        tau_rw_cmd = tau_des - tau_mtq_actual

        # Allocate RW commands (full-length vector)
        u_rw_full = self.M_rw_act @ tau_rw_cmd
        u_rw_full = limit(u=u_rw_full, umax=self.max_u)

        # Total command vector
        u_total = u_mtq_full + u_rw_full
        return u_total
