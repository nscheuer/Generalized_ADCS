__all__ = ["MTQ_w_RW_QPG"]

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import lsq_linear
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools

from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.controller import Controller, MTQ_w_RW_LP
from ADCS.controller.helpers.quaternion_math import vector_alignment_error
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat, skewsym, limit


class MTQ_w_RW_QPG(MTQ_w_RW_LP):
    r"""
    MTQ_w_RW_QPG
    ============

    Gyroscopically–Weighted Quadratic Torque Allocation
    ---------------------------------------------------

    This controller implements a **dynamic, state-dependent Weighted Least Squares (WLS)** allocation scheme.
    It extends the standard QP formulation by introducing a weighting matrix :math:`W(\boldsymbol{\omega})` that evolves with the spacecraft's angular velocity vector.

    By adjusting the weighting parameter :math:`\gamma`, the controller can prioritize torque accuracy along the instantaneous spin axis :math:`\hat{\boldsymbol{\omega}}` versus the transverse plane.

    [Image of spacecraft spin axis and torque components]

    Key Features:

    - **Dynamic Weighting:** The optimization cost function changes in real-time based on the body rate :math:`\boldsymbol{\omega}`.
    - **Spin-Axis Priority:** Allows the ADCS to penalize torque errors along the rotation axis more (or less) heavily than cross-axis errors.
    - **Tunable Gain:** The scalar :math:`\gamma` controls the strength of this anisotropic weighting.

    Weighting Formulation
    ---------------------

    Let the unit direction of the angular velocity be
    :math:`\hat{\boldsymbol{\omega}} = \boldsymbol{\omega} / \|\boldsymbol{\omega}\|`.

    The weighting matrix :math:`W` is constructed as a rank-1 update to the identity matrix:

    .. math::

        W(\boldsymbol{\omega}) = I_{3 \times 3} + \gamma \, (\hat{\boldsymbol{\omega}} \hat{\boldsymbol{\omega}}^T)

    This matrix scales errors parallel to :math:`\boldsymbol{\omega}` by a factor of :math:`(1+\gamma)` relative to perpendicular errors.

    Optimization Problem
    --------------------

    The controller solves the weighted bounded least squares problem:

    .. math::

        \min_{\boldsymbol{u}} \quad
        \left\| W(\boldsymbol{\omega}) \left( A_{\mathrm{tot}} \boldsymbol{u} - \boldsymbol{\tau}_{\mathrm{des}} \right) \right\|_2^2

    Subject to:

    .. math::

        -u_{i,\max} \le u_i \le u_{i,\max}

    Physical Interpretation
    ^^^^^^^^^^^^^^^^^^^^^^^

    - **If** :math:`\gamma > 0`: The controller "cares more" about matching the requested torque component along the spin axis. This is useful for spin-stabilized maneuvers where maintaining the spin rate is critical.
    - **If** :math:`\gamma = 0`: The controller reverts to the standard isotropic QP (minimizing Euclidean error).
    - **If** :math:`\gamma < 0`: The controller prioritizes transverse torque authority (e.g., for nutation damping).

    """
    def __init__(self, est_sat: EstimatedSatellite, p_gain: float, d_gain: float, gamma: float, c_gain: float, h_target: np.ndarray | list = np.zeros(3)) -> None:
        self.gamma = gamma
        super().__init__(est_sat=est_sat, p_gain=p_gain, d_gain=d_gain, c_gain=c_gain, h_target=h_target)

    def find_u(
        self,
        x_hat: np.ndarray,
        sens: np.ndarray,
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal: Goal | None = None,
    ) -> np.ndarray:
        if goal is None:
            goal = No_Goal()

        w = x_hat[0:3]
        q = x_hat[3:7]

        if isinstance(goal, No_Goal):
            k_w = self.d_gain
            k_h = self.c_gain

            # --- 1. Magnetic Field (Body Frame) ---
            b_body = np.asarray(self.M_mtm_read @ sens, float).reshape(3,)
            b_norm = np.linalg.norm(b_body)
            if b_norm < 1e-9:
                return np.zeros(self.n_actuators)
            b_hat = b_body / b_norm

            # --- 2. System Momentum Vector (Generic N-RW) ---
            # Calculate total vector momentum stored in all wheels
            if self.n_rw > 0 and len(x_hat) >= 7 + self.n_rw:
                h_rw_scalars = x_hat[7 : 7 + self.n_rw]
                h_sys = self.rw_axes @ h_rw_scalars # Matrix (3, N) @ Vec (N,) -> (3,)
            else:
                h_sys = np.zeros(3)

            # --- 3. Rate Damping (Perpendicular to B) ---
            # "Magnetic B-dot" logic: dampen rates only in the plane where MTQs can act
            w_perp = w - np.dot(w, b_hat) * b_hat
            tau_bdot_perp = -k_w * w_perp

            # --- 4. Momentum Dumping Torque (3D Vector) ---
            # Desired torque on body to reduce system momentum error
            h_err = h_sys - self.h_target  # Vector subtraction
            tau_dump_des = k_h * h_err     # Gain * Vector error

            # Saturation: Limit the dumping torque magnitude to avoid transient spikes
            mag = np.linalg.norm(tau_dump_des)
            limit_val = k_h * 0.001        # Example cap (adjust as needed)
            if mag > limit_val:
                tau_dump_des *= limit_val / (mag + 1e-12)

            # Project dump torque into MTQ-achievable plane (Perp to B)
            tau_dump_perp = tau_dump_des - np.dot(tau_dump_des, b_hat) * b_hat

            # Gating: Avoid dumping if the required torque is parallel to B (uncancellable)
            denom = np.linalg.norm(tau_dump_des) + 1e-12
            gamma = np.linalg.norm(tau_dump_perp) / denom
            tau_dump_cmd = gamma * tau_dump_des

            # --- 5. MTQ Allocation & Saturation Check ---
            tau_mtq_des = tau_bdot_perp - tau_dump_perp
            
            alpha_mtq = 0.0 # Default to 0: If no MTQs, we cannot dump.
            u_mtq_scaled = np.zeros(self.n_mtq) # Default empty/zero

            if self.n_mtq > 0:
                # Standard Allocation
                B_skew = skewsym(b_body)
                M_mag_eff = -B_skew @ self.A_mtq
                u_mtq_raw = np.linalg.pinv(M_mag_eff) @ tau_mtq_des

                # Check saturation
                mtq_cmds = u_mtq_raw[self.mtq_indices]
                alpha_mtq = 1.0
                if np.any(np.abs(mtq_cmds) > self.mtq_umax):
                    alpha_mtq = np.min(self.mtq_umax / (np.abs(mtq_cmds) + 1e-12))
                
                u_mtq_scaled = alpha_mtq * u_mtq_raw

            # --- 6. RW Command (Scaled) ---
            # If alpha_mtq is 0 (due to saturation or 0 MTQs), RW torque is reduced 
            # to 0 to prevent spinning up the body.
            tau_rw_req = alpha_mtq * (tau_dump_cmd + tau_bdot_perp)

            u_rw = self.M_rw_act @ tau_rw_req
            u_rw = np.clip(u_rw, -self.rw_umax, self.rw_umax)

            # --- Final Output ---
            u_out = u_mtq_scaled + u_rw     
        else:
        
            n_rw = len([a for a in est_sat.actuators if isinstance(a, RW)])
            if len(x_hat) >= 7 + n_rw:
                h_rw_states = x_hat[7 : 7 + n_rw]
            else:
                h_rw_states = np.array([rw.h for rw in est_sat.actuators if isinstance(rw, RW)])

            goal_vec_eci, w_ref_eci = goal.to_ref(os0=os_hat)
            R_b2i = rot_mat(q)
            w_ref_body = R_b2i.T @ w_ref_eci

            q_err = goal.error(q=q, body_boresight=est_sat.boresight, os0=os_hat)
            q_err = vector_alignment_error(
                q=q,
                eci_goal=goal_vec_eci,
                body_boresight=est_sat.boresight,
            )
            w_err = w - w_ref_body
            tau_pd = -self.p_gain * q_err - self.d_gain * w_err

            h_rw_body = np.zeros(3)
            rw_counter = 0
            for actuator in est_sat.actuators:
                if isinstance(actuator, RW):
                    h_rw_body += np.asarray(actuator.axis).flatten() * h_rw_states[rw_counter]
                    rw_counter += 1
            
            J = est_sat.J_0
            tau_gyro = np.cross(w, J @ w + h_rw_body)

            tau_des = tau_pd + tau_gyro
            
            b_body = np.asarray(self.M_mtm_read @ sens, float).reshape(3,)

            u_rw_cmd, u_mtq_cmd, alpha = self.allocate_max_torque_in_direction(
                tau_des, b_body, est_sat, w
            )

            u_out = np.zeros(len(est_sat.actuators))
            
            rw_indices = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, RW)]
            u_out[rw_indices] = u_rw_cmd
            
            mtq_indices = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, MTQ)]
            u_out[mtq_indices] = u_mtq_cmd

            # self.plot_torques(tau_des, b_body, est_sat)

        return u_out

    def allocate_max_torque_in_direction(self, tau_des: np.ndarray, b_body: np.ndarray, est_sat: EstimatedSatellite, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        tau_des = np.asarray(tau_des, float).reshape(3,)
        t_mag = np.linalg.norm(tau_des)
        if t_mag < 1e-9:
            n_rw = len([a for a in est_sat.actuators if isinstance(a, RW)])
            n_mtq = len([a for a in est_sat.actuators if isinstance(a, MTQ)])
            return np.zeros(n_rw), np.zeros(n_mtq), 1.0

        # 1) Setup matrices exactly like before
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]

        if rws:
            A_rw = np.column_stack([np.asarray(rw.axis, float).reshape(3,) for rw in rws])
            u_rw_lims = np.array([rw.u_max for rw in rws], dtype=float)
        else:
            A_rw = np.zeros((3, 0))
            u_rw_lims = np.zeros(0, dtype=float)

        if mtqs:
            b_skew = -skewsym(b_body)
            A_mtq_axes = np.column_stack([np.asarray(m.axis, float).reshape(3,) for m in mtqs])
            A_mtq = b_skew @ A_mtq_axes
            u_mtq_lims = np.array([m.u_max for m in mtqs], dtype=float)
        else:
            A_mtq = np.zeros((3, 0))
            u_mtq_lims = np.zeros(0, dtype=float)

        A_total = np.hstack([A_rw, A_mtq])  # (3, n_act)
        n_act = A_total.shape[1]

        n_rw = len(rws)
        n_mtq = len(mtqs)

        if n_act == 0:
            return np.zeros(n_rw), np.zeros(n_mtq), 0.0

        # 2) Bounds: [-u_max, +u_max]
        lb = np.concatenate([-u_rw_lims, -u_mtq_lims]) if n_act else np.zeros(0)
        ub = np.concatenate([ u_rw_lims,  u_mtq_lims]) if n_act else np.zeros(0)

        # -----------------------------
        # NEW: Weighting W(omega)
        # -----------------------------
        omega = np.asarray(omega, float).reshape(3,)
        om2 = float(np.dot(omega, omega))
        gamma = float(getattr(self, "gamma", 0.0))   # uses self.gamma; defaults to 0 if absent

        if om2 < 1e-12 or abs(gamma) < 1e-15:
            W = np.eye(3)
        else:
            W = np.eye(3) + gamma * (np.outer(omega, omega) / om2)

        # Weighted LSQ: minimize || W (A_total u - tau_des) ||^2
        A_w = W @ A_total
        b_w = W @ tau_des

        res = lsq_linear(A_w, b_w, bounds=(lb, ub), method="trf")

        if not res.success:
            return np.zeros(n_rw), np.zeros(n_mtq), 0.0

        u_sol = res.x

        # 4) Compute alpha as before
        tau_ach = A_total @ u_sol
        tau_hat = tau_des / (t_mag + 1e-12)
        T_along = float(np.dot(tau_ach, tau_hat))
        alpha = max(0.0, T_along / (t_mag + 1e-12))

        u_rw_cmd = u_sol[:n_rw]
        u_mtq_cmd = u_sol[n_rw:n_rw + n_mtq]

        return u_rw_cmd, u_mtq_cmd, alpha