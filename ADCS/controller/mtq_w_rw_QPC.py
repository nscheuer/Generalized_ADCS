__all__ = ["MTQ_w_RW_QPC"]

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.special import logsumexp
from scipy.optimize import lsq_linear, minimize, LinearConstraint, Bounds, linprog
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


class MTQ_w_RW_QPC(MTQ_w_RW_LP):
    r"""
    MTQ_w_RW_QPC
    ============

    Constrained Quadratic–Programming Torque Allocation
    ---------------------------------------------------

    This controller implements a **Constrained Least Squares (CLS)** allocation scheme.
    It extends the standard QP formulation by enforcing an additional **Lyapunov-based stability constraint** on the instantaneous power delivered to the system.

    In underactuated systems (like magnetic ADCS), a standard Least Squares allocator can sometimes produce a torque vector that minimizes the geometric error but has a component in the "wrong" direction relative to the spacecraft's rotation. This can inadvertently add kinetic energy to the system when the desired control law intended to remove it (damping).

    **This controller strictly forbids such destabilizing allocations.**

    

    Key Features:

    - **Stability Guarantee:** Enforces :math:`\dot{V}_{\mathrm{ach}} \le \dot{V}_{\mathrm{des}}` (roughly) to ensure the allocator never "fights" the high-level control law's energy intent.
    - **Optimization Objective:** Minimizes :math:`\|\boldsymbol{\tau}_{\mathrm{des}} - \boldsymbol{\tau}_{\mathrm{ach}}\|^2` subject to the stability constraint.
    - **Safe Degradation:** If the desired torque is physically impossible without violating stability (e.g., due to B-field alignment), the controller sacrifices tracking accuracy to maintain stability.

    Optimization Problem
    --------------------

    **Objective Function:**
    
    .. math::

        \min_{\boldsymbol{u}} \quad \frac{1}{2} \| A_{\mathrm{tot}} \boldsymbol{u} - \boldsymbol{\tau}_{\mathrm{des}} \|_2^2

    **Standard Constraints:**
    
    .. math::

        -u_{i,\max} \le u_i \le u_{i,\max}

    **Stability Constraint (The "Energy Gate"):**
    
    Let :math:`P = \boldsymbol{\omega} \cdot \boldsymbol{\tau}` be the mechanical power. The controller enforces:

    .. math::

        \boldsymbol{\omega}^T (A_{\mathrm{tot}} \boldsymbol{u}) \le \max(0, \boldsymbol{\omega}^T \boldsymbol{\tau}_{\mathrm{des}})

    - **Damping Case:** If the control law wants to spin down (:math:`\boldsymbol{\omega}^T \boldsymbol{\tau}_{\mathrm{des}} < 0`), the allocator is **forced** to produce a net torque that dissipates energy (or is at least neutral). It cannot produce a "best fit" torque that accidentally spins the satellite up.
    - **Slew Case:** If the control law wants to add energy (:math:`\boldsymbol{\omega}^T \boldsymbol{\tau}_{\mathrm{des}} > 0`), the constraint is relaxed, allowing the allocator to track the reference normally.

    Solver Implementation
    ^^^^^^^^^^^^^^^^^^^^^
    
    This problem is solved using `scipy.optimize.minimize(method='SLSQP')` to handle the linear inequality constraints involving the state vector :math:`\boldsymbol{\omega}`. 
    
    If the solver fails to find a feasible solution (rare), it gracefully falls back to the geometry-aware Linear Program (LP) solution to ensure a safe, scaled command is always available.
    """
    def __init__(self, est_sat: EstimatedSatellite, p_gain: float, d_gain: float, c_gain: float, h_target: np.ndarray | list = np.zeros(3)) -> None:
        super().__init__(est_sat=est_sat, p_gain=p_gain, d_gain=d_gain, c_gain=c_gain, h_target=h_target)

    def find_u(
        self,
        x_hat: np.ndarray,
        sens: np.ndarray,
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal: Goal | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if goal is None:
            goal = No_Goal()

        w = x_hat[0:3]
        q = x_hat[3:7]
        
        # Calculate total vector momentum stored in all wheels
        if self.n_rw > 0 and len(x_hat) >= 7 + self.n_rw:
            h_rw_scalars = x_hat[7 : 7 + self.n_rw]
            h_sys = self.rw_axes @ h_rw_scalars # Matrix (3, N) @ Vec (N,) -> (3,)
        else:
            h_sys = np.zeros(3)

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
            w_err = w
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
            if self.n_rw>0:
                rws = [a for a in est_sat.actuators if isinstance(a, RW)]
                h_err = h_rw_scalars - np.column_stack([np.asarray(rw.axis, float).reshape(3,) for rw in rws]).T@self.h_target  # Vector subtraction
            else:
                h_err = np.zeros((0,0))
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
                tau_des, b_body, est_sat, w_err, h_err
            )

            u_out = np.zeros(len(est_sat.actuators))
            
            rw_indices = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, RW)]
            u_out[rw_indices] = u_rw_cmd
            
            mtq_indices = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, MTQ)]
            u_out[mtq_indices] = u_mtq_cmd

            # self.plot_torques(tau_des, b_body, est_sat)

        return u_out, tau_des

    def allocate_max_torque_in_direction(self, tau_des: np.ndarray, b_body: np.ndarray, est_sat: EstimatedSatellite, omega: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
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
            J_rw = np.diagflat(np.vstack([rw.J for rw in rws]))
            if len(rws)>1:
                J_rw_inv = np.linalg.inv(J_rw)
            else:
                J_rw_inv = np.asarray(1/J_rw)
        else:
            A_rw = np.zeros((3, 0))
            u_rw_lims = np.zeros(0, dtype=float)
            J_rw = np.zeros((0,0))
            J_rw_inv = np.zeros((0,0))

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
        taudes_dot_omega = np.dot(omega,tau_des)
        
        SCALE = 1e9


        
        # # Step 1: Get LP baseline
        # tau_lp, lambda_star, u_lp = self.solve_lp_scaling(tau_des*SCALE, A_total, lb*SCALE, ub*SCALE)
        # energy_lp = np.dot(omega, tau_lp/SCALE)
        # # print(f"lambda*={lambda_star:.4f}, energy_lp={energy_lp:.6e}, ||tau_lp||={np.linalg.norm(tau_lp/SCALE):.6e}, ||tau_des||={np.linalg.norm(tau_des):.6e}")
        
            
        def fun(u):
            r = A_total @ u - tau_des*SCALE
            return 0.5 * np.dot(r,r)#/np.linalg.norm(tau_des)

        def jac(u):
            r = A_total @ u - tau_des*SCALE
            return A_total.T @ r#/np.linalg.norm(tau_des)
            
        threshold = 0.00001
        
        Irw = np.hstack([np.eye(n_rw),np.zeros((n_rw,n_mtq)),np.zeros((n_rw,n_act-n_mtq-n_rw))])
        
        Irw = Irw.reshape((n_rw,n_act))
        h = h.reshape((n_rw,1))
        J_rw_inv = J_rw_inv.reshape((n_rw,n_rw))
        C = omega @ A_total - (A_rw.T @ omega + J_rw_inv @ h).T @ Irw
        
        if False:#taudes_dot_omega > 0 :

        
            ub_constraint = taudes_dot_omega # logsumexp([0,taudes_dot_omega])#max(0,np.dot(omega,tau_des))
            
        
            lb_constraint = 0.0  # Don't brake
            # lin_ineq = LinearConstraint(omega@A_total, lb=lb_constraint*SCALE,ub=ub_constraint*SCALE)  # a @ u <= s

            # Reasonable starting point
            u0 = 0.5*(lb+ub)*SCALE
            # u0 = np.sign((np.linalg.pinv(A_total) @ tau_des*SCALE - 0.5*(lb+ub)*SCALE)/(0.5*(ub-lb)*SCALE))*(0.5*(ub-lb)*SCALE) + 0.5*(lb+ub)*SCALE
            
            bounds = Bounds(lb*SCALE, ub*SCALE)
            # C = omega @ A_total - (J_rw_inv @ h).T @ Irw
            # res = lsq_linear(A_c, b_c, bounds=(lb, ub), method="trf")
            
            # print(J_rw_inv,h)
            res = minimize(
                fun, u0, jac=jac,
                method="SLSQP",
                constraints=[
                                {
                                    "type": "ineq",
                                    "fun": lambda u, C=C,ub_constraint=ub_constraint,SCALE=SCALE: ub_constraint*SCALE - (C@ u),
                                    "jac": lambda u, C=C: -C
                                }
                                # {
                                    # "type": "ineq",
                                    # "fun": lambda u: ub_constraint*SCALE - (omega @ A_total @ u),
                                    # "jac": lambda u: -omega @ A_total
                                # }
                                # ,
                                # {
                                    # "type": "ineq",
                                    # "fun": lambda u,C=C,lb_constraint=lb_constraint,SCALE=SCALE: (C @ u) - lb_constraint*SCALE,
                                    # "jac": lambda u,C=C: C
                                # }
                            ],
                bounds=bounds,
                options={}
            )
        

            u_sol = res.x/SCALE
            tau_qp = A_total @ u_sol
            energy_qp = np.dot(omega, tau_qp)
            # print(u0/SCALE)
            # print(u_sol)

            # print(f"energy_qp={energy_qp:.6e}, energy_des={taudes_dot_omega:.6e},||tau_qp||={np.linalg.norm(tau_qp):.6e}, ||tau_des||={np.linalg.norm(tau_des):.6e}")
            # print(f"QP better on energy: {energy_qp <= energy_lp + 1e-9}")
            # print(f"QP better on tracking: {np.linalg.norm(tau_qp - tau_des) <= np.linalg.norm(tau_lp/SCALE - tau_des) + 1e-9}")
        
        else:
            
            ub_constraint = max(0,taudes_dot_omega)#0.0 #don't accelerate
        
            lb_constraint = min(taudes_dot_omega, 0)#taudes_dot_omega # Don't brake more than expected
            # lin_ineq = LinearConstraint(omega@A_total,ub=ub_constraint)  # a @ u <= s
            
            # def fun(u):
                # r = A_total @ u - tau_des
                # return 0.5 * np.dot(r,r)#/np.linalg.norm(tau_des)

            # def jac(u):
                # r = A_total @ u - tau_des
                # return A_total.T @ r#/np.linalg.norm(tau_des)

            # # Reasonable starting point
            # u0 = 0.5*(lb+ub)
            
            bounds = Bounds(lb*SCALE, ub*SCALE)
            Irw = np.hstack([np.eye(n_rw),np.zeros((n_rw,n_mtq)),np.zeros((n_rw,n_act-n_mtq-n_rw))])
            
            Irw = Irw.reshape((n_rw,n_act))
            h = h.reshape((n_rw,1))
            J_rw_inv = J_rw_inv.reshape((n_rw,n_rw))
            C = omega @ A_total - (A_rw.T @ omega + J_rw_inv @ h).T @ Irw
            # C = omega @ A_total - (J_rw_inv @ h).T @ Irw


            
            u0 = 0.5*(lb+ub)*SCALE
            
            res = minimize(
                fun, u0, jac=jac,
                method="SLSQP",
                constraints=[
                            {"type": "ineq", "fun": lambda u: ub_constraint*SCALE - (omega @ A_total @ u), "jac": lambda u: -omega @ A_total},
                             # {"type": "ineq", "fun": lambda u: -lb_constraint*SCALE + (omega @ A_total @ u), "jac": lambda u: omega @ A_total}
                             ],
                # constraints={"type": "ineq", "fun": lambda u: ub_constraint*SCALE - (C @ u), "jac": lambda u: -C},
                bounds=bounds,
                options={}
            )
        
            # res = lsq_linear(A_total, tau_des*SCALE, bounds=(lb*SCALE, ub*SCALE), method="trf")
            
            u_sol = res.x/SCALE
            tau_qp = A_total @ u_sol
            energy_qp = np.dot(omega, tau_qp)

            # print(f"energy_qp={energy_qp:.6e}, energy_des={taudes_dot_omega:.6e},||tau_qp||={np.linalg.norm(tau_qp):.6e}, ||tau_des||={np.linalg.norm(tau_des):.6e}")
            # print(f"QP better on energy: {energy_qp <= energy_lp + 1e-9}")
            # print(f"QP better on tracking: {np.linalg.norm(tau_qp - tau_des) <= np.linalg.norm(tau_lp - tau_des) + 1e-9}")
        
            
        u_sol = res.x/SCALE
        
        
        res_uncon = lsq_linear(A_total, tau_des*SCALE, bounds=(lb*SCALE, ub*SCALE), method="trf")
            
        u_sol_uncon = res_uncon.x/SCALE
        tau_uncon = A_total @ u_sol_uncon
        unconstrained_energy = np.dot(omega, tau_uncon)
        constrained_energy = omega @ A_total @ u_sol

        if unconstrained_energy > 1e-9 and constrained_energy < unconstrained_energy - 1e-9:
            print(f"Constraint active: wanted {unconstrained_energy:.2e}, got {constrained_energy:.2e}")
                    
        if not res.success:
            print(f"QP failed: {res.message}")
            _, alpha, u_sol_SCALED = self.solve_lp_scaling(tau_des*SCALE, A_total, lb*SCALE, ub*SCALE)
            u_sol = u_sol_SCALED/SCALE
            
        # 4) Compute alpha as before
        
      
        tau_ach = A_total @ u_sol
        tau_hat = tau_des / (t_mag + 1e-12)
        T_along = float(np.dot(tau_ach, tau_hat))
        alpha = max(0.0, T_along / (t_mag + 1e-12))
        u_rw_cmd = u_sol[:n_rw]
        u_mtq_cmd = u_sol[n_rw:n_rw + n_mtq]
        return u_rw_cmd, u_mtq_cmd, alpha
        
        
    def solve_lp_scaling(self, tau_des, A_total, lb, ub):
        """
        Find max λ such that λ*τ_des is achievable.
        Returns (τ_lp, λ_star, u_lp)
        """
        t_mag = np.linalg.norm(tau_des)
        if t_mag < 1e-12:
            return np.zeros(3), 1.0, np.zeros(len(lb))
        
        tau_hat = tau_des / t_mag
        n_act = len(lb)
        
        # Variables: [u, T_available]
        # Maximize T_available (minimize -T_available)
        c = np.zeros(n_act + 1)
        c[-1] = -1.0
        
        # Constraint: A_total @ u = T_available * tau_hat
        A_eq = np.hstack([A_total, -tau_hat.reshape(3, 1)])
        b_eq = np.zeros(3)
        
        # Bounds: lb ≤ u ≤ ub, 0 ≤ T_available
        bounds = [(lb[i], ub[i]) for i in range(n_act)] + [(0, None)]
        
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            u_lp = res.x[:n_act]
            T_available = res.x[-1]
            
            # Scale down if we have more capacity than needed
            if T_available > t_mag:
                scale = t_mag / T_available
                u_lp = u_lp * scale
                lambda_star = 1.0
            else:
                lambda_star = T_available / t_mag
            
            tau_lp = A_total @ u_lp
            return tau_lp, lambda_star, u_lp
        else:
            
            print(f"LP failed: {res.message}")
            return np.zeros(3), 0.0, np.zeros(n_act)
   