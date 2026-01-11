__all__ = ["MTQ_w_RW_QPW"]

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import lsq_linear
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools

from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.controller import Controller
from ADCS.controller.helpers.quaternion_math import vector_alignment_error
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat, skewsym, limit


class MTQ_w_RW_QPW(Controller):
    def __init__(
        self,
        est_sat: EstimatedSatellite,
        p_gain: float,
        d_gain: float,
        c_gain: float,
        h_target: float
    ) -> None:
        self.mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]
        self.rws  = [a for a in est_sat.actuators if isinstance(a, RW)]
        
        self.n_mtq = len(self.mtqs)
        self.n_rw = len(self.rws)
        
        self.p_gain = float(p_gain)
        self.d_gain = float(d_gain)
        self.c_gain = float(c_gain)
        self.h_target = np.asarray(h_target, dtype=float).reshape(3,)

        self.M_mtm_read, _ = self.build_sensor_matrix_pinv(
            sensors=est_sat.sensors + est_sat.rw_actuators, sensor_type=MTM
        )

        self.M_mtq_act, self.mtq_indices = self.build_torque_to_u_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=MTQ
        )
        self.M_rw_act, self.rw_indices = self.build_torque_to_u_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=RW
        )

        self.A_mtq = self.build_u_to_torque_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=MTQ
        )

        if self.n_rw > 0:
            self.rw_axes = np.column_stack([np.asarray(rw.axis, float).reshape(3,) for rw in self.rws])
        else:
            self.rw_axes = np.zeros((3, 0))

        # 8. Limits
        self.n_actuators = len(est_sat.actuators)
        self.mtq_umax = np.array([a.u_max for a in est_sat.actuators if isinstance(a, MTQ)], dtype=float)
        self.rw_umax  = np.array([a.u_max for a in est_sat.actuators if isinstance(a, RW)], dtype=float)

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
                tau_des, b_body, est_sat
            )

            u_out = np.zeros(len(est_sat.actuators))
            
            rw_indices = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, RW)]
            u_out[rw_indices] = u_rw_cmd
            
            mtq_indices = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, MTQ)]
            u_out[mtq_indices] = u_mtq_cmd

            # self.plot_torques(tau_des, b_body, est_sat)

        return u_out

    def allocate_max_torque_in_direction(self, tau_des: np.ndarray, b_body: np.ndarray, est_sat: EstimatedSatellite) -> tuple[np.ndarray, np.ndarray, float]:
        tau_des = np.asarray(tau_des, float).reshape(3,)
        t_mag = np.linalg.norm(tau_des)
        if t_mag < 1e-9:
            n_rw = len([a for a in est_sat.actuators if isinstance(a, RW)])
            n_mtq = len([a for a in est_sat.actuators if isinstance(a, MTQ)])
            return np.zeros(n_rw), np.zeros(n_mtq), 1.0

        # 1) Setup matrices exactly like before
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]

        # RW torque map: tau_rw = A_rw * u_rw
        if rws:
            A_rw = np.column_stack([np.asarray(rw.axis, float).reshape(3,) for rw in rws])
            u_rw_lims = np.array([rw.u_max for rw in rws], dtype=float)
        else:
            A_rw = np.zeros((3, 0))
            u_rw_lims = np.zeros(0, dtype=float)

        # MTQ torque map: tau_mtq = ( -skew(B) * A_axes ) * u_mtq
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

        tau_hat = tau_des / (t_mag + 1e-12)

        # Projection operators
        P_par  = np.outer(tau_hat, tau_hat)
        P_perp = np.eye(3) - P_par

        # Weights (tune w_perp >> w_par)
        w_par  = 1.0
        w_perp = 100.0

        # Weighting matrix (square root, since lsq minimizes ||Ax-b||^2)
        W = np.sqrt(w_par) * P_par + np.sqrt(w_perp) * P_perp

        A_w = W @ A_total
        b_w = W @ tau_des

        res = lsq_linear(A_w, b_w, bounds=(lb, ub), method="trf")

        if not res.success:
            # fall back: best effort (still keep shape)
            return np.zeros(n_rw), np.zeros(n_mtq), 0.0

        u_sol = res.x  # length n_act

        # 4) Compute alpha as "how much of tau_des magnitude you achieved along its direction"
        tau_ach = A_total @ u_sol
        tau_hat = tau_des / (t_mag + 1e-12)
        T_along = float(np.dot(tau_ach, tau_hat))          # signed magnitude along desired direction
        alpha = max(0.0, T_along / (t_mag + 1e-12))        # normalize by requested magnitude, clamp to [0, inf)

        # Split commands
        u_rw_cmd = u_sol[:n_rw]
        u_mtq_cmd = u_sol[n_rw:n_rw + n_mtq]

        return u_rw_cmd, u_mtq_cmd, alpha

    def plot_torques(self, tau_des: np.ndarray, b_body: np.ndarray, est_sat: EstimatedSatellite) -> None:
        # --- Solve allocation ---
        u_rw, u_mtq, alpha = self.allocate_max_torque_in_direction(tau_des, b_body, est_sat)

        rws  = [a for a in est_sat.actuators if isinstance(a, RW)]
        mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]

        tau_rw  = sum(np.asarray(rw.axis) * u_rw[i]  for i, rw in enumerate(rws))  if rws  else np.zeros(3)
        tau_mtq = sum(np.cross(np.asarray(m.axis) * u_mtq[i], b_body)
                    for i, m in enumerate(mtqs)) if mtqs else np.zeros(3)

        tau_tot = tau_rw + tau_mtq

        # --- Build capacity point clouds ---
        rw_pts, mtq_pts = [], []

        if rws:
            limits = [[-rw.u_max, rw.u_max] for rw in rws]
            rw_pts = np.array([
                sum(np.asarray(rws[i].axis) * c[i] for i in range(len(rws)))
                for c in itertools.product(*limits)
            ])

        if mtqs and np.linalg.norm(b_body) > 1e-9:
            limits = [[-m.u_max, m.u_max] for m in mtqs]
            mtq_pts = np.array([
                np.cross(sum(np.asarray(mtqs[i].axis) * c[i] for i in range(len(mtqs))), b_body)
                for c in itertools.product(*limits)
            ])

        # --- Figure ---
        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])

        # --- Hulls ---
        if len(rw_pts):
            self.plot_capacity(ax, rw_pts, face_color="gray", edge_color="black", alpha=0.25)

        if len(mtq_pts):
            self.plot_capacity(ax, mtq_pts, face_color="royalblue", edge_color="navy", alpha=0.25)

        # --- Vectors ---
        def vec(v, c, lbl, ls="-"):
            ax.plot([0, v[0]], [0, v[1]], [0, v[2]],
                    color=c, linewidth=2.5, linestyle=ls, label=lbl, clip_on=False)

        vec(tau_rw,  "purple",     "RW Allocation")
        vec(tau_mtq, "royalblue",  "MTQ Allocation")
        vec(tau_tot, "limegreen",  r"$\tau_{achieved}$")

        if np.linalg.norm(tau_des) > 1e-9:
            vec(tau_des, "red", r"$\tau_{des}$", ls="--")

        # --- Axis focus: MTQ hull ---
        focus_pts = mtq_pts if len(mtq_pts) else rw_pts
        if len(focus_pts):
            m = np.max(np.linalg.norm(focus_pts, axis=1))
            lim = m * 1.2
        else:
            lim = 1.0

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

        ax.set_xlabel("X [Nm]")
        ax.set_ylabel("Y [Nm]")
        ax.set_zlabel("Z [Nm]")
        ax.set_title(f"Torque Allocation (α = {alpha:.2f})\nScroll to zoom")
        ax.legend()

        # --- Scroll zoom ---
        def on_scroll(event):
            s = 0.85 if event.button == "up" else 1.15
            for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
                lo, hi = ax.get_xlim()
                span = (hi - lo) * s / 2
                setter(-span, span)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("scroll_event", on_scroll)
        plt.show()

    
    def plot_capacity(self, ax, points: np.ndarray,
                  face_color: str,
                  edge_color: str,
                  alpha: float = 0.25):

        pts = np.unique(points, axis=0)
        if len(pts) < 2:
            return

        center = pts.mean(axis=0)
        U, S, Vh = np.linalg.svd(pts - center)
        rank = np.sum(S > 1e-10)

        # --- 1D: Line ---
        if rank == 1:
            proj = (pts - center) @ Vh[0]
            p0, p1 = pts[np.argmin(proj)], pts[np.argmax(proj)]
            ax.plot(*zip(p0, p1),
                    color=edge_color,
                    linewidth=4.5,
                    alpha=alpha,
                    solid_capstyle="round",
                    clip_on=False)
            return

        # --- 2D: Filled polygon ---
        if rank == 2:
            proj = (pts - center) @ Vh[:2].T
            hull = ConvexHull(proj)
            loop = pts[hull.vertices]

            poly = Poly3DCollection([loop],
                                    facecolor=face_color,
                                    edgecolor=edge_color,
                                    linewidth=1.2,
                                    alpha=alpha)
            poly.set_clip_on(False)
            ax.add_collection3d(poly)
            return

        # --- 3D: Volume hull ---
        hull = ConvexHull(pts, qhull_options="QJ")

        faces = [pts[s] for s in hull.simplices]
        poly = Poly3DCollection(faces,
                                facecolor=face_color,
                                edgecolor="none",
                                alpha=alpha)
        poly.set_clip_on(False)
        ax.add_collection3d(poly)

        # --- silhouette edges only ---
        edge_count = {}
        for s in hull.simplices:
            for i, j in ((0,1),(1,2),(2,0)):
                e = tuple(sorted((s[i], s[j])))
                edge_count[e] = edge_count.get(e, 0) + 1

        for (i, j), c in edge_count.items():
            if c == 1:
                ax.plot(*zip(pts[i], pts[j]),
                        color=edge_color,
                        linewidth=1.0,
                        alpha=0.9,
                        clip_on=False)