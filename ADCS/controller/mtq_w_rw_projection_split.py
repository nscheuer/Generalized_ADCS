__all__ = ["MTQ_w_RW_Projection_Split"]

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools

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
    Minimal MTQ + RW controller using a projection-split allocator.

    Kept features:
      - feed-forward omega tracking via Goal.to_ref()
      - feed-forward gyroscopic compensation
    
    New Feature:
      - 'allocate_max_torque_in_direction': Uses Linear Programming to find the 
        maximum possible torque strictly parallel to tau_des given actuator limits.
    """

    def __init__(
        self,
        est_sat: EstimatedSatellite,
        p_gain: float,
        d_gain: float,
        c_gain: float,
        h_target: np.ndarray,
        pinv_rcond: float = 1e-6,
    ) -> None:

        self.p_gain = float(p_gain)
        self.d_gain = float(d_gain)
        self.c_gain = float(c_gain)
        self.pinv_rcond = float(pinv_rcond)

        # MTM reconstruction
        self.M_mtm_read, _ = self.build_sensor_matrix_pinv(
            sensors=est_sat.sensors + est_sat.rw_actuators, sensor_type=MTM
        )

        # Torque → command maps
        self.M_rw_act, self.rw_indices = self.build_torque_to_u_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=RW
        )

        # Command → torque map for MTQs (full vector length)
        self.A_mtq = self.build_u_to_torque_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=MTQ
        )

        # RW geometry
        self.rw_axes = np.vstack([
            np.asarray(rw.axis, float).reshape(3,)
            for rw in est_sat.actuators if isinstance(rw, RW)
        ])

        self.h_target = np.asarray(h_target, dtype=float).reshape(3,)
        self.max_u = self.find_max_torque(actuators=est_sat.actuators)

    def find_u(
        self,
        x_hat: np.ndarray,
        sens: np.ndarray,
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal: Goal | None = None,
    ) -> np.ndarray:

        if goal is None:
            goal = Goal()

        # ---------- State Extraction ----------
        w = x_hat[0:3]
        q = x_hat[3:7]
        
        # Assumption: State vector x_hat contains [w(3), q(4), h_rw_1, h_rw_2, ...]
        # We extract RW momentum states starting at index 7.
        # Ideally, we verify the length matches the number of RWs.
        n_rw = len([a for a in est_sat.actuators if isinstance(a, RW)])
        if len(x_hat) >= 7 + n_rw:
            h_rw_states = x_hat[7 : 7 + n_rw]
        else:
            # Fallback if state vector is short (e.g. perfect attitude sensor only)
            # Use the stored values in the actuator objects (measurements)
            h_rw_states = np.array([rw.h for rw in est_sat.actuators if isinstance(rw, RW)])

        # ---------- Reference ----------
        goal_vec_eci, w_ref_eci = goal.to_ref(os0=os_hat)
        R_b2i = rot_mat(q)
        w_ref_body = R_b2i.T @ w_ref_eci

        # ---------- Errors ----------
        q_err = vector_alignment_error(
            q=q,
            eci_goal=goal_vec_eci,
            body_boresight=est_sat.boresight,
        )
        w_err = w - w_ref_body

        # ---------- 1. PD Control Law ----------
        tau_pd = -self.p_gain * q_err - self.d_gain * w_err

        # ---------- 2. Gyroscopic Compensation (Feed-Forward) ----------
        # Reconstruct total RW angular momentum vector in body frame
        # h_rw_body = sum( h_i * axis_i )
        h_rw_body = np.zeros(3)
        rw_counter = 0
        for actuator in est_sat.actuators:
            if isinstance(actuator, RW):
                h_rw_body += np.asarray(actuator.axis).flatten() * h_rw_states[rw_counter]
                rw_counter += 1
        
        # Compensation Torque: w x (J*w + h_rw)
        # This cancels the natural gyroscopic precession of the body
        J = est_sat.J_0
        tau_gyro = np.cross(w, J @ w + h_rw_body)

        # ---------- 3. Total Desired Torque ----------
        # The allocator now receives a request that includes the force needed 
        # to hold the satellite steady against precession.
        tau_des = tau_pd + tau_gyro
        
        b_body = np.asarray(self.M_mtm_read @ sens, float).reshape(3,)

        # ---------- 4. Projection Allocation (LP) ----------
        # Solves for optimal u to maximize torque in direction of tau_des
        u_rw_cmd, u_mtq_cmd, alpha = self.allocate_max_torque_in_direction(
            tau_des, b_body, est_sat
        )

        # ---------- Construct Output Vector ----------
        u_out = np.zeros(len(est_sat.actuators))
        
        rw_indices = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, RW)]
        u_out[rw_indices] = u_rw_cmd
        
        mtq_indices = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, MTQ)]
        u_out[mtq_indices] = u_mtq_cmd

        # Optional: visualizing usually blocks execution, so it is commented out for runtime
        # self.plot_torques(tau_des, b_body, est_sat)

        return u_out

    def allocate_max_torque_in_direction(self, tau_des: np.ndarray, b_body: np.ndarray, est_sat: EstimatedSatellite) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Uses Linear Programming to find the actuator commands (RW + MTQ) that 
        MAXIMIZE the torque magnitude strictly in the direction of tau_des, 
        subject to all actuator limits.
        
        Returns:
            u_rw: Torque commands for RWs [Nm]
            u_mtq: Dipole commands for MTQs [Am^2]
            alpha: Scaling factor (0.0 to >1.0) achieved relative to tau_des
        """
        
        t_mag = np.linalg.norm(tau_des)
        if t_mag < 1e-9:
            # Re-calculate indices just to return correct shaped zeros
            n_rw = len([a for a in est_sat.actuators if isinstance(a, RW)])
            n_mtq = len([a for a in est_sat.actuators if isinstance(a, MTQ)])
            return np.zeros(n_rw), np.zeros(n_mtq), 1.0

        # 1. Setup Matrices
        # -----------------------------------------------------
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]
        
        # Build RW Map
        if rws:
            A_rw = np.column_stack([rw.axis for rw in rws]) 
            u_rw_lims = np.array([rw.u_max for rw in rws])
        else:
            A_rw = np.zeros((3, 0))
            u_rw_lims = np.zeros(0)

        # Build MTQ Map
        if mtqs:
            b_skew = -skewsym(b_body)
            A_mtq_axes = np.column_stack([m.axis for m in mtqs])
            A_mtq = b_skew @ A_mtq_axes 
            u_mtq_lims = np.array([m.u_max for m in mtqs])
        else:
            A_mtq = np.zeros((3, 0))
            u_mtq_lims = np.zeros(0)

        A_total = np.hstack([A_rw, A_mtq])
        n_act = A_total.shape[1]

        # 2. Setup Normalized Linear Program
        # -----------------------------------------------------
        # Instead of A @ u = alpha * tau_des, we solve:
        # A @ u = T_available * tau_hat
        # This prevents the constraint column from vanishing when tau_des is small.
        
        tau_hat = tau_des / t_mag
        
        # Decision Variables x = [u_1, ..., u_n, T_available]
        # Maximize T_available => Minimize -T_available
        c = np.zeros(n_act + 1)
        c[-1] = -1.0 

        # A_eq @ x = 0  =>  [ A_total  |  -tau_hat ] @ [u; T_avail] = 0
        A_eq = np.hstack([A_total, -tau_hat.reshape(3,1)])
        b_eq = np.zeros(3)

        # Bounds
        bounds = []
        for lim_val in u_rw_lims: bounds.append((-lim_val, lim_val))
        for lim_val in u_mtq_lims: bounds.append((-lim_val, lim_val))
        
        # T_available bounds: [0, infinity] 
        # (We find the absolute max physical limit in this direction)
        bounds.append((0, None)) 

        # 3. Solve with HiGHS
        # -----------------------------------------------------
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        n_rw = len(rws)
        n_mtq = len(mtqs)

        if res.success:
            u_sol = res.x[:n_act]
            T_max_available = res.x[-1] # Max torque Nm achievable in this direction

            # 4. Scaling Logic
            # -------------------------------------------------
            # If we are strictly physically limited to less than we asked for:
            if T_max_available <= t_mag:
                # Saturated: Output max possible (u_sol is already at max)
                alpha = T_max_available / t_mag if t_mag > 0 else 0.0
                u_rw_cmd = u_sol[:n_rw]
                u_mtq_cmd = u_sol[n_rw:]
                return u_rw_cmd, u_mtq_cmd, alpha
            else:
                # Not Saturated: We have more capacity than requested.
                # Linearly scale down the commands to exactly match tau_des.
                scale_factor = t_mag / T_max_available
                u_scaled = u_sol * scale_factor
                
                u_rw_cmd = u_scaled[:n_rw]
                u_mtq_cmd = u_scaled[n_rw:]
                return u_rw_cmd, u_mtq_cmd, 1.0

        else:
            # Solver failed (usually singular geometry where NO torque is possible)
            return np.zeros(n_rw), np.zeros(n_mtq), 0.0

    def plot_torques(self, tau_des: np.ndarray, b_body: np.ndarray, est_sat: EstimatedSatellite) -> None:
        """
        Interactive visualization of the Allocation Strategy.
        - Scroll to Zoom
        - Shows RW Capacity Hull (Gray volume)
        - Shows MTQ Capacity Hull (Cyan plane)
        - Shows Vector Addition
        """
        # --- 1. Solve LP for Optimal Vectors ---
        u_rw_lp, u_mtq_lp, alpha_lp = self.allocate_max_torque_in_direction(tau_des, b_body, est_sat)

        # Reconstruct physical torque vectors
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        tau_rw_vec = np.zeros(3)
        for i, rw in enumerate(rws):
            tau_rw_vec += np.asarray(rw.axis) * u_rw_lp[i]

        mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]
        tau_mtq_vec = np.zeros(3)
        for i, m in enumerate(mtqs):
            tau_mtq_vec += np.cross(np.asarray(m.axis) * u_mtq_lp[i], b_body)

        tau_total = tau_rw_vec + tau_mtq_vec

        # --- 2. Setup Figure ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # KEY FIX: Force equal aspect ratio so vectors don't look distorted/inverted
        ax.set_box_aspect([1, 1, 1]) 

        # --- 3. Plot RW Capacity (Convex Hull of all wheels) ---
        # This shows the total volume of torque the RWs can produce
        if rws:
            # Get corners of hypercube: [-u_max, u_max] for each wheel
            rw_limits = [[-rw.u_max, rw.u_max] for rw in rws]
            rw_corners = list(itertools.product(*rw_limits))
            
            # Map command corners to physical torque points
            rw_points = []
            for corner in rw_corners:
                t_point = np.zeros(3)
                for i, val in enumerate(corner):
                    t_point += np.asarray(rws[i].axis) * val
                rw_points.append(t_point)
            
            self.plot_hull(ax, np.array(rw_points), color='gray', alpha=0.1, edge_color='black')

        # --- 4. Plot MTQ Capacity (Convex Hull in B-plane) ---
        if mtqs and np.linalg.norm(b_body) > 1e-9:
            mtq_limits = [[-m.u_max, m.u_max] for m in mtqs]
            mtq_corners = list(itertools.product(*mtq_limits))
            
            mtq_points = []
            for corner in mtq_corners:
                m_total = np.zeros(3)
                for i, val in enumerate(corner):
                    m_total += np.asarray(mtqs[i].axis) * val
                # Torque = m x B
                mtq_points.append(np.cross(m_total, b_body))
            
            # Plot faint plane (Cyan)
            self.plot_hull(ax, np.array(mtq_points), color='cyan', alpha=0.15, edge_color='teal')

        # --- 5. Plot Vectors ---
        # Helper for quivers to ensure they are anchored at origin
        def plot_vec(v, color, label, style='-'):
            if np.linalg.norm(v) > 1e-10:
                ax.quiver(0, 0, 0, v[0], v[1], v[2], 
                          color=color, arrow_length_ratio=0.1, linewidth=2.5, linestyle=style, label=label)

        # A. RW Vector (Purple)
        plot_vec(tau_rw_vec, 'purple', 'RW Allocation')
        
        # B. MTQ Vector (Royal Blue - High Contrast against Cyan plane)
        plot_vec(tau_mtq_vec, 'royalblue', 'MTQ Allocation')
        
        # C. Total Achieved (Lime Green)
        plot_vec(tau_total, 'limegreen', r'$\tau_{achieved}$')

        # D. Desired (Red Dashed)
        # Plot a longer line to show the "direction" we were aiming for
        if np.linalg.norm(tau_des) > 1e-10:
            scale = 1.0 if alpha_lp >= 1.0 else 1.5 # Draw it slightly longer if we saturated
            v_des = tau_des * scale
            ax.quiver(0, 0, 0, v_des[0], v_des[1], v_des[2], 
                      color='red', alpha=0.6, linestyle='--', linewidth=1.5, label=r'$\tau_{des}$')

        # --- 6. Visual Aids (Parallelogram summation) ---
        # Draw dotted lines connecting tips: Origin -> RW -> Total
        if np.linalg.norm(tau_rw_vec) > 1e-9 and np.linalg.norm(tau_mtq_vec) > 1e-9:
            # Line from RW tip to Total (represents adding MTQ vector)
            ax.plot([tau_rw_vec[0], tau_total[0]], 
                    [tau_rw_vec[1], tau_total[1]], 
                    [tau_rw_vec[2], tau_total[2]], color='royalblue', linestyle=':', alpha=0.5)
            # Line from MTQ tip to Total (represents adding RW vector)
            ax.plot([tau_mtq_vec[0], tau_total[0]], 
                    [tau_mtq_vec[1], tau_total[1]], 
                    [tau_mtq_vec[2], tau_total[2]], color='purple', linestyle=':', alpha=0.5)

        # --- 7. Formatting & Zoom Logic ---
        ax.set_xlabel('X [Nm]')
        ax.set_ylabel('Y [Nm]')
        ax.set_zlabel('Z [Nm]')
        ax.set_title(f"LP Allocation (Alpha: {alpha_lp:.2f})\nScroll to Zoom")
        ax.legend()

        # Initial Limits: Find max extent of data
        all_vecs = [tau_rw_vec, tau_mtq_vec, tau_total, tau_des]
        if rws: all_vecs.extend(rw_points) # Include RW hull in bounds
        
        max_val = 0
        for v in all_vecs:
            max_val = max(max_val, np.max(np.abs(v)))
        
        limit = max_val * 1.1 if max_val > 0 else 1.0
        
        # Set initial view
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)

        # --- Scroll to Zoom Handler ---
        def on_scroll(event):
            # Get current limits
            xlim = ax.get_xlim()
            curr_width = xlim[1] - xlim[0]
            
            # Zoom factor
            base_scale = 1.2
            if event.button == 'up': # Zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'down': # Zoom out
                scale_factor = base_scale
            else:
                scale_factor = 1.0

            new_width = curr_width * scale_factor
            limit = new_width / 2

            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_zlim(-limit, limit)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('scroll_event', on_scroll)
        plt.show()

    def plot_hull(self, ax, points: np.ndarray, color, alpha, edge_color):
        """Plot convex hull with ONLY outer boundary edges (no internal triangulation lines)."""
        if len(points) < 4:
            return

        try:
            hull = ConvexHull(points, qhull_options='QJ')

            # --- Plot faces WITHOUT edges ---
            faces = [points[s] for s in hull.simplices]
            poly = Poly3DCollection(
                faces,
                alpha=alpha,
                facecolors=color,
                edgecolors='none'   # ← disable triangulation edges
            )
            ax.add_collection3d(poly)

            # --- Extract unique boundary edges ---
            edge_count = {}

            for simplex in hull.simplices:
                edges = [
                    tuple(sorted((simplex[0], simplex[1]))),
                    tuple(sorted((simplex[1], simplex[2]))),
                    tuple(sorted((simplex[2], simplex[0])))
                ]
                for e in edges:
                    edge_count[e] = edge_count.get(e, 0) + 1

            # --- Draw only edges that appear once ---
            for (i, j), count in edge_count.items():
                if count == 1:  # boundary edge
                    p1, p2 = points[i], points[j]
                    ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        color=edge_color,
                        linewidth=0.8,
                        alpha=0.9
                    )

        except Exception:
            pass