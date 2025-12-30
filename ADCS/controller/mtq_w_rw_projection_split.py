__all__ = ["MTQ_w_RW_Projection_Split"]

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
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

    Allocation:
      • MTQs realize achievable torque ⟂ B
      • RWs realize the residual torque
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

        # ---------- State ----------
        w = x_hat[0:3]
        q = x_hat[3:7]

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

        # ---------- Base torque ----------
        tau_des = -self.p_gain * q_err - self.d_gain * w_err
        b_body = np.asarray(self.M_mtm_read @ sens, float).reshape(3,)
        self._plot_torques(tau_des, b_body, est_sat)
        pass

    def _plot_torques(self, tau_des: np.ndarray, b_body: np.ndarray, est_sat: EstimatedSatellite) -> None:
        """
        Visualizes the desired torque vs. the physical actuator envelopes.
        """
        b_norm = np.linalg.norm(b_body)
        
        # separate actuators
        mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # ---------------------------------------------------------
        # 1. Calculate View Limits based ONLY on MTQ Capacity
        # ---------------------------------------------------------
        # We ignore tau_des and RW max torque for the zoom level.
        # This ensures the MTQ plane is always the "hero" of the plot.
        
        mtq_sum_max = sum([m.u_max for m in mtqs]) if mtqs else 0
        
        # If B is strong, max torque is high. If B is weak, max torque is low.
        # We add a small floor (1e-6) to prevent singular plots if B=0.
        current_mtq_max_torque = mtq_sum_max * b_norm
        limit_val = max(current_mtq_max_torque * 1.5, 1e-6)

        # ---------------------------------------------------------
        # 2. Plot Reaction Wheels (Visually Clamped)
        # ---------------------------------------------------------
        for i, rw in enumerate(rws):
            axis = np.asarray(rw.axis, dtype=float)
            
            # CRITICAL FIX: Do not draw lines to rw.u_max if it's huge.
            # Matplotlib 3D clipping will hide lines if endpoints are too far away.
            # Instead, draw the line just slightly larger than our view box.
            visual_extent = limit_val * 2.0 
            
            start = -axis * visual_extent
            end = axis * visual_extent
            
            # Plot the "infinite" axis line
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                    color='gray', linestyle=':', alpha=0.5, linewidth=1.0)
            
            # Add label at the edge of the view
            label_pos = axis * (limit_val * 0.9)
            ax.text(label_pos[0], label_pos[1], label_pos[2], f'RW{i}', fontsize=8, color='black')

            # Optional: If the true physical limit is actually within view, plot a dot
            if rw.u_max <= limit_val:
                true_tip = axis * rw.u_max
                ax.scatter([true_tip[0]], [true_tip[1]], [true_tip[2]], color='k', s=10)

        # ---------------------------------------------------------
        # 3. Plot MTQ Torque Envelope
        # ---------------------------------------------------------
        if b_norm > 1e-9 and len(mtqs) > 0:
            limits = [a.u_max for a in mtqs]
            ranges = [[-lim, lim] for lim in limits]
            dipole_corners = np.array(list(itertools.product(*ranges)))
            
            torque_points = []
            for corner_scalars in dipole_corners:
                m_total = np.zeros(3)
                for i, scalar in enumerate(corner_scalars):
                    m_total += scalar * np.asarray(mtqs[i].axis) 
                
                tau_corner = np.cross(m_total, b_body)
                torque_points.append(tau_corner)
            
            torque_points = np.array(torque_points)

            try:
                hull = ConvexHull(torque_points, qhull_options='QJ')
                verts = [torque_points[s] for s in hull.simplices]
                poly = Poly3DCollection(verts, alpha=0.4, facecolors='cyan', edgecolors='teal', linewidths=0.5)
                ax.add_collection3d(poly)
            except Exception:
                ax.scatter(torque_points[:,0], torque_points[:,1], torque_points[:,2], color='cyan', s=10)

        # ---------------------------------------------------------
        # 4. Plot B-Field Direction
        # ---------------------------------------------------------
        if b_norm > 1e-9:
            # Scale to 80% of view
            b_vis = (b_body / b_norm) * (limit_val * 0.8)
            ax.quiver(0, 0, 0, b_vis[0], b_vis[1], b_vis[2], 
                      color='blue', linestyle='--', arrow_length_ratio=0.1, 
                      label='B-field', linewidth=1.5)

        # ---------------------------------------------------------
        # 5. Plot Desired Torque
        # ---------------------------------------------------------
        if np.linalg.norm(tau_des) > 1e-9:
            # If tau_des is huge, it will just shoot off the plot.
            # This is desirable so we don't lose the zoom on the MTQ plane.
            ax.quiver(0, 0, 0, tau_des[0], tau_des[1], tau_des[2],
                      color='red', linewidth=2.5, arrow_length_ratio=0.1, label=r'$\tau_{des}$')

        # ---------------------------------------------------------
        # 6. Final Formatting
        # ---------------------------------------------------------
        ax.set_xlim([-limit_val, limit_val])
        ax.set_ylim([-limit_val, limit_val])
        ax.set_zlim([-limit_val, limit_val])

        ax.set_xlabel('X [Nm]')
        ax.set_ylabel('Y [Nm]')
        ax.set_zlabel('Z [Nm]')
        ax.set_title(f'MTQ Capability Plane (View Limit: {limit_val:.2e} Nm)')
        ax.legend()
        
        plt.show()