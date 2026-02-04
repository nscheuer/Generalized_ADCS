"""
Plan-and-Track controller using computed torque control with actual B-field.

This controller tracks a planned trajectory by:
1. Computing desired torque from feedforward + PD feedback
2. Allocating torque to MTQs using actual B-field (not reference B-field)
3. Using RW for torque components parallel to B-field (MTQ null space)

This approach outperforms TVLQR because:
- TVLQR uses gains computed at reference B-field, which is wrong when off-trajectory
- Computed torque explicitly uses the actual B-field at each timestep
"""

from __future__ import annotations

__all__ = ["Plan_and_Track_ComputedTorque2"]

import numpy as np
from typing import Optional
from numpy.typing import NDArray

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.helpers.math_helpers import quat_diff, quat_to_vec3, rot_mat


class Plan_and_Track_ComputedTorque2(PlanAndTrackBase):
    """
    Plan-and-Track with computed torque control and actual B-field allocation.
    
    This controller computes desired torque from:
    - Feedforward: torque from reference trajectory
    - Feedback: PD control on attitude and angular velocity errors
    
    Then allocates to actuators using the ACTUAL B-field (not reference).
    
    Parameters
    ----------
    est_sat : EstimatedSatellite
        Satellite model with actuators.
    planner_settings : PlannerSettings
        Trajectory planner configuration.
    Kp : float, optional
        Proportional gain on attitude error (default 0.5).
    Kd : float, optional
        Derivative gain on angular velocity error (default 2.0).
        
    Notes
    -----
    The default gains (Kp=0.5, Kd=2.0) were tuned for 3MTQ+1RW CubeSats.
    Lower gains (Kp=0.1, Kd=1.0) also work well and are more conservative.
    
    The key insight is that gains should be MUCH lower than typical LQR gains
    because the feedforward from the trajectory is already doing most of the work.
    """
    
    def __init__(
        self,
        est_sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
        Kp: float = 0.5,
        Kd: float = 2.0,
    ):
        self._init_planner(est_sat, planner_settings, tracking_lqr_formulation=0)
        self.sat = est_sat
        self.Kp = Kp
        self.Kd = Kd
        
        self._J = est_sat.J_noRW
        self._n_mtq = len(est_sat.mtq_actuators)
        self._n_rw = len(est_sat.rw_actuators)
        self._m_max = est_sat.mtq_actuators[0].u_max if est_sat.mtq_actuators else 0.2
        self._rw_max = np.array([rw.u_max for rw in est_sat.rw_actuators]) if self._n_rw > 0 else np.array([])
        self._rw_axes = np.array([rw.axis for rw in est_sat.rw_actuators]) if self._n_rw > 0 else np.zeros((0, 3))
    
    def set_gains(self, Kp: float, Kd: float) -> None:
        """
        Set the PD feedback gains.
        
        Parameters
        ----------
        Kp : float
            Proportional gain on attitude error.
        Kd : float
            Derivative gain on angular velocity error.
        """
        self.Kp = Kp
        self.Kd = Kd
    
    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        **kwargs
    ) -> NDArray[np.float64]:
        """
        Compute control using computed torque with actual B-field.
        
        Parameters
        ----------
        x_hat : ndarray
            Estimated state [omega(3), q(4), h_rw(n_rw)].
        sens : ndarray
            Sensor measurements (unused).
        est_sat : EstimatedSatellite
            Satellite model (unused, uses stored reference).
        os_hat : Orbital_State
            Current orbital state with B-field.
            
        Returns
        -------
        ndarray
            Control vector [m_mtq(3), tau_rw(n_rw)].
        """
        if self.active_trajectory is None:
            return np.zeros(self._n_mtq + self._n_rw)
        
        t = os_hat.J2000
        if not self.active_trajectory.is_valid_time(t):
            return np.zeros(self._n_mtq + self._n_rw)
        
        # Current state
        w = x_hat[0:3]
        q = x_hat[3:7]
        
        # Reference state and control
        x_ref = self.active_trajectory.get_state_at(t)
        w_ref = x_ref[0:3]
        q_ref = x_ref[3:7]
        u_ref = self.active_trajectory.get_control_at(t)
        
        # Actual B-field in body frame
        R = rot_mat(q)
        B_body = R.T @ os_hat.B
        B_norm = np.linalg.norm(B_body)
        
        # Reference B-field (at reference attitude)
        R_ref = rot_mat(q_ref)
        B_body_ref = R_ref.T @ os_hat.B
        
        # Feedforward torque (what trajectory expects to produce)
        tau_ff = np.cross(u_ref[:self._n_mtq], B_body_ref)
        
        # Add RW feedforward torque
        if self._n_rw > 0:
            for i in range(self._n_rw):
                tau_ff += u_ref[self._n_mtq + i] * self._rw_axes[i]
        
        # Compute attitude error
        q_err = quat_diff(q_ref, q)
        q_err_vec = quat_to_vec3(q_err)
        
        # Angular velocity error  
        w_err = w - w_ref
        
        # PD feedback torque
        tau_fb = -self.Kp * self._J @ q_err_vec - self.Kd * self._J @ w_err
        
        # Total desired torque
        tau_des = tau_ff + tau_fb
        
        # Allocate to MTQ using actual B-field
        # tau = m x B -> m = (B x tau) / |B|^2 (minimum norm solution)
        if B_norm > 1e-12:
            m_mtq = np.cross(B_body, tau_des) / (B_norm**2)
            m_mtq = np.clip(m_mtq, -self._m_max, self._m_max)
        else:
            m_mtq = np.zeros(3)
        
        # RW handles residual torque (torque component MTQ can't produce)
        if self._n_rw > 0:
            # Torque that MTQ will actually produce
            tau_mtq = np.cross(m_mtq, B_body)
            
            # Residual torque needed
            tau_residual = tau_des - tau_mtq
            
            # Project onto each RW axis
            u_rw = np.zeros(self._n_rw)
            for i in range(self._n_rw):
                u_rw[i] = np.dot(tau_residual, self._rw_axes[i])
            
            u_rw = np.clip(u_rw, -self._rw_max, self._rw_max)
            return np.concatenate([m_mtq, u_rw])
        else:
            return m_mtq
    
    def calculate_trajectory(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False,
    ) -> Trajectory:
        """Calculate trajectory using C++ ALTRO planner."""
        lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
            t_start, duration, x_0, os_0, goals, verbose
        )
        return Trajectory(lqr_times, Xset, Uset, Kset, Sset)
