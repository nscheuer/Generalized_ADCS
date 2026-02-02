"""
Plan-and-Track controller using TVLQR with actual B-field correction.

This controller fixes the fundamental TVLQR limitation for MTQ systems:
TVLQR computes MTQ commands based on the B-field at the reference attitude,
but when the actual attitude differs, the B-field in body frame is different.

Solution:
1. Compute desired torque from TVLQR feedback
2. Re-solve for MTQ dipole moment using actual B-field

This achieves sub-degree tracking for 3MTQ+1RW systems while standard
TVLQR oscillates with ~5-20° amplitude.
"""
from __future__ import annotations

__all__ = ["Plan_and_Track_ActualB", "Plan_and_Track_ActualB_Python"]

import numpy as np
from typing import Optional
from numpy.typing import NDArray

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.controller.helpers import (
    PlannerSettings, Trajectory, PythonALILQRv2,
    IterationData, LivePlannerViz
)
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.helpers.math_helpers import rot_mat


def _solve_mtq_for_torque(tau_desired: np.ndarray, B_body: np.ndarray, m_max: float) -> np.ndarray:
    """
    Solve for MTQ dipole moment to produce desired torque.
    
    Uses minimum-norm solution: m = (B × τ) / |B|²
    
    Note: This can only produce torque perpendicular to B. The component
    of tau_desired parallel to B cannot be achieved.
    """
    B_norm_sq = np.dot(B_body, B_body)
    if B_norm_sq < 1e-20:
        return np.zeros(3)
    m = np.cross(B_body, tau_desired) / B_norm_sq
    return np.clip(m, -m_max, m_max)


class Plan_and_Track_ActualB(PlanAndTrackBase):
    """
    Plan-and-Track with TVLQR planning and actual B-field MTQ correction.
    
    Uses C++ ALTRO for trajectory planning, then tracks using TVLQR gains
    but recomputes MTQ allocation using the actual B-field at each timestep.
    
    This fixes the B-field mismatch issue that causes TVLQR to oscillate
    for MTQ-based systems.
    """
    
    def __init__(
        self,
        sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
    ):
        self._init_planner(sat, planner_settings, tracking_lqr_formulation=0)
        self.sat = sat
        
        self._n_mtq = len(sat.mtq_actuators)
        self._m_max = sat.mtq_actuators[0].u_max if sat.mtq_actuators else 0.2
        
        self._n_rw = len(sat.rw_actuators)
        self._has_rw = self._n_rw > 0
        if self._has_rw:
            self._rw_u_max = np.array([rw.u_max for rw in sat.rw_actuators])
        else:
            self._rw_u_max = None
    
    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        B_body: Optional[NDArray[np.float64]] = None,
        **kwargs
    ) -> NDArray[np.float64]:
        """
        Compute control using TVLQR feedback with actual B-field MTQ allocation.
        
        Steps:
        1. Get TVLQR reference and feedback
        2. Compute reference torque and feedback torque (using reference B-field)
        3. Re-solve for MTQ dipole using actual B-field
        4. Use standard TVLQR for RW control
        """
        if self.active_trajectory is None:
            return np.zeros(self._n_mtq + self._n_rw)
        
        current_time = os_hat.J2000
        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(
                f"Trajectory expired. Current: {current_time}, "
                f"Traj: [{self.active_trajectory.start_time}, {self.active_trajectory.end_time}]"
            )
        
        # Get reference state and control
        x_ref = self.active_trajectory.get_state_at(current_time)
        u_ref = self.active_trajectory.get_control_at(current_time)
        K = self.active_trajectory.get_gain_at(current_time)
        
        # Compute state error using trajectory's method
        dx = self.active_trajectory._state_diff(x_hat, x_ref)
        
        # Compute feedback: du = -K @ dx
        du = -K @ dx
        
        # Get B-field in body frame (actual and reference)
        R_actual = rot_mat(x_hat[3:7])
        B_body_actual = R_actual.T @ os_hat.B
        
        R_ref = rot_mat(x_ref[3:7])
        B_body_ref = R_ref.T @ os_hat.B
        
        # Compute torques
        # Reference torque: τ_ref = m_ref × B_ref
        tau_ref = np.cross(u_ref[:self._n_mtq], B_body_ref)
        
        # Feedback torque: τ_fb = dm × B_ref (using reference B-field)
        tau_fb = np.cross(du[:self._n_mtq], B_body_ref)
        
        # Total desired torque
        tau_des = tau_ref + tau_fb
        
        # Solve for MTQ dipole using actual B-field
        m_actual = _solve_mtq_for_torque(tau_des, B_body_actual, self._m_max)
        
        # RW control: use standard TVLQR
        if self._has_rw:
            u_rw = np.clip(u_ref[self._n_mtq:] + du[self._n_mtq:], 
                          -self._rw_u_max, self._rw_u_max)
            return np.concatenate([m_actual, u_rw])
        else:
            return m_actual
    
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


class Plan_and_Track_ActualB_Python(PlanAndTrackBase):
    """
    Plan-and-Track with Python ALILQR planning and actual B-field correction.
    
    Same tracking as Plan_and_Track_ActualB but uses Python planner
    for live visualization of trajectory optimization.
    """
    
    def __init__(
        self,
        sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
    ):
        self._init_planner(sat, planner_settings, tracking_lqr_formulation=0)
        self.sat = sat
        
        self._n_mtq = len(sat.mtq_actuators)
        self._m_max = sat.mtq_actuators[0].u_max if sat.mtq_actuators else 0.2
        
        self._n_rw = len(sat.rw_actuators)
        self._has_rw = self._n_rw > 0
        if self._has_rw:
            self._rw_u_max = np.array([rw.u_max for rw in sat.rw_actuators])
        else:
            self._rw_u_max = None
    
    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        B_body: Optional[NDArray[np.float64]] = None,
        **kwargs
    ) -> NDArray[np.float64]:
        """Compute control (same as C++ version)."""
        if self.active_trajectory is None:
            return np.zeros(self._n_mtq + self._n_rw)
        
        current_time = os_hat.J2000
        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError("Trajectory expired")
        
        x_ref = self.active_trajectory.get_state_at(current_time)
        u_ref = self.active_trajectory.get_control_at(current_time)
        K = self.active_trajectory.get_gain_at(current_time)
        
        dx = self.active_trajectory._state_diff(x_hat, x_ref)
        du = -K @ dx
        
        R_actual = rot_mat(x_hat[3:7])
        B_body_actual = R_actual.T @ os_hat.B
        
        R_ref = rot_mat(x_ref[3:7])
        B_body_ref = R_ref.T @ os_hat.B
        
        tau_ref = np.cross(u_ref[:self._n_mtq], B_body_ref)
        tau_fb = np.cross(du[:self._n_mtq], B_body_ref)
        tau_des = tau_ref + tau_fb
        
        m_actual = _solve_mtq_for_torque(tau_des, B_body_actual, self._m_max)
        
        if self._has_rw:
            u_rw = np.clip(u_ref[self._n_mtq:] + du[self._n_mtq:],
                          -self._rw_u_max, self._rw_u_max)
            return np.concatenate([m_actual, u_rw])
        else:
            return m_actual
    
    def calculate_trajectory(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False,
        visualize: bool = False,
        viz_save_path: Optional[str] = None,
    ) -> Optional[Trajectory]:
        """Calculate trajectory using Python ALILQR with optional visualization."""
        py_planner = PythonALILQRv2(self.sat, os_0.orb, self.planner_settings)
        
        viz = None
        if visualize:
            viz = LivePlannerViz(self.sat, os_0.orb, goals, t_start, duration)
        
        def callback(data: IterationData):
            if viz:
                viz.update(data)
        
        # Two-pass optimization
        result1 = py_planner.optimize(
            x_0, t_start, duration, goals,
            pass_num=1, verbose=verbose,
            iteration_callback=callback if viz else None
        )
        
        if result1 is None or not result1.success:
            if verbose:
                print("Pass 1 failed")
            return None
        
        result2 = py_planner.optimize(
            x_0, t_start, duration, goals,
            pass_num=2, verbose=verbose,
            X_warm=result1.X, U_warm=result1.U,
            iteration_callback=callback if viz else None
        )
        
        result = result2 if (result2 and result2.success) else result1
        
        if viz and viz_save_path:
            viz.save(viz_save_path)
        
        N = result.U.shape[1]
        dt = self.planner_settings.dt_tvlqr
        times = t_start + np.arange(N + 1) * dt * TimeConstants.sec2cent
        
        return Trajectory(times, result.X, result.U, result.K, result.S)
