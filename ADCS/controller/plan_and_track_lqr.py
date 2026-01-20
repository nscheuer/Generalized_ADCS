"""
Plan and Track LQR Controller for spacecraft attitude control.

This module implements a trajectory-following controller that uses the ALTRO
trajectory planner to compute optimal trajectories and TVLQR for tracking.
"""
from __future__ import annotations

__all__ = ["Plan_and_Track_LQR"]

import numpy as np
from typing import Optional
from numpy.typing import NDArray

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite


class Plan_and_Track_LQR(PlanAndTrackBase):
    """
    Trajectory-following controller using ALTRO planning and TVLQR tracking.

    This controller computes optimal trajectories using the ALTRO (Augmented
    Lagrangian TRajectory Optimizer) and tracks them using Time-Varying LQR
    feedback control.

    Attributes:
        est_sat: Estimated satellite model
        planner_settings: Configuration for the trajectory planner
        csat: C++ satellite model for the planner
        planner: C++ ALTRO planner instance
        active_trajectory: Currently active trajectory for tracking
        state_dim: Dimension of state vector
        ctrl_dim: Dimension of control vector
    """

    def __init__(self, est_sat: EstimatedSatellite, planner_settings: PlannerSettings) -> None:
        """
        Initialize the Plan and Track LQR controller.

        Args:
            est_sat: Estimated satellite model with actuators and sensors
            planner_settings: Configuration for the ALTRO trajectory planner
        """
        # tracking_lqr_formulation=0 is standard TVLQR
        self._init_planner(est_sat, planner_settings, tracking_lqr_formulation=0)

    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal_vector_eci: Optional[NDArray[np.float64]] = None,
        w_ref: Optional[NDArray[np.float64]] = None,
        clip: bool = True
    ) -> NDArray[np.float64]:
        """
        Compute control using TVLQR tracking.

        Args:
            x_hat: Estimated state vector
            sens: Sensor measurements (unused)
            est_sat: Estimated satellite model (unused)
            os_hat: Estimated orbital state (for time)
            goal_vector_eci: Goal vector in ECI (unused, from trajectory)
            w_ref: Reference angular velocity (unused, from trajectory)
            clip: If True, clip control to hardware actuator limits. Default True.

        Returns:
            Control vector (clipped to hardware limits if clip=True)
        """
        current_time = os_hat.J2000

        if self.active_trajectory is None:
            raise RuntimeError(f"Plan_and_Track_LQR: No active trajectory set at t={current_time}")

        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(f"Plan_and_Track_LQR: Active trajectory expired or not started. "
                                f"Current: {current_time}, Traj: [{self.active_trajectory.start_time}, {self.active_trajectory.end_time}]")

        ctrl = self.active_trajectory.compute_tracking_control(current_time, x_hat)
        return self.clip_control(ctrl, clip=clip)

    def calculate_trajectory(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False,
        vecsPy_precomputed: tuple = None,
        N_precomputed: int = None,
        t_end_precomputed: float = None
    ) -> Trajectory:
        """
        Calculate an optimal trajectory using ALTRO.

        Args:
            t_start: Start time in J2000 centuries
            duration: Duration in seconds
            x_0: Initial state vector
            os_0: Initial orbital state
            goals: Goal list for attitude reference
            verbose: Whether to print debug information
            vecsPy_precomputed: Optional pre-computed environment vectors to skip slow orbit propagation
            N_precomputed: Number of timesteps (required with vecsPy_precomputed)
            t_end_precomputed: End time in J2000 centuries (required with vecsPy_precomputed)

        Returns:
            Trajectory object with states, controls, and gains
        """
        lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
            t_start, duration, x_0, os_0, goals, verbose,
            vecsPy_precomputed, N_precomputed, t_end_precomputed
        )
        return Trajectory(lqr_times, Xset, Uset, Kset, Sset)
