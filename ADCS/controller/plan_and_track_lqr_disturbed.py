"""
Plan and Track LQR Controller with Disturbance Compensation.

This module implements a trajectory-following controller that uses the ALTRO
trajectory planner with findKwDist for integrated disturbance compensation.
The controller pulls disturbance torque estimates from the EstimatedSatellite's
disturbance models and uses them in the TVLQR feedback law.

The disturbance dynamics model assumes constant disturbance (d_dot = 0),
which is integrated into the LQR formulation via the C matrix in findKwDist.
This allows the controller to compensate for modeled disturbance torques
such as residual magnetic dipole, solar radiation pressure, gravity gradient,
and aerodynamic drag.
"""
from __future__ import annotations

__all__ = ["Plan_and_Track_LQR_Disturbed"]

import numpy as np
from typing import Optional
from numpy.typing import NDArray

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite


class Plan_and_Track_LQR_Disturbed(PlanAndTrackBase):
    """
    Trajectory-following controller using ALTRO planning and TVLQR with disturbance compensation.

    This controller uses findKwDist (tracking_LQR_formulation=2) to compute
    gains that include disturbance compensation. The disturbance torque is
    obtained from the EstimatedSatellite's disturbance models (e.g., SRP,
    drag, gravity gradient, residual dipole).

    The disturbance dynamics model assumes constant disturbance:
        d_dot = 0
    which is integrated into the LQR formulation via the C matrix.

    Attributes:
        est_sat: Estimated satellite model with disturbance models
        planner_settings: Configuration for the trajectory planner
        csat: C++ satellite model for the planner
        planner: C++ ALTRO planner instance
        active_trajectory: Currently active trajectory for tracking
    """

    def __init__(
        self,
        est_sat: EstimatedSatellite,
        planner_settings: PlannerSettings
    ) -> None:
        """
        Initialize the Plan and Track LQR controller with disturbance compensation.

        Args:
            est_sat: Estimated satellite model with actuators, sensors, and
                disturbance models. The disturbance torque will be computed
                from est_sat.dist_torques() at each control step.
            planner_settings: Configuration for the ALTRO trajectory planner
        """
        # tracking_lqr_formulation=2 is KwDist formulation with disturbance estimation
        self._init_planner(est_sat, planner_settings, tracking_lqr_formulation=2)

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
        Compute control using TVLQR tracking with disturbance compensation.

        The disturbance torque is computed from the EstimatedSatellite's
        disturbance models (SRP, drag, gravity gradient, residual dipole, etc.)
        and fed into the KwDist feedback law.

        Args:
            x_hat: Estimated state [omega(3), q(4), h(n_rw)]
            sens: Sensor measurements (unused in this controller)
            est_sat: Estimated satellite model with disturbance models
            os_hat: Current orbital state estimate
            goal_vector_eci: Goal direction in ECI (optional, unused)
            w_ref: Reference angular velocity (optional, unused)
            clip: If True, clip control to hardware actuator limits. Default True.

        Returns:
            Control vector (clipped to hardware limits if clip=True)
        """
        current_time = os_hat.J2000

        if self.active_trajectory is None:
            raise RuntimeError(f"Plan_and_Track_LQR_Disturbed: No active trajectory set at t={current_time}")

        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(
                f"Plan_and_Track_LQR_Disturbed: Active trajectory expired or not started. "
                f"Current: {current_time}, Traj: [{self.active_trajectory.start_time}, {self.active_trajectory.end_time}]"
            )

        # Get disturbance torque estimate from the satellite's disturbance models
        dist_torque = est_sat.dist_torques(x=x_hat, os=os_hat)

        # Update disturbance estimate in trajectory before computing control
        self.active_trajectory.update_disturbance_estimate(dist_torque)

        # Compute control with disturbance compensation
        u = self.active_trajectory.compute_tracking_control(current_time, x_hat)

        return self.clip_control(u, clip=clip)

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
        Calculate an optimal trajectory using ALTRO with KwDist gains.

        Args:
            t_start: Start time in J2000 centuries
            duration: Trajectory duration in seconds
            x_0: Initial state vector
            os_0: Initial orbital state
            goals: Goal list defining the pointing objectives
            verbose: If True, print debug information
            vecsPy_precomputed: Optional pre-computed environment vectors to skip slow orbit propagation
            N_precomputed: Number of timesteps (required with vecsPy_precomputed)
            t_end_precomputed: End time in J2000 centuries (required with vecsPy_precomputed)

        Returns:
            Trajectory object with KwDist gains (use_disturbance_estimation=True)
        """
        if verbose:
            print(f"Planning trajectory with KwDist: Start={t_start:.5f}, Dur={duration}s")

        lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
            t_start, duration, x_0, os_0, goals, verbose,
            vecsPy_precomputed, N_precomputed, t_end_precomputed
        )

        # Create trajectory with disturbance estimation enabled
        return Trajectory(
            lqr_times, Xset, Uset, Kset, Sset,
            use_disturbance_estimation=True
        )
