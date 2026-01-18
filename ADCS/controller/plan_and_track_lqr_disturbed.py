"""
Plan and Track LQR Controller with Disturbance Estimation.

This module implements a trajectory-following controller that uses the ALTRO
trajectory planner with findKwDist for integrated disturbance estimation.
The controller maintains an internal disturbance estimate that is updated
each timestep and used in the TVLQR feedback law.

The disturbance dynamics model assumes constant disturbance (d_dot = 0),
which is integrated into the LQR formulation via the C matrix in findKwDist.
This allows the controller to adapt to unknown constant torque disturbances
such as residual magnetic dipole, solar radiation pressure, or unmodeled
reaction wheel friction.
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
    Trajectory-following controller using ALTRO planning and TVLQR with disturbance estimation.

    This controller uses findKwDist (tracking_LQR_formulation=2) to compute
    gains that include disturbance compensation. The augmented state includes
    a 3D disturbance torque estimate that is updated based on control error.

    The disturbance dynamics model assumes constant disturbance:
        d_dot = 0
    which is integrated into the LQR formulation via the C matrix.

    Attributes:
        est_sat: Estimated satellite model
        planner_settings: Configuration for the trajectory planner
        csat: C++ satellite model for the planner
        planner: C++ ALTRO planner instance
        active_trajectory: Currently active trajectory for tracking
        dist_estimate: Current disturbance torque estimate (3D vector)
        dist_gain: Gain for updating disturbance estimate from tracking error
    """

    dist_estimate: NDArray[np.float64]
    dist_gain: float

    def __init__(
        self,
        est_sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
        dist_gain: float = 0.1
    ) -> None:
        """
        Initialize the Plan and Track LQR controller with disturbance estimation.

        Args:
            est_sat: Estimated satellite model with actuators and sensors
            planner_settings: Configuration for the ALTRO trajectory planner
            dist_gain: Gain for updating disturbance estimate from tracking error.
                Higher values adapt faster but may be noisier. Default: 0.1
        """
        # tracking_lqr_formulation=2 is KwDist formulation with disturbance estimation
        self._init_planner(est_sat, planner_settings, tracking_lqr_formulation=2)
        self.dist_estimate = np.zeros(3)
        self.dist_gain = dist_gain

    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal_vector_eci: Optional[NDArray[np.float64]] = None,
        w_ref: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """
        Compute control using TVLQR tracking with disturbance compensation.

        Args:
            x_hat: Estimated state [omega(3), q(4), h(n_rw)]
            sens: Sensor measurements (unused in this controller)
            est_sat: Estimated satellite model
            os_hat: Current orbital state estimate
            goal_vector_eci: Goal direction in ECI (optional, unused)
            w_ref: Reference angular velocity (optional, unused)

        Returns:
            Control vector clipped to actuator limits
        """
        current_time = os_hat.J2000

        if self.active_trajectory is None:
            raise RuntimeError(f"Plan_and_Track_LQR_Disturbed: No active trajectory set at t={current_time}")

        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(
                f"Plan_and_Track_LQR_Disturbed: Active trajectory expired or not started. "
                f"Current: {current_time}, Traj: [{self.active_trajectory.start_time}, {self.active_trajectory.end_time}]"
            )

        # Update disturbance estimate in trajectory before computing control
        self.active_trajectory.update_disturbance_estimate(self.dist_estimate)

        # Compute control with disturbance compensation
        u = self.active_trajectory.compute_tracking_control(current_time, x_hat)

        # Update disturbance estimate based on tracking error
        # Simple integrator: dist += gain * (w_actual - w_expected)
        # This assumes the disturbance manifests as angular velocity error
        x_ref = self.active_trajectory.get_state_at(current_time)
        w_error = x_hat[0:3] - x_ref[0:3]
        self.dist_estimate += self.dist_gain * w_error * self.planner_settings.dt_tvlqr

        return np.clip(u, -self.planner_settings.umax, self.planner_settings.umax)

    def reset_disturbance_estimate(self) -> None:
        """Reset the disturbance estimate to zero."""
        self.dist_estimate = np.zeros(3)

    def calculate_trajectory(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False
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

        Returns:
            Trajectory object with KwDist gains (use_disturbance_estimation=True)
        """
        if verbose:
            print(f"Planning trajectory with KwDist: Start={t_start:.5f}, Dur={duration}s")

        lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
            t_start, duration, x_0, os_0, goals, verbose
        )

        # Create trajectory with disturbance estimation enabled
        return Trajectory(
            lqr_times, Xset, Uset, Kset, Sset,
            use_disturbance_estimation=True
        )
