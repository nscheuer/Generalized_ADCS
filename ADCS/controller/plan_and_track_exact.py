from __future__ import annotations

__all__ = ["Plan_and_Track_Exact"]

import numpy as np

from ADCS.state import EstimatorState, State
from typing import Optional
from numpy.typing import NDArray

from ADCS.CONOPS.goals import Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.controller.helpers.trajectory import Trajectory
from ADCS.controller.plan_and_track import PlannerSettings
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite


class Plan_and_Track_Exact(PlanAndTrackBase):
    r"""
    Plan-and-Track Exact controller using ALTRO planning with open-loop execution.

    This controller plans an optimal attitude-control trajectory using the C++
    ALTRO planner and then executes the planned control sequence open-loop
    (no feedback tracking). It is primarily intended for:

    - Validating feasibility and quality of planned trajectories.
    - Testing planner settings and cost weights in isolation from tracking.
    - Characterizing actuator usage and constraint satisfaction.

    Relationship to other components
    --------------------------------
    This class derives from :class:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase`,
    which provides:

    - Planner construction via :meth:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase._init_planner`.
    - Environment propagation via :meth:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase._propagate_environment`.
    - Common trajectory-optimization logic via
      :meth:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase._calculate_trajectory_common`.

    Mathematical description
    ------------------------
    The planner returns a discrete-time nominal trajectory

    .. math::

       \{\mathbf{x}_k\}_{k=0}^{N-1}, \quad \{\mathbf{u}_k\}_{k=0}^{N-2},

    sampled at times :math:`t_k`. The exact controller applies the nominal
    input directly:

    .. math::

       \mathbf{u}(t_k) = \mathbf{u}_k,

    without using the TVLQR feedback gains :math:`K_k`. Any model mismatch,
    disturbances, or estimation errors therefore appear as open-loop tracking
    error, making this variant useful for assessing the robustness margin of
    the planned trajectory.

    Quaternion mode
    ---------------
    During initialization, this controller sets the planner to use the full
    quaternion representation (exact 4-parameter attitude):

    - ``quat_to_3vec_mode = 2``

    The tracking formulation is configured as standard TVLQR on the C++ side
    (even though this class does not apply feedback), since the planner may
    still compute linearizations and gain-related data as part of its output.

    :param est_sat: Estimated satellite model with actuators and sensors.
    :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
    :param planner_settings: ALTRO trajectory planner configuration bundle.
    :type planner_settings: :class:`~ADCS.controller.plan_and_track.PlannerSettings`
    :return: None.
    :rtype: None

    """

    def __init__(self, est_sat: EstimatedSatellite, planner_settings: PlannerSettings) -> None:
        r"""
        Construct the open-loop Plan-and-Track Exact controller.

        This initializes the underlying C++ planner stack using
        :meth:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase._init_planner`
        with an exact quaternion attitude representation. No active trajectory is
        created by this method; a trajectory must be computed with
        :meth:`~Plan_and_Track_Exact.calculate_trajectory` and then installed using
        :meth:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase.set_active_trajectory`
        before calling :meth:`~Plan_and_Track_Exact.find_u`.

        Planner configuration choices
        -----------------------------
        - Standard TVLQR formulation is selected for planner output consistency.
        - Exact quaternion mode is selected to avoid reduced-parameter attitude
        representations in planning.

        :param est_sat: Estimated satellite model with actuator and sensor models.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param planner_settings: ALTRO planner configuration settings.
        :type planner_settings: :class:`~ADCS.controller.plan_and_track.PlannerSettings`
        :return: None.
        :rtype: None

        """
        # tracking_lqr_formulation=0 is standard TVLQR
        # quat_to_3vec_mode=2 uses exact 4-parameter quaternion
        self._init_planner(est_sat, planner_settings, tracking_lqr_formulation=0, quat_to_3vec_mode=2)

    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal: Optional[Goal] = None,
    ) -> NDArray[np.float64]:
        r"""
        Return the planned open-loop control input at the current time.

        This method selects the control value from the active trajectory at the
        current orbital time and returns it directly. No feedback is applied.

        Time selection
        --------------
        The current time is read from the estimated orbital state as
        ``os_hat.J2000`` (J2000 centuries). Let this be :math:`t`. If the active
        trajectory is valid at :math:`t`, the controller returns:

        .. math::

        \mathbf{u}(t) = \mathbf{u}_{\mathrm{traj}}(t),

        where :math:`\mathbf{u}_{\mathrm{traj}}(t)` is obtained via
        :meth:`~ADCS.controller.helpers.Trajectory.get_control_at`.

        Validity conditions
        -------------------
        The method raises a runtime error if:

        - No active trajectory has been installed via :meth:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase.set_active_trajectory`.
        - The trajectory time window does not contain the current time, as tested by :meth:`~ADCS.controller.helpers.Trajectory.is_valid_time`.

        Parameter usage
        ---------------
        The arguments ``x_hat``, ``sens``, ``est_sat``, ``goal_vector_eci``, and
        ``w_ref`` are accepted to match the
        :class:`~ADCS.controller.Controller` interface but are not used for control
        computation in the open-loop variant.

        :param x_hat: Estimated state vector. Not used in open-loop execution.
        :type x_hat: numpy.typing.NDArray[numpy.float64]
        :param sens: Sensor measurement vector. Not used in open-loop execution.
        :type sens: numpy.typing.NDArray[numpy.float64]
        :param est_sat: Estimated satellite model. Not used in open-loop execution.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param os_hat: Estimated orbital state providing the current time.
        :type os_hat: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param goal_vector_eci: Goal vector in ECI frame. Not used in open-loop execution.
        :type goal_vector_eci: typing.Optional[numpy.typing.NDArray[numpy.float64]]
        :param w_ref: Reference angular velocity. Not used in open-loop execution.
        :type w_ref: typing.Optional[numpy.typing.NDArray[numpy.float64]]
        :return: Control vector from the active trajectory at the current time.
        :rtype: numpy.typing.NDArray[numpy.float64]

        """
        current_time = os_hat.J2000

        if self.active_trajectory is None:
            raise RuntimeError(f"Plan_and_Track_LQR: No active trajectory set at t={current_time}")

        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(f"Plan_and_Track_LQR: Active trajectory expired or not started. "
                                f"Current: {current_time}, Traj: [{self.active_trajectory.start_time}, {self.active_trajectory.end_time}]")

        return self.active_trajectory.get_control_at(current_time)

    def calculate_trajectory(
        self,
        t_start: float,
        duration: float,
        x_0: State,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False
    ) -> Trajectory:
        r"""
        Plan an optimal trajectory using ALTRO and package it as a Trajectory object.

        This method calls the common planning pipeline in
        :meth:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase._calculate_trajectory_common`
        to obtain the planner outputs, and then wraps them into a
        :class:`~ADCS.controller.helpers.Trajectory` instance.

        Outputs
        -------
        The returned trajectory contains:

        - ``lqr_times``: time grid for the nominal plan and tracking data.
        - ``Xset``: nominal state sequence.
        - ``Uset``: nominal control sequence (reordered into Python actuator order).
        - ``Kset``: TVLQR gains (reordered into Python actuator order), included for completeness even though this controller does not apply them.
        - ``Sset``: auxiliary tracking/planner data returned by the C++ backend.

        Mathematical interpretation
        ---------------------------
        Let the planner’s nominal sequences be :math:`\mathbf{x}_k` and
        :math:`\mathbf{u}_k` at times :math:`t_k`. The resulting trajectory object
        represents the piecewise sampling of these sequences and provides
        time-indexed access to the nominal control via
        :meth:`~ADCS.controller.helpers.Trajectory.get_control_at`.

        :param t_start: Planning start time in J2000 centuries.
        :type t_start: float
        :param duration: Planning horizon length in seconds.
        :type duration: float
        :param x_0: Initial state vector used to seed the optimizer.
        :type x_0: ADCS.state.State
        :param os_0: Initial orbital state used to seed environment propagation.
        :type os_0: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param goals: Goal list used to compute inertial reference vectors along the horizon.
        :type goals: :class:`~ADCS.CONOPS.goallist.GoalList`
        :param verbose: If true, enable planner verbosity and print vector diagnostics.
        :type verbose: bool
        :return: Planned trajectory container with times, states, controls, gains, and auxiliary data.
        :rtype: :class:`~ADCS.controller.helpers.Trajectory`

        """
        lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
            t_start, duration, x_0, os_0, goals, verbose
        )
        return Trajectory.from_arrays(lqr_times, Xset, Uset, Kset, Sset)
