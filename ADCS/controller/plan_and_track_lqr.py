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
    r"""
    Plan-and-Track controller using ALTRO planning with TVLQR feedback tracking.

    This controller plans an optimal attitude trajectory using the ALTRO
    (Augmented Lagrangian TRajectory Optimizer) and executes it in closed loop
    using a time-varying linear quadratic regulator (TVLQR). The TVLQR gains are
    computed along the nominal trajectory returned by the planner and are used
    to stabilize the spacecraft about that trajectory.

    Relationship to the Plan-and-Track framework
    ---------------------------------------------
    This class derives from
    :class:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase`, which
    provides:

    - Construction and configuration of the C++ ALTRO planner.
    - Orbit and environment propagation into planner-compatible arrays.
    - A shared trajectory-optimization pipeline.

    Mathematical formulation
    ------------------------
    Let the nonlinear spacecraft attitude dynamics be linearized about the
    nominal planned trajectory, yielding a discrete-time linear time-varying
    system:

    .. math::

       \mathbf{x}_{k+1} = A_k \mathbf{x}_k + B_k \mathbf{u}_k,

    where :math:`\mathbf{x}_k` is the attitude state deviation and
    :math:`\mathbf{u}_k` is the control input deviation at time step
    :math:`k`.

    The planner computes a nominal state-control sequence
    :math:`(\mathbf{x}_k^\ast, \mathbf{u}_k^\ast)` and associated TVLQR gains
    :math:`K_k`. The tracking control law applied by this controller is:

    .. math::

       \mathbf{u}_k =
       \mathbf{u}_k^\ast -
       K_k
       \left(
           \mathbf{x}_k - \mathbf{x}_k^\ast
       \right).

    This feedback stabilizes the system about the planned trajectory while
    compensating for moderate disturbances and modeling errors.

    Intended use
    ------------
    This controller is the standard closed-loop Plan-and-Track variant and is
    appropriate for nominal mission operations where feedback tracking is
    required but explicit disturbance estimation is not needed.

    :param est_sat: Estimated satellite model with actuators and sensors.
    :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
    :param planner_settings: ALTRO trajectory planner configuration bundle.
    :type planner_settings: :class:`~ADCS.controller.helpers.PlannerSettings`
    :return: None.
    :rtype: None

    """

    def __init__(self, est_sat: EstimatedSatellite, planner_settings: PlannerSettings) -> None:
        r"""
        Construct the Plan-and-Track LQR controller.

        This initializes the underlying C++ ALTRO planner using the standard TVLQR
        tracking formulation. The planner is configured through the provided
        :class:`~ADCS.controller.helpers.PlannerSettings`.

        No trajectory is generated during construction. A trajectory must be
        planned using :meth:`~Plan_and_Track_LQR.calculate_trajectory` and installed
        via :meth:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase.set_active_trajectory`
        before control commands can be generated.

        Planner configuration
        ---------------------
        - ``tracking_lqr_formulation = 0`` selects standard TVLQR gains.
        - The quaternion-to-vector mode defaults to the reduced representation
        defined by the base class.

        :param est_sat: Estimated satellite model with actuator and sensor models.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param planner_settings: ALTRO planner configuration settings.
        :type planner_settings: :class:`~ADCS.controller.helpers.PlannerSettings`
        :return: None.
        :rtype: None

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
        w_ref: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        r"""
        Compute the TVLQR tracking control input at the current time.

        This method evaluates the time-varying LQR feedback law associated with the
        active trajectory. The current time is obtained from the orbital state
        estimate and used to interpolate the nominal trajectory and TVLQR gains.

        Control evaluation
        ------------------
        Let :math:`t` be the current time extracted from ``os_hat.J2000``. If the
        active trajectory is valid at :math:`t`, the controller computes:

        .. math::

        \mathbf{u}(t) = \mathbf{u}^\ast(t) - K(t) \left(\mathbf{x}(t) - \mathbf{x}^\ast(t)\right),

        where:

        - :math:`\mathbf{x}(t)` is the estimated state ``x_hat``,
        - :math:`\mathbf{x}^\ast(t)` is the nominal state from the trajectory,
        - :math:`\mathbf{u}^\ast(t)` is the nominal control,
        - :math:`K(t)` is the time-varying LQR gain.

        Validity checks
        ---------------
        A runtime error is raised if:

        - No active trajectory has been set using :meth:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase.set_active_trajectory`.
        - The current time lies outside the valid time interval of the trajectory, as determined by :meth:`~ADCS.controller.helpers.Trajectory.is_valid_time`.

        Parameter usage
        ---------------
        The parameters ``sens``, ``est_sat``, ``goal_vector_eci``, and ``w_ref`` are
        included to satisfy the
        :class:`~ADCS.controller.Controller` interface but are not used directly in
        the control computation, since all references are taken from the active
        trajectory.

        :param x_hat: Estimated state vector.
        :type x_hat: numpy.typing.NDArray[numpy.float64]
        :param sens: Sensor measurement vector. Not directly used.
        :type sens: numpy.typing.NDArray[numpy.float64]
        :param est_sat: Estimated satellite model. Not directly used.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param os_hat: Estimated orbital state providing the current time.
        :type os_hat: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param goal_vector_eci: Goal vector in ECI frame. Not directly used.
        :type goal_vector_eci: typing.Optional[numpy.typing.NDArray[numpy.float64]]
        :param w_ref: Reference angular velocity. Not directly used.
        :type w_ref: typing.Optional[numpy.typing.NDArray[numpy.float64]]
        :return: Control vector computed by TVLQR tracking.
        :rtype: numpy.typing.NDArray[numpy.float64]

        """

        current_time = os_hat.J2000

        if self.active_trajectory is None:
            raise RuntimeError(f"Plan_and_Track_LQR: No active trajectory set at t={current_time}")

        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(f"Plan_and_Track_LQR: Active trajectory expired or not started. "
                                f"Current: {current_time}, Traj: [{self.active_trajectory.start_time}, {self.active_trajectory.end_time}]")

        return self.active_trajectory.compute_tracking_control(current_time, x_hat)

    def calculate_trajectory(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False
    ) -> Trajectory:
        r"""
        Plan an optimal trajectory using ALTRO and prepare it for TVLQR tracking.

        This method invokes the shared planning routine provided by
        :meth:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase._calculate_trajectory_common`
        to compute a nominal trajectory and associated TVLQR gains. The results are
        packaged into a :class:`~ADCS.controller.helpers.Trajectory` object suitable
        for closed-loop tracking.

        Planner outputs
        ---------------
        The returned trajectory contains:

        - A discrete time grid for planning and tracking.
        - The nominal state sequence :math:`\mathbf{x}_k^\ast`.
        - The nominal control sequence :math:`\mathbf{u}_k^\ast`.
        - The time-varying LQR gain sequence :math:`K_k`.
        - Auxiliary solver data required for tracking.

        These data are subsequently used by
        :meth:`~Plan_and_Track_LQR.find_u` to compute feedback control commands.

        :param t_start: Planning start time in J2000 centuries.
        :type t_start: float
        :param duration: Planning horizon length in seconds.
        :type duration: float
        :param x_0: Initial state vector used to seed the optimizer.
        :type x_0: numpy.ndarray
        :param os_0: Initial orbital state used to seed environment propagation.
        :type os_0: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param goals: Goal list defining the pointing objectives.
        :type goals: :class:`~ADCS.CONOPS.goallist.GoalList`
        :param verbose: If true, enable planner verbosity and diagnostic output.
        :type verbose: bool
        :return: Planned trajectory with nominal states, controls, and TVLQR gains.
        :rtype: :class:`~ADCS.controller.helpers.Trajectory`

        """
        lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
            t_start, duration, x_0, os_0, goals, verbose
        )
        return Trajectory(lqr_times, Xset, Uset, Kset, Sset)
