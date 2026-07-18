from __future__ import annotations

__all__ = ["Plan_and_Track_LQR_Disturbed"]

import numpy as np

from ADCS.state import EstimatedState, State
from typing import Optional
from numpy.typing import NDArray

from ADCS.CONOPS.goals import Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.controller.helpers.trajectory import Trajectory
from ADCS.controller.plan_and_track import PlannerSettings
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite


class Plan_and_Track_LQR_Disturbed(PlanAndTrackBase):
    r"""
    Plan-and-Track TVLQR controller with integrated disturbance estimation.

    This controller extends the standard Plan-and-Track framework by augmenting
    the tracking controller with an explicit disturbance estimate. Trajectories
    are planned using the ALTRO planner with the ``findKwDist`` formulation,
    which produces time-varying LQR gains compatible with an augmented-state
    system that includes a constant disturbance model.

    The controller then executes the trajectory in closed loop using TVLQR
    feedback while continuously updating an internal estimate of the unknown
    disturbance torque.

    Relationship to the base framework
    ----------------------------------
    This class derives from
    :class:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase` and therefore
    reuses:

    - Planner construction via
      :meth:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase._init_planner`.
    - Environment propagation via
      :meth:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase._propagate_environment`.
    - Common trajectory optimization via
      :meth:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase._calculate_trajectory_common`.

    Disturbance-augmented dynamics
    ------------------------------
    Let the nominal discrete-time attitude dynamics (after linearization about
    the planned trajectory) be

    .. math::

       \mathbf{x}_{k+1} = A_k \mathbf{x}_k + B_k \mathbf{u}_k,

    where :math:`\mathbf{x}` is the attitude state and :math:`\mathbf{u}` is the
    control input. A constant disturbance torque
    :math:`\mathbf{d} \in \mathbb{R}^3` enters additively through a disturbance
    matrix :math:`C_k`:

    .. math::

       \mathbf{x}_{k+1} = A_k \mathbf{x}_k + B_k \mathbf{u}_k + C_k \mathbf{d}.

    The disturbance is modeled as constant:

    .. math::

       \dot{\mathbf{d}} = \mathbf{0}.

    Defining the augmented state

    .. math::

       \tilde{\mathbf{x}}_k =
       \begin{bmatrix}
           \mathbf{x}_k \\
           \mathbf{d}_k
       \end{bmatrix},

    the augmented dynamics become

    .. math::

       \tilde{\mathbf{x}}_{k+1} =
       \begin{bmatrix}
           A_k & C_k \\
           0   & I
       \end{bmatrix}
       \tilde{\mathbf{x}}_k +
       \begin{bmatrix}
           B_k \\
           0
       \end{bmatrix}
       \mathbf{u}_k.

    The ``findKwDist`` formulation used by the planner computes TVLQR gains
    :math:`K_k` for this augmented system, enabling feedback laws of the form

    .. math::

       \mathbf{u}_k =
       \mathbf{u}_k^\ast -
       K_k
       \left(
           \tilde{\mathbf{x}}_k -
           \tilde{\mathbf{x}}_k^\ast
       \right),

    where :math:`(\cdot)^\ast` denotes the planned nominal trajectory.

    Online disturbance adaptation
    ------------------------------
    In addition to the planner-provided augmented gains, this controller
    maintains a simple integrator-based disturbance estimate
    :math:`\hat{\mathbf{d}}`. At each control step, the estimate is updated using
    the angular velocity tracking error:

    .. math::

       \hat{\mathbf{d}}_{k+1} =
       \hat{\mathbf{d}}_k +
       \gamma
       \left(
           \boldsymbol{\omega}_k -
           \boldsymbol{\omega}_k^\ast
       \right)
       \Delta t,

    where:

    - :math:`\boldsymbol{\omega}_k` is the measured angular velocity,
    - :math:`\boldsymbol{\omega}_k^\ast` is the nominal angular velocity from
      the trajectory,
    - :math:`\gamma` is a user-selected disturbance adaptation gain,
    - :math:`\Delta t` is the TVLQR discretization step.

    This estimate is injected into the active trajectory prior to computing
    the tracking control.

    Intended use
    ------------
    This controller is suited for scenarios with approximately constant or
    slowly varying external torques, such as:

    - Residual magnetic dipole moments.
    - Solar radiation pressure biases.
    - Unmodeled reaction wheel friction.

    :param est_sat: Estimated satellite model with actuators and sensors.
    :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
    :param planner_settings: ALTRO trajectory planner configuration bundle.
    :type planner_settings: :class:`~ADCS.controller.plan_and_track.PlannerSettings`
    :param dist_gain: Scalar gain controlling the disturbance adaptation rate.
    :type dist_gain: float
    :return: None.
    :rtype: None

    """

    dist_estimate: NDArray[np.float64]
    dist_gain: float

    def __init__(
        self,
        est_sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
        dist_gain: float = 0.1
    ) -> None:
        r"""
        Construct the Plan-and-Track LQR controller with disturbance estimation.

        This initializes the underlying C++ planner using the ``findKwDist``
        tracking formulation and sets the initial disturbance estimate to zero.

        Planner configuration
        ---------------------
        The planner is initialized with:

        - ``tracking_lqr_formulation = 2`` to enable disturbance-augmented TVLQR.
        - Default quaternion-to-vector settings inherited from the base class.

        Disturbance adaptation
        ----------------------
        The internal disturbance estimate :math:`\hat{\mathbf{d}}` is initialized
        as

        .. math::

        \hat{\mathbf{d}}_0 = \mathbf{0},

        and subsequently updated during calls to
        :meth:`~Plan_and_Track_LQR_Disturbed.find_u`.

        :param est_sat: Estimated satellite model with actuator and sensor models.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param planner_settings: ALTRO planner configuration settings.
        :type planner_settings: :class:`~ADCS.controller.plan_and_track.PlannerSettings`
        :param dist_gain: Gain for updating the disturbance estimate from tracking
            error.
        :type dist_gain: float
        :return: None.
        :rtype: None

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
        goal: Optional[Goal] = None,
    ) -> NDArray[np.float64]:
        r"""
        Compute TVLQR tracking control with disturbance compensation.

        This method applies time-varying LQR feedback around the active trajectory
        while compensating for an estimated constant disturbance torque.

        Control law
        -----------
        Let :math:`t` be the current time extracted from ``os_hat.J2000``. The
        controller computes the nominal tracking control using the active
        trajectory:

        .. math::

        \mathbf{u}(t) = \mathbf{u}^\ast(t) - K(t) \left(\mathbf{x}(t) - \mathbf{x}^\ast(t)\right) - \hat{\mathbf{d}}(t),

        where:

        - :math:`\mathbf{u}^\ast(t)` is the nominal control,
        - :math:`K(t)` is the TVLQR gain,
        - :math:`\hat{\mathbf{d}}(t)` is the current disturbance estimate.

        Disturbance update
        ------------------
        After computing the control, the disturbance estimate is updated using the
        angular velocity error according to

        .. math::

        \hat{\mathbf{d}} \leftarrow \hat{\mathbf{d}} + \gamma \left(\boldsymbol{\omega} - \boldsymbol{\omega}^\ast \right) \Delta t.

        Actuator limits
        ---------------
        The resulting control vector is clipped elementwise to the actuator limits
        specified by ``planner_settings.umax``.

        Validity checks
        ---------------
        A runtime error is raised if no active trajectory is set or if the current
        time lies outside the trajectory’s valid time window, as determined by
        :meth:`~ADCS.controller.helpers.Trajectory.is_valid_time`.

        :param x_hat: Estimated state vector.
        :type x_hat: numpy.typing.NDArray[numpy.float64]
        :param sens: Sensor measurement vector. Not directly used.
        :type sens: numpy.typing.NDArray[numpy.float64]
        :param est_sat: Estimated satellite model.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param os_hat: Estimated orbital state providing the current time.
        :type os_hat: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param goal_vector_eci: Goal direction in ECI frame. Not used.
        :type goal_vector_eci: typing.Optional[numpy.typing.NDArray[numpy.float64]]
        :param w_ref: Reference angular velocity. Not used.
        :type w_ref: typing.Optional[numpy.typing.NDArray[numpy.float64]]
        :return: Control vector after TVLQR feedback and disturbance compensation.
        :rtype: numpy.typing.NDArray[numpy.float64]

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
        w_error = x_hat.w - x_ref.w
        self.dist_estimate += self.dist_gain * w_error * self.planner_settings.dt_tvlqr

        return np.clip(u, -self.planner_settings.umax, self.planner_settings.umax)

    def reset_disturbance_estimate(self) -> None:
        r"""
        Reset the internal disturbance estimate to zero.

        This method sets the disturbance estimate back to

        .. math::

        \hat{\mathbf{d}} = \mathbf{0},

        effectively disabling disturbance compensation until the estimate is built
        up again through subsequent tracking updates.

        :return: None.
        :rtype: None

        """
        self.dist_estimate = np.zeros(3)

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
        Plan an optimal trajectory using ALTRO with disturbance-aware TVLQR gains.

        This method invokes the shared planning pipeline provided by
        :meth:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase._calculate_trajectory_common`
        and returns a trajectory configured for disturbance estimation.

        Trajectory contents
        -------------------
        The resulting :class:`~ADCS.controller.helpers.Trajectory` object contains:

        - Nominal state and control sequences.
        - Time-varying LQR gains compatible with the augmented disturbance model.
        - Auxiliary planner data needed for tracking.

        The trajectory is marked with ``use_disturbance_estimation = True``, which
        enables disturbance-related logic inside the trajectory during tracking.

        :param t_start: Planning start time in J2000 centuries.
        :type t_start: float
        :param duration: Planning horizon length in seconds.
        :type duration: float
        :param x_0: Initial state vector used to seed the optimizer.
        :type x_0: ADCS.state.State
        :param os_0: Initial orbital state used to seed environment propagation.
        :type os_0: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param goals: Goal list defining the pointing objectives.
        :type goals: :class:`~ADCS.CONOPS.goallist.GoalList`
        :param verbose: If true, enable planner verbosity and diagnostic output.
        :type verbose: bool
        :return: Planned trajectory with disturbance-aware TVLQR gains.
        :rtype: :class:`~ADCS.controller.helpers.Trajectory`

        """
        if verbose:
            print(f"Planning trajectory with KwDist: Start={t_start:.5f}, Dur={duration}s")

        lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
            t_start, duration, x_0, os_0, goals, verbose
        )

        # Create trajectory with disturbance estimation enabled
        return Trajectory.from_arrays(
            lqr_times, Xset, Uset, Kset, Sset,
            use_disturbance_estimation=True
        )
