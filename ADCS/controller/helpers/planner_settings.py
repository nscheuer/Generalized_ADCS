from __future__ import annotations

__all__ = ["PlannerSettings"]

import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray

from ADCS.controller.helpers.planner_subsettings import SolverPassConfig, CostWeights, InitTrajConfig, ConvergenceConfig, AugLagConfig
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance, Prop_Disturbance, General_Disturbance


class PlannerSettings:
    r"""
    Container class for all configuration parameters required by the trajectory planner.

    This class aggregates physical parameters, solver configurations, actuator limits,
    cost weights, timing parameters, and disturbance models into a single interface
    consumed by the C++ trajectory planner.

    Mathematically, the planner solves a constrained optimal control problem of the form

    .. math::

        \min_{\mathbf{u}(t)} \;
        \int_{0}^{T}
        \ell\bigl(\mathbf{x}(t), \mathbf{u}(t)\bigr)\, dt
        + \ell_N\bigl(\mathbf{x}(T)\bigr)

    subject to nonlinear rigid-body dynamics

    .. math::

        \dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u}, \mathbf{d})

    actuator saturation limits, and optional environmental disturbance torques
    :math:`\mathbf{d}`.

    The settings in this class parameterize both the dynamics and the optimization
    algorithm, including multi-pass augmented Lagrangian iLQR configurations.

    :param est_sat:
        Estimated satellite model providing inertia, actuator definitions,
        state dimension, and disturbances.

    :type est_sat:
        :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`

    :param dt_control:
        Control update period in seconds.

    :type dt_control:
        float

    :param pass1_config:
        Solver configuration for the first (exploration) optimization pass.

    :type pass1_config:
        :class:`~ADCS.controller.helpers.planner_subsettings.SolverPassConfig`

    :param pass2_config:
        Solver configuration for the second (refinement) optimization pass.

    :type pass2_config:
        :class:`~ADCS.controller.helpers.planner_subsettings.SolverPassConfig`

    :param cost_main:
        Cost weights for the main optimization pass.

    :type cost_main:
        :class:`~ADCS.controller.helpers.planner_subsettings.CostWeights`

    :param cost_second:
        Cost weights for the second optimization pass.

    :type cost_second:
        :class:`~ADCS.controller.helpers.planner_subsettings.CostWeights`

    :param cost_tvlqr:
        Cost weights for time-varying LQR tracking.

    :type cost_tvlqr:
        :class:`~ADCS.controller.helpers.planner_subsettings.CostWeights`

    :param init_traj:
        Initial trajectory generation configuration.

    :type init_traj:
        :class:`~ADCS.controller.helpers.planner_subsettings.InitTrajConfig`

    :param dt_tvlqr:
        Time step for TVLQR controller discretization.

    :type dt_tvlqr:
        float

    :param tvlqr_len:
        Duration of each TVLQR segment in seconds.

    :type tvlqr_len:
        float

    :param tvlqr_overlap:
        Overlap duration between consecutive TVLQR segments in seconds.

    :type tvlqr_overlap:
        float

    :param dt_tp:
        Time step used by the trajectory planner.

    :type dt_tp:
        float

    :param precalculation_time:
        Planning horizon precomputation time in seconds.

    :type precalculation_time:
        float

    :param traj_overlap:
        Overlap duration between planned trajectories.

    :type traj_overlap:
        float

    :param bdot_on:
        Flag enabling B-dot detumbling logic.

    :type bdot_on:
        int

    :param debug_plot_on:
        Flag enabling debug plotting.

    :type debug_plot_on:
        bool

    :param include_gg:
        Flag enabling gravity-gradient disturbance modeling.

    :type include_gg:
        bool

    :param include_resdipole:
        Flag enabling residual dipole disturbance modeling.

    :type include_resdipole:
        bool

    :param include_prop:
        Flag enabling propulsion disturbance modeling.

    :type include_prop:
        bool

    :param include_drag:
        Flag enabling aerodynamic drag disturbance modeling.

    :type include_drag:
        bool

    :param include_srp:
        Flag enabling solar radiation pressure disturbance modeling.

    :type include_srp:
        bool

    :param include_gendist:
        Flag enabling generic disturbance torque modeling.

    :type include_gendist:
        bool

    """
    def __init__(
            self, 
            est_sat: EstimatedSatellite, 
            dt_control: float = 1.0,
            pass1_config: SolverPassConfig = None,
            pass2_config: SolverPassConfig = None,
            cost_main: CostWeights = None,
            cost_second: CostWeights = None,
            cost_tvlqr: CostWeights = None,
            init_traj: InitTrajConfig = None,
            dt_tvlqr: float = 1,
            tvlqr_len: float = 60,
            tvlqr_overlap: float = 15,
            dt_tp: float = 30.0,
            precalculation_time: float = 100,
            traj_overlap: float = 150,
            bdot_on: int = 1,
            multistart_modes: list = None,
            debug_plot_on: bool = False,
            include_gg: bool = False,
            include_resdipole: bool = False,
            include_prop: bool = False, 
            include_drag: bool = False, 
            include_srp: bool = False, 
            include_gendist: bool = False
    ) -> None:
        
        self.est_sat = est_sat

        # Physics and Timing
        self.dt_tvlqr = dt_tvlqr
        self.tvlqr_len = tvlqr_len
        self.tvlqr_overlap = tvlqr_overlap
        self.dt_tp = 10*dt_tvlqr if dt_tp is None else dt_tp
        self.precalculation_time = precalculation_time
        self.default_traj_length = tvlqr_len
        self.traj_overlap = traj_overlap
        self.debug_plot_on = debug_plot_on
        self.bdot_on = bdot_on
        self.multistart_modes = multistart_modes  # List of bdot modes for multi-start, e.g., [0, 1, 4, 5]
        self.verbosity = False
        self.eps = 2.22044604925031e-16
        
        # Pass 2 options
        self.skip_pass2_optimization = True   # ZOH forward sim + K-gains (avoids wound trajectories)
        self.pass2_warm_start_mode = 3        # 0=ZOH, 1=K-gain, 2=SLERP, 3=closed-loop inv dyn
        self.use_euler_pass2 = True           # Euler integration for Pass 2 (faster)
        
        # Infeasible start (ALTRO-style): Pass 1 uses slack variables on dynamics
        # constraint (s_k = 0 enforced via AL), allowing topologically correct but
        # dynamically infeasible trajectories (e.g., SLERP). Pass 2 is standard.
        # Only active when skip_pass2_optimization=False (2-pass mode).
        self.use_infeasible_start = False
        
        # Multi-resolution fallback: when enabled, calculate_trajectory tries
        # coarse dt first, then falls back to finer dt if the plan is wound.
        # This resolves 180° maneuver bifurcation at minimal compute cost.
        self.auto_refine_dt = False
        self.auto_refine_dt_fine = 1.0       # Fine dt to fall back to (seconds)
        self.auto_refine_dt_slerp = 2.0     # dt for SLERP+slacks fallback (seconds)
        self.auto_refine_spike_thresh = 40.0  # Plan angle (°) above which plan is "wound"
        
        # Time-varying angle cost: w_ang *= min + (1-min) * (k/(N-1))^power
        # power=0.0 → disabled (constant angle cost over trajectory)
        # power>0 → angle cost ramps from min at start to full at end
        # min = floor multiplier at k=0 (prevents zero angle cost at start)
        # Terminal cost (k=N-1) always uses full weight regardless of this setting
        # Pushes angle errors toward start of trajectory, preventing wound local minima
        self.ang_cost_time_power = 0.0
        self.ang_cost_time_min = 0.1  # 10% of full angle cost at trajectory start

        # Solver Configurations
        # Two-pass optimization strategy:
        # - Pass 1 (Exploration): Lower penalty (1e-3), more iterations for global search
        # - Pass 2 (Refinement): Higher penalty (1e4), fewer iterations for constraint satisfaction
        converge1 = ConvergenceConfig(max_outer_iter=20, max_inner_iter=60)
        auglag1 = AugLagConfig(penalty_init=1e-3)
        self.pass1 = pass1_config if pass1_config else SolverPassConfig(convergence=converge1, aug_lag=auglag1)

        converge2 = ConvergenceConfig(max_outer_iter=20, max_inner_iter=30)
        auglag2 = AugLagConfig(penalty_init=1e4)
        self.pass2 = pass2_config if pass2_config else SolverPassConfig(convergence=converge2, aug_lag=auglag2)

        # Initilization
        self.init_traj = init_traj if init_traj else InitTrajConfig()

        # Hardware Constraints
        self.control_limit_scale = 0.75
        self.umax = self.control_limit_scale * np.array([act.u_max for act in self.est_sat.actuators])
        self.wmax = 50*np.pi/180.0  # 50°/s - allows aggressive RW usage
        self.sun_limit_angle = 1*np.pi/180.0
        self.camera_axis = np.array([[0, 1, 0]]).T

        # Actuator Weights for the C++ model construction
        self.mtq_control_weight = 1e3
        self.rw_control_weight = 1e8
        self.magic_control_weight = 0.0001
        self.rw_AM_weight = 1e4
        self.rw_stic_weight = 1e0
        self.RWh_max_mult = 0.8
        self.RWh_stiction_mult = 0.01
        self.RWh_ok_mult = 0.5

        # Cost Configuration
        # Terminal costs 10x higher than running costs to prioritize goal reaching
        self.cost_main = cost_main if cost_main else CostWeights(
            angle=1e3,
            angle_N=1e6,   # 10x running cost
            ang_vel=1e3,
            ang_vel_N=1e5, # 10x running cost
            ang_vel_mag=0.0,
            ang_vel_mag_N=0.0,
            control_mult=1.0,
            ang_cost_func_type=2,
        )
        self.cost_second = cost_second if cost_second else self.cost_main#CostWeights(
        #     angle=1e3,
        #     angle_N=1e3,
        #     ang_vel=1.0,
        #     ang_vel_N=1.0,
        #     ang_vel_mag=0.0,
        #     ang_vel_mag_N=0.0,
        #     control_mult=100.0,
        #     ang_cost_func_type=2,
        # )
        self.cost_tvlqr = cost_tvlqr if cost_tvlqr else self.cost_main#CostWeights(
        #     angle=1e2,
        #     angle_N=1e3,
        #     ang_vel=1e6,
        #     ang_vel_N=1e8,
        #     ang_vel_mag=0.0,
        #     ang_vel_mag_N=0.0,
        #     control_mult=1.0,
        #     ang_cost_func_type=2,
        # )

        # Disturbance Settings
        self.plan_for_aero = include_drag
        self.plan_for_prop = include_prop
        self.plan_for_srp = include_srp
        self.plan_for_gg = include_gg
        self.plan_for_gendist = include_gendist
        self.plan_for_resdipole = include_resdipole

        self.srp_coeff = np.zeros((3,))
        self.drag_coeff = np.zeros((3,))
        self.coeff_N = 0
        self.res_dipole = sum([j.current_torque if isinstance(j, Dipole_Disturbance) else np.zeros(3) for j in est_sat.disturbances], start=np.zeros(3)).reshape((3,))
        self.prop_torque = sum([j.current_torque if isinstance(j, Prop_Disturbance) else np.zeros(3) for j in est_sat.disturbances], start=np.zeros(3)).reshape((3,))
        self.gendist_torq = np.array([0, 0, 0])
        self.J_est = est_sat.J_0

    def systemSettings(self) -> Tuple[NDArray[np.float64], float, float, float, float, float]:
        r"""
        Return system-level configuration parameters for the C++ planner.

        These parameters define the rigid-body dynamics discretization and numerical
        tolerances used by the planner. The inertia matrix enters the rotational
        equations of motion

        .. math::

            \mathbf{J}\dot{\boldsymbol{\omega}} =
            \boldsymbol{\tau} - \boldsymbol{\omega} \times (\mathbf{J}\boldsymbol{\omega})

        where :math:`\mathbf{J}` is the estimated inertia tensor.

        :return:
            Tuple containing inertia matrix, trajectory planner time step,
            TVLQR time step, numerical epsilon, TVLQR segment length,
            and TVLQR overlap duration.

        :rtype:
            Tuple[numpy.ndarray, float, float, float, float, float]

        """
        return (self.J_est, self.dt_tp, self.dt_tvlqr, self.eps,
                self.tvlqr_len, self.tvlqr_overlap)

    def mainAlilqrSettings(self) -> Tuple[Tuple, Tuple, Tuple, Tuple]:
        r"""
        Return solver settings for the first augmented Lagrangian iLQR pass.

        The first pass prioritizes global exploration with a low penalty parameter
        and increased iteration limits, solving a relaxed constrained optimization
        problem.

        :return:
            Tuple containing line search, augmented Lagrangian,
            convergence, and regularization settings.

        :rtype:
            Tuple[Tuple, Tuple, Tuple, Tuple]

        """
        return (
            self.pass1.line_search.to_tuple(),
            self.pass1.aug_lag.to_tuple(),
            self.pass1.convergence.to_tuple(state_len=self.est_sat.state_len),
            self.pass1.regularization.to_tuple()
        )

    def secondAlilqrSettings(self) -> Tuple[Tuple, Tuple, Tuple, Tuple]:
        r"""
        Return solver settings for the second augmented Lagrangian iLQR pass.

        The second pass increases constraint penalties to enforce feasibility
        and refine the solution obtained from the first pass.

        :return:
            Tuple containing line search, augmented Lagrangian,
            convergence, and regularization settings.

        :rtype:
            Tuple[Tuple, Tuple, Tuple, Tuple]

        """
        return (
            self.pass2.line_search.to_tuple(),
            self.pass2.aug_lag.to_tuple(),
            self.pass2.convergence.to_tuple(state_len=self.est_sat.state_len),
            self.pass2.regularization.to_tuple()
        )

    def initTrajSettings(self) -> Tuple[float, float, Tuple, Tuple]:
        r"""
        Return initial trajectory generation parameters.

        These settings define how the initial guess trajectory is constructed
        prior to optimization, which can significantly affect convergence.

        :return:
            Tuple containing initial trajectory configuration parameters.

        :rtype:
            Tuple[float, float, Tuple, Tuple]

        """
        return self.init_traj.to_tuple()

    def optMainCostSettings(self) -> Tuple:
        r"""
        Return cost weights for the main optimization pass.

        The running and terminal costs are defined as

        .. math::

            \ell = \mathbf{x}^\top \mathbf{Q} \mathbf{x}
            + \mathbf{u}^\top \mathbf{R} \mathbf{u}

        with increased terminal penalties to emphasize goal convergence.

        :return:
            Tuple encoding main-pass cost weights.

        :rtype:
            Tuple

        """
        return self.cost_main.to_tuple()

    def optSecondCostSettings(self) -> Tuple:
        r"""
        Return cost weights for the second optimization pass.

        These weights may differ from the main pass to emphasize constraint
        satisfaction or control smoothing.

        :return:
            Tuple encoding second-pass cost weights.

        :rtype:
            Tuple

        """
        return self.cost_second.to_tuple()

    def optTVLQRCostSettings(self, tracking_LQR_formulation: int) -> Tuple:
        r"""
        Return cost weights for the time-varying LQR controller.

        The TVLQR controller minimizes a quadratic tracking cost

        .. math::

            J = \int
            (\mathbf{x} - \mathbf{x}_r)^\top \mathbf{Q}
            (\mathbf{x} - \mathbf{x}_r)
            + \mathbf{u}^\top \mathbf{R} \mathbf{u} \, dt

        where the formulation can be adjusted via the tracking flag.

        :param tracking_LQR_formulation:
            Integer flag selecting the tracking LQR formulation.

        :type tracking_LQR_formulation:
            int

        :return:
            Tuple encoding TVLQR cost weights.

        :rtype:
            Tuple

        """
        return self.cost_tvlqr.to_tuple(tracking_LQR_formulation)

    def planner_disturbance_settings(self) -> Tuple[Tuple[bool, ...], NDArray, NDArray, int, NDArray, NDArray, NDArray]:
        r"""
        Return disturbance configuration parameters for the C++ planner.

        Disturbance torques are modeled as additive inputs

        .. math::

            \boldsymbol{\tau}_{\text{total}} =
            \boldsymbol{\tau}_{\text{control}} +
            \boldsymbol{\tau}_{\text{dist}}

        where the disturbance torque may include aerodynamic drag, solar radiation
        pressure, gravity-gradient effects, propulsion torques, residual dipole
        effects, or generic disturbances.

        :return:
            Tuple containing disturbance enable flags, SRP coefficients,
            drag coefficients, normalization factor, propulsion torque,
            generic disturbance torque, and residual dipole torque.

        :rtype:
            Tuple[Tuple[bool, ...], numpy.ndarray, numpy.ndarray, int,
                numpy.ndarray, numpy.ndarray, numpy.ndarray]

        """
        return (
            (self.plan_for_aero, self.plan_for_prop, self.plan_for_srp, 
             self.plan_for_gg, self.plan_for_resdipole, self.plan_for_gendist),
            self.srp_coeff, self.drag_coeff, self.coeff_N, 
            self.prop_torque, self.gendist_torq, self.res_dipole
        )