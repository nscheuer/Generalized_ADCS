"""
ALTRO Trajectory Planner Settings Configuration.

This module defines PlannerSettings, the main configuration class for the Augmented
Lagrangian iLQR (ALTRO) trajectory optimizer used for spacecraft attitude control.

Key Concepts
------------
**Two-Pass Optimization**: The planner runs two sequential optimization passes:
  - Pass 1 (Main): Finds a feasible trajectory from the initial guess
  - Pass 2 (Refinement): Polish the solution with tighter convergence

**Timing Hierarchy**: Three time scales interact:
  - dt_tp: Trajectory planner timestep (coarse, e.g., 30s) - controls trajectory resolution
  - dt_tvlqr: TVLQR controller timestep (fine, e.g., 1s) - controls feedback update rate
  - dt_control: Actual control loop rate - typically matches dt_tvlqr

**Cost Function Structure**: The cost J = Σ[state_cost + control_cost] + terminal_cost
  - Running costs (angle, ang_vel) penalize deviation during trajectory
  - Terminal costs (*_N) penalize final state error - typically 10x running costs
  - control_mult scales ALL actuator costs uniformly

Settings Interdependencies
--------------------------
1. **Timing Relations**:
   - dt_tp should be a multiple of dt_tvlqr (e.g., dt_tp = 10 * dt_tvlqr)
   - tvlqr_len determines how far ahead the TVLQR looks
   - tvlqr_overlap prevents discontinuities when switching trajectories
   - precalculation_time must exceed tvlqr_len for smooth operation

2. **Cost Weight Interactions**:
   - Higher angle costs → prioritizes pointing accuracy over velocity damping
   - Higher ang_vel costs → smoother trajectories but slower convergence
   - control_mult scales ALL actuator costs uniformly
   - Individual actuator weights (mtq_control_weight, rw_control_weight) set relative priority

3. **Actuator Weight Effects**:
   - mtq_control_weight vs rw_control_weight ratio determines actuator preference
   - Higher RW weight → prefers MTQs when both available
   - rw_AM_weight penalizes RW momentum buildup (prevents saturation)
   - rw_stic_weight penalizes low RW speeds (avoids stiction region)

4. **Constraint Interactions**:
   - wmax (angular velocity limit) affects trajectory aggressiveness
   - sun_limit_angle + camera_axis define keep-out zone
   - control_limit_scale reduces actuator limits for margin (default 75%)

5. **Convergence Tuning**:
   - Pass 1: Lower penalty_init (1e-3) for exploration, more iterations
   - Pass 2: Higher penalty_init (1e4) for constraint enforcement, fewer iterations
   - If not converging: increase max_outer_iter, decrease reg_init

Tuning Guide
------------
**Quick Start Presets**:

For **fast, aggressive maneuvers** (large angles, short time):
    cost_main = CostWeights(
        angle=1e4, angle_N=1e5,      # High pointing priority
        ang_vel=1e2, ang_vel_N=1e3,  # Allow high rates
        control_mult=0.1             # Cheap control
    )
    settings.wmax = 30 * np.pi / 180  # 30 deg/s limit

For **smooth, precise pointing** (imaging, communication):
    cost_main = CostWeights(
        angle=1e3, angle_N=1e4,
        ang_vel=1e5, ang_vel_N=1e6,  # Penalize velocity heavily
        control_mult=10.0            # Expensive control = smoother
    )
    settings.wmax = 5 * np.pi / 180   # 5 deg/s limit

For **detumbling** (rate reduction only):
    cost_main = CostWeights(
        angle=0.0, angle_N=0.0,      # Don't care about pointing
        ang_vel=1e4, ang_vel_N=1e6,  # Only minimize rates
        control_mult=1.0
    )

**Common Problems and Solutions**:

Problem: Trajectory doesn't converge (max iterations reached)
  - Increase max_outer_iter (try 40-50)
  - Decrease reg_init (try 1e-4)
  - Increase penalty_scale (try 20-50)
  - Check if maneuver is physically feasible given actuator limits

Problem: Oscillating controls (chattering)
  - Increase control_mult (penalize control changes)
  - Increase ang_vel cost weights
  - Decrease dt_tp for finer resolution
  - Check regularization settings (increase reg_min)

Problem: Constraint violations in final trajectory
  - Increase penalty_init for pass 2 (try 1e5-1e6)
  - Decrease c_max tolerance
  - Increase max_outer_iter

Problem: Slow computation time
  - Increase dt_tp (coarser trajectory)
  - Decrease max_inner_iter
  - Use lighter convergence tolerances (grad_tol=1e-2)

Problem: RW saturation during tracking
  - Increase rw_AM_weight (penalize momentum buildup)
  - Decrease RWh_max_mult (leave more margin)
  - Consider longer trajectory duration

**Tuning Order** (recommended sequence):
  1. Start with default CostWeights
  2. Adjust angle vs ang_vel ratio based on mission (pointing vs smoothness)
  3. Set appropriate wmax for your satellite
  4. Run trajectory, check convergence
  5. If not converging, adjust solver settings (iterations, penalties)
  6. If oscillating, increase control_mult or ang_vel weights
  7. Fine-tune actuator weights based on preferred actuator usage
"""
from __future__ import annotations

__all__ = ["PlannerSettings"]

import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray

from ADCS.controller.helpers.planner_subsettings import SolverPassConfig, CostWeights, InitTrajConfig, ConvergenceConfig, AugLagConfig
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance, Prop_Disturbance, General_Disturbance


class PlannerSettings:
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
        self.verbosity = False
        self.eps = 2.22044604925031e-16

        # Solver Configurations
        self.pass1 = pass1_config if pass1_config else SolverPassConfig()

        if pass2_config:
            self.pass2 = pass2_config
        else:
            self.pass2 = SolverPassConfig()
            self.pass2.convergence.max_outer_iter = 15
            self.pass2.convergence.max_inner_iter = 200
            self.pass2.regularization.reg_max = 1e12

        converge1 = ConvergenceConfig(max_outer_iter=20, max_inner_iter=150)
        auglag1 = AugLagConfig(penalty_init=1e-3)
        self.pass1 = pass1_config if pass1_config else SolverPassConfig(convergence=converge1, aug_lag=auglag1)
        converge2 = ConvergenceConfig(max_outer_iter=20, max_inner_iter=75)
        auglag2 = AugLagConfig(penalty_init=1e4)
        self.pass2 = pass2_config if pass2_config else SolverPassConfig(convergence=converge2, aug_lag=auglag2)

        # Initilization
        self.init_traj = init_traj if init_traj else InitTrajConfig()

        # Hardware Constraints
        self.control_limit_scale = 0.75
        self.umax = self.control_limit_scale * np.array([act.u_max for act in self.est_sat.actuators])
        self.wmax = 20*np.pi/180.0
        self.sun_limit_angle = 1*np.pi/180.0
        self.camera_axis = np.array([[0, 0, 1]]).T

        # Actuator Weights for the C++ model construction
        self.mtq_control_weight = 1e3
        self.rw_control_weight = 1e5
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
            angle_N=1e4,   # 10x running cost
            ang_vel=1e4,
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
        #     angle=1e5,
        #     angle_N=1e8,
        #     ang_vel=1e8,
        #     ang_vel_N=1e10,
        #     ang_vel_mag=0.0,
        #     ang_vel_mag_N=0.0,
        #     control_mult=1e5 / self.mtq_control_weight,
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
        """Return system configuration tuple for C++ planner."""
        return (self.J_est, self.dt_tp, self.dt_tvlqr, self.eps,
                self.tvlqr_len, self.tvlqr_overlap)

    def mainAlilqrSettings(self) -> Tuple[Tuple, Tuple, Tuple, Tuple]:
        """Return first pass solver settings for C++ planner."""
        return (
            self.pass1.line_search.to_tuple(),
            self.pass1.aug_lag.to_tuple(),
            self.pass1.convergence.to_tuple(state_len=self.est_sat.state_len),
            self.pass1.regularization.to_tuple()
        )

    def secondAlilqrSettings(self) -> Tuple[Tuple, Tuple, Tuple, Tuple]:
        """Return second pass solver settings for C++ planner."""
        return (
            self.pass2.line_search.to_tuple(),
            self.pass2.aug_lag.to_tuple(),
            self.pass2.convergence.to_tuple(state_len=self.est_sat.state_len),
            self.pass2.regularization.to_tuple()
        )

    def initTrajSettings(self) -> Tuple[float, float, Tuple, Tuple]:
        """Return initial trajectory generation settings."""
        return self.init_traj.to_tuple()

    def optMainCostSettings(self) -> Tuple:
        """Return main pass cost weights for C++ planner."""
        return self.cost_main.to_tuple()

    def optSecondCostSettings(self) -> Tuple:
        """Return second pass cost weights for C++ planner."""
        return self.cost_second.to_tuple()

    def optTVLQRCostSettings(self, tracking_LQR_formulation: int) -> Tuple:
        """Return TVLQR cost weights with tracking formulation flag."""
        return self.cost_tvlqr.to_tuple(tracking_LQR_formulation)

    def planner_disturbance_settings(self) -> Tuple[Tuple[bool, ...], NDArray, NDArray, int, NDArray, NDArray, NDArray]:
        return (
            (self.plan_for_aero, self.plan_for_prop, self.plan_for_srp, 
             self.plan_for_gg, self.plan_for_resdipole, self.plan_for_gendist),
            self.srp_coeff, self.drag_coeff, self.coeff_N, 
            self.prop_torque, self.gendist_torq, self.res_dipole
        )