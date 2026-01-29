"""
Shared planner settings for Monte Carlo tests.

These are the well-conditioned normalized settings that produce good results.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.controller.helpers import (
    PlannerSettings, create_planner_settings,
    NormalizedPlannerConfig, NormalizedActuatorCosts, NormalizedStateCosts,
)
from ADCS.controller.helpers.normalized_settings import (
    PlannerPresets, NormalizedConstraints
)


def create_good_planner_settings(sat, dt_planning: float = 1.0, has_rw: bool = None):
    """
    Create well-conditioned planner settings.
    
    Uses torque-effective MTQ scaling and high penalty for best results.
    Tested performance (10 runs, 90° slews):
    - Plan time: ~3.4s
    - Mean error: ~24° 
    - 70% of runs < 30° error
    - 90% of runs < 60° error
    
    Parameters
    ----------
    sat : Satellite
        The satellite object
    dt_planning : float
        Planning timestep in seconds
    has_rw : bool, optional
        Whether the satellite has reaction wheels. If None, auto-detected.
        
    Returns
    -------
    PlannerSettings
        Well-conditioned planner settings
    """
    # Auto-detect RWs if not specified
    if has_rw is None:
        has_rw = len(sat.rw_actuators) > 0
    
    if has_rw:
        # Use torque-effective MTQ scaling + scale normalization
        config = NormalizedPlannerConfig(
            actuator_costs=NormalizedActuatorCosts(
                mtq_cost=1.0,
                rw_torque_cost=5.0,
                rw_momentum_cost=10.0,
                rw_stiction_cost=0.1,
                use_torque_effective_mtq_scaling=True,  # Makes MTQs cheap
                expected_B_field_uT=30.0,
            ),
            state_costs=NormalizedStateCosts(
                angle_cost=100.0,
                angle_terminal_cost=1000.0,
                ang_vel_cost=100.0,
                ang_vel_terminal_cost=1000.0,
                use_scale_normalization=True,
                angle_scale_deg=90.0,
                ang_vel_scale_deg_s=20.0,
            ),
            constraints=NormalizedConstraints(
                max_angular_velocity_deg_s=20.0,
                control_margin=0.25,
                rw_momentum_margin=0.5,
            ),
        )
    else:
        # MTQ-only preset with scale normalization
        config = PlannerPresets.mtq_only_normalized()
    
    settings = create_planner_settings(sat, config)
    
    # Key settings
    settings.bdot_on = 0  # Random init, not B-dot (better for slew maneuvers)
    settings.dt_tp = 10   # Coarse planning timestep
    settings.dt_tvlqr = dt_planning
    settings.verbosity = False
    
    # RW-specific settings
    if has_rw:
        settings.rw_AM_weight = 1e4
        settings.RWh_ok_mult = 0.5
    
    # Convergence settings
    settings.pass1.convergence.max_outer_iter = 10
    settings.pass1.convergence.max_inner_iter = 15
    settings.pass2.convergence.max_outer_iter = 8
    settings.pass2.convergence.max_inner_iter = 15
    
    # High penalty augmented Lagrangian settings - works best with torque-effective MTQ
    settings.pass1.aug_lag.penalty_init = 10.0
    settings.pass1.aug_lag.penalty_max = 1e8
    settings.pass2.aug_lag.penalty_init = 1e5
    settings.pass2.aug_lag.penalty_max = 1e18
    
    return settings


def create_fast_planner_settings(sat, dt_planning: float = 1.0, has_rw: bool = None):
    """
    Create faster planner settings (fewer iterations, lower accuracy).
    
    Parameters
    ----------
    sat : Satellite
        The satellite object
    dt_planning : float
        Planning timestep in seconds
    has_rw : bool, optional
        Whether the satellite has reaction wheels. If None, auto-detected.
        
    Returns
    -------
    PlannerSettings
        Fast planner settings
    """
    settings = create_good_planner_settings(sat, dt_planning, has_rw)
    
    # Fewer iterations for speed
    settings.pass1.convergence.max_outer_iter = 6
    settings.pass1.convergence.max_inner_iter = 10
    settings.pass2.convergence.max_outer_iter = 4
    settings.pass2.convergence.max_inner_iter = 10
    
    return settings


def create_legacy_planner_settings(sat, dt_planning: float = 1.0):
    """
    Create legacy (default) planner settings for comparison.
    """
    settings = PlannerSettings(
        est_sat=sat, 
        bdot_on=0, 
        dt_tp=10, 
        dt_tvlqr=dt_planning
    )
    settings.verbosity = False
    settings.pass1.convergence.max_outer_iter = 8
    settings.pass1.convergence.max_inner_iter = 30
    settings.pass2.convergence.max_outer_iter = 4
    settings.pass2.convergence.max_inner_iter = 15
    
    return settings


def create_adaptive_planner_settings(
    sat, 
    duration: float,
    dt_planning: float = 1.0,
    has_rw: bool = None,
    goal_changes: int = 1,
    verbose: bool = False
):
    """
    Create planner settings that auto-adapt based on trajectory properties.
    
    Automatically adjusts:
    - dt_tp (coarseness) based on duration
    - Iteration counts based on problem size
    - Cost weights based on trajectory length
    - MTQ cost scaling for better convergence
    
    Parameters
    ----------
    sat : Satellite
        The satellite object
    duration : float
        Trajectory duration in seconds
    dt_planning : float
        TVLQR planning timestep in seconds
    has_rw : bool, optional
        Whether the satellite has reaction wheels. If None, auto-detected.
    goal_changes : int, optional
        Number of goal changes in the trajectory. More changes = finer resolution.
        Default 1 (single fixed goal).
    verbose : bool, optional
        Print auto-scaling decisions.
        
    Returns
    -------
    PlannerSettings
        Auto-scaled planner settings
    """
    import numpy as np
    
    # Auto-detect RWs if not specified
    if has_rw is None:
        has_rw = len(sat.rw_actuators) > 0
    
    # === Auto-scale dt_tp based on duration ===
    # Target: 30-80 planning steps for good balance of speed and accuracy
    # More goal changes = need finer resolution
    target_steps_base = 50
    target_steps = target_steps_base + 20 * (goal_changes - 1)  # More steps for changing goals
    target_steps = np.clip(target_steps, 30, 100)
    
    # Compute dt_tp to achieve target steps
    dt_tp_raw = duration / target_steps
    # Round to nice values: 5, 10, 20, 50, 100
    nice_values = [5, 10, 20, 50, 100]
    dt_tp = min(nice_values, key=lambda x: abs(x - dt_tp_raw))
    
    # Ensure minimum 15 steps
    actual_steps = duration / dt_tp
    if actual_steps < 15:
        dt_tp = max(5, duration / 20)
    
    # dt_tvlqr should be finer than dt_tp
    dt_tvlqr = max(dt_planning, dt_tp / 10)
    
    if verbose:
        print(f"Auto-scaling for {duration}s trajectory:")
        print(f"  dt_tp: {dt_tp}s ({int(duration/dt_tp)} steps)")
        print(f"  dt_tvlqr: {dt_tvlqr}s")
    
    # === Create base settings ===
    if has_rw:
        config = NormalizedPlannerConfig(
            actuator_costs=NormalizedActuatorCosts(
                mtq_cost=1.0,
                rw_torque_cost=5.0,
                rw_momentum_cost=10.0,
                rw_stiction_cost=0.1,
                use_torque_effective_mtq_scaling=True,
                expected_B_field_uT=30.0,
            ),
            state_costs=NormalizedStateCosts(
                angle_cost=100.0,
                angle_terminal_cost=1000.0,
                ang_vel_cost=100.0,
                ang_vel_terminal_cost=1000.0,
                ang_cost_func_type=3,  # Quadratic geodesic - better for long trajectories
                use_scale_normalization=True,
                angle_scale_deg=90.0,
                ang_vel_scale_deg_s=20.0,
            ),
            constraints=NormalizedConstraints(
                max_angular_velocity_deg_s=20.0,
                control_margin=0.25,
                rw_momentum_margin=0.5,
            ),
        )
    else:
        config = PlannerPresets.mtq_only_normalized()
    
    settings = create_planner_settings(sat, config)
    
    # Apply computed timesteps
    settings.dt_tp = dt_tp
    settings.dt_tvlqr = dt_tvlqr
    settings.bdot_on = 0
    settings.verbosity = False
    
    # === Auto-scale iterations based on problem size ===
    # More steps = need more iterations, but cap to avoid excessive runtime
    step_factor = np.sqrt(actual_steps / 50)  # Normalized to 50 steps
    duration_factor = (duration / 120) ** 0.3  # Slower growth with duration
    
    # Use max of step and duration factors, but cap at 2x
    iter_factor = min(2.0, max(step_factor, duration_factor))
    
    base_outer = 12
    base_inner = 18
    
    settings.pass1.convergence.max_outer_iter = int(base_outer * iter_factor)
    settings.pass1.convergence.max_inner_iter = int(base_inner * iter_factor)
    settings.pass2.convergence.max_outer_iter = int(base_outer * 0.8 * iter_factor)
    settings.pass2.convergence.max_inner_iter = int(base_inner * iter_factor)
    
    if verbose:
        print(f"  Pass1 iters: {settings.pass1.convergence.max_outer_iter} outer, {settings.pass1.convergence.max_inner_iter} inner")
        print(f"  Pass2 iters: {settings.pass2.convergence.max_outer_iter} outer, {settings.pass2.convergence.max_inner_iter} inner")
    
    # === Auto-scale penalties based on duration ===
    # Longer trajectories need MUCH lower penalties to allow exploration
    # This is critical for convergence on long horizons
    penalty_duration_factor = (duration / 120) ** 0.75  # Aggressive scaling
    
    settings.pass1.aug_lag.penalty_init = 1.0 / penalty_duration_factor
    settings.pass1.aug_lag.penalty_max = 1e6 / penalty_duration_factor
    settings.pass2.aug_lag.penalty_init = 100.0 / penalty_duration_factor
    settings.pass2.aug_lag.penalty_max = 1e14
    
    # === Cheap MTQ for better convergence on long trajectories ===
    if duration > 200:
        mtq_scale = 0.01  # Very cheap MTQ for long trajectories
    elif duration > 100:
        mtq_scale = 0.1
    else:
        mtq_scale = 1.0
    
    settings.mtq_control_weight *= mtq_scale
    
    if verbose:
        print(f"  MTQ cost scale: {mtq_scale}x")
        print(f"  Pass1 penalty_init: {settings.pass1.aug_lag.penalty_init:.2f}")
    
    # RW-specific settings
    if has_rw:
        settings.rw_AM_weight = 1e4
        settings.RWh_ok_mult = 0.5
    
    return settings
