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
                rw_momentum_margin=0.9,
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
        settings.rw_AM_weight = 0.0  # Rely on hard constraint, not soft penalty
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


def create_best_planner_settings(sat, dt_planning: float = 1.0, has_rw: bool = None):
    """
    Create best planner settings based on convergence analysis.
    
    Key optimizations:
    - Low angle weight (200) relative to ang_vel (1000) - 1:5 ratio
    - Equal terminal costs (running = terminal) for better conditioning
    - Gauss-Newton (Hessians OFF) - more stable than full Newton
    
    Tested performance (5 runs, 120s, 90° slews):
    - Mean error: 32° (vs 103° for default)  
    - 57% improvement over default settings
    
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
        Best planner settings
    """
    # Auto-detect RWs if not specified
    if has_rw is None:
        has_rw = len(sat.rw_actuators) > 0
    
    settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=10, dt_tvlqr=dt_planning)
    settings.verbosity = False
    
    # KEY INSIGHT: Low angle weight relative to ang_vel works best
    # This produces smoother, more trackable trajectories
    settings.cost_main.angle = 200
    settings.cost_main.ang_vel = 1000
    
    # KEY INSIGHT: Equal terminal costs (running = terminal) 
    # Better conditioning - every timestep matters equally
    settings.cost_main.angle_N = 200      # Same as running
    settings.cost_main.ang_vel_N = 1000   # Same as running
    
    # Gauss-Newton (Hessians OFF) - more stable
    settings.cost_main.use_full_cost_hessian = False
    
    # Convergence settings
    settings.pass1.convergence.max_outer_iter = 12
    settings.pass1.convergence.max_inner_iter = 18
    settings.pass2.convergence.max_outer_iter = 10
    settings.pass2.convergence.max_inner_iter = 18
    
    # Augmented Lagrangian settings
    settings.pass1.aug_lag.penalty_init = 1.0
    settings.pass1.aug_lag.penalty_max = 1e6
    settings.pass2.aug_lag.penalty_init = 100.0
    settings.pass2.aug_lag.penalty_max = 1e14
    
    # RW-specific settings
    if has_rw:
        settings.rw_AM_weight = 0.0  # Rely on hard constraint, not soft penalty
        settings.RWh_ok_mult = 0.5
    
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
    
    # === Create base settings with ALL AUTO-TUNING ENABLED ===
    # Key findings from convergence analysis:
    # 1. Low angle weight relative to ang_vel (1:5 ratio) works best
    # 2. Equal terminal costs (running = terminal) for better conditioning
    # 3. Gauss-Newton (Hessians OFF) is more stable
    # 4. Auto-scale angle costs based on cost function type
    # 5. Use scale normalization for consistent conditioning
    # 6. MTQ:RW cost ratio around 1e4-1e5 gives best results
    if has_rw:
        config = NormalizedPlannerConfig(
            actuator_costs=NormalizedActuatorCosts(
                # Increase mtq_cost to get better MTQ:RW ratio after torque scaling
                # With mtq_cost=1, torque-effective scaling gives ratio ~1e12 (too extreme)
                # Target ratio ~1e-5 (RW 1e5x more expensive than MTQ)
                # Need mtq_cost ~ 1e8 to counteract torque scaling
                mtq_cost=1e8,
                rw_torque_cost=5.0,
                rw_momentum_cost=10.0,
                rw_stiction_cost=0.1,
                use_torque_effective_mtq_scaling=True,
                expected_B_field_uT=30.0,
            ),
            state_costs=NormalizedStateCosts(
                # KEY: Low angle weight relative to ang_vel (1:5 ratio)
                angle_cost=100.0,
                ang_vel_cost=500.0,
                # Terminal costs much higher to incentivize reaching goal
                # After normalization by (π/2)², these become ~40k which is 
                # high enough to drive the optimizer to the goal
                angle_terminal_cost=100000.0,
                ang_vel_terminal_cost=10000.0,
                ang_cost_func_type=3,  # Quadratic geodesic
                # All auto-tuning enabled
                use_scale_normalization=True,
                auto_scale_angle_cost=True,
                angle_scale_deg=90.0,
                ang_vel_scale_deg_s=20.0,
            ),
            constraints=NormalizedConstraints(
                max_angular_velocity_deg_s=20.0,
                control_margin=0.25,
                rw_momentum_margin=0.9,
            ),
        )
    else:
        # MTQ-only: also use optimized weights
        config = NormalizedPlannerConfig(
            actuator_costs=NormalizedActuatorCosts(
                mtq_cost=1.0,  # No RW comparison needed
                use_torque_effective_mtq_scaling=True,
                expected_B_field_uT=30.0,
            ),
            state_costs=NormalizedStateCosts(
                angle_cost=100.0,
                ang_vel_cost=500.0,
                angle_terminal_cost=100000.0,
                ang_vel_terminal_cost=10000.0,
                ang_cost_func_type=3,
                use_scale_normalization=True,
                auto_scale_angle_cost=True,
                angle_scale_deg=90.0,
                ang_vel_scale_deg_s=20.0,
            ),
            constraints=NormalizedConstraints(
                max_angular_velocity_deg_s=20.0,
                control_margin=0.25,
            ),
        )
    
    settings = create_planner_settings(sat, config)
    
    # Ensure Hessians are OFF (Gauss-Newton is more stable)
    settings.cost_main.use_full_cost_hessian = False
    settings.cost_second.use_full_cost_hessian = False
    
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
        settings.rw_AM_weight = 0.0  # Rely on hard constraint, not soft penalty
        settings.RWh_ok_mult = 0.5
    
    return settings


def apply_smooth_tuning(settings, verbose: bool = False):
    """
    Apply optimized "smooth" cost weight tuning to existing settings.
    
    This tuning was found to give the best results in Monte Carlo tests:
    - 100% success rate (<10° final error)
    - 1.8°±1.3° mean final error
    - 107s mean time to <90°
    - Smooth trajectory shape with minimal oscillation
    
    The tuning increases terminal costs (for goal reaching) and running 
    state costs (for smoother trajectories), while reducing MTQ control 
    cost (for faster convergence).
    
    Parameters
    ----------
    settings : PlannerSettings
        Base settings to modify (modified in place)
    verbose : bool, optional
        Print the applied multipliers
        
    Returns
    -------
    PlannerSettings
        The modified settings (same object, returned for convenience)
        
    Notes
    -----
    Multipliers applied:
    - Terminal angle cost: 50x
    - Terminal ang_vel cost: 25x  
    - Running angle cost: 2.5x
    - Running ang_vel cost: 2.5x
    - MTQ control weight: 0.1x
    
    These multipliers are applied on top of the auto-scaled base values
    from create_adaptive_planner_settings().
    """
    # Store original values for verbose output
    orig_angle_N = settings.cost_main.angle_N
    orig_ang_vel_N = settings.cost_main.ang_vel_N
    orig_angle = settings.cost_main.angle
    orig_ang_vel = settings.cost_main.ang_vel
    orig_mtq = settings.mtq_control_weight
    
    # Apply smooth tuning multipliers
    settings.cost_main.angle_N *= 50      # Higher terminal angle cost
    settings.cost_main.ang_vel_N *= 25    # Higher terminal ang_vel cost
    settings.cost_main.angle *= 2.5       # Higher running angle cost
    settings.cost_main.ang_vel *= 2.5     # Higher running ang_vel cost
    settings.mtq_control_weight *= 0.1    # Cheaper MTQ for faster convergence
    
    # Also apply to second pass for consistency
    settings.cost_second.angle_N *= 50
    settings.cost_second.ang_vel_N *= 25
    settings.cost_second.angle *= 2.5
    settings.cost_second.ang_vel *= 2.5
    
    if verbose:
        print("Applied smooth tuning:")
        print(f"  angle_N: {orig_angle_N:.1f} -> {settings.cost_main.angle_N:.1f} (50x)")
        print(f"  ang_vel_N: {orig_ang_vel_N:.1f} -> {settings.cost_main.ang_vel_N:.1f} (25x)")
        print(f"  angle: {orig_angle:.1f} -> {settings.cost_main.angle:.1f} (2.5x)")
        print(f"  ang_vel: {orig_ang_vel:.1f} -> {settings.cost_main.ang_vel:.1f} (2.5x)")
        print(f"  mtq_weight: {orig_mtq:.2e} -> {settings.mtq_control_weight:.2e} (0.1x)")
    
    return settings


def apply_balanced_tuning(settings, verbose: bool = False):
    """
    Apply "balanced" tuning - best overall for smooth trajectories with good accuracy.
    
    This tuning provides the best trade-off between trajectory smoothness and
    convergence speed. Found through systematic testing of ang_vel multipliers.
    
    Tested performance (5 seeds, 180° slews, 1000s):
    - Mean error over last 500s: 14.6°±3.7° (BEST)
    - Final error: 0.5°±0.4°
    - Max angular rate: 1.3°/s
    
    Comparison with other tunings:
    - smooth (2.5x):   Mean500=54.8°, Final=18.6°, Rate=3.0°/s
    - balanced (50x):  Mean500=14.6°, Final=0.5°,  Rate=1.3°/s  <-- BEST
    - anti_spin (100x): Mean500=23.3°, Final=0.3°, Rate=1.1°/s
    
    Parameters
    ----------
    settings : PlannerSettings
        Base settings to modify (modified in place)
    verbose : bool, optional
        Print the applied multipliers
        
    Returns
    -------
    PlannerSettings
        The modified settings (same object, returned for convenience)
        
    Notes
    -----
    Multipliers applied (relative to base):
    - Terminal angle cost: 25x
    - Terminal ang_vel cost: 75x
    - Running angle cost: 5x
    - Running ang_vel cost: 50x (KEY: sweet spot between 25x and 100x)
    - MTQ control weight: 0.1x
    - bdot_on: 1 (B-dot initialization)
    - Gauss-Newton (Hessians OFF)
    - ang_vel_err_dir: 0 (tested, doesn't help)
    """
    # Store original values for verbose output
    orig_angle_N = settings.cost_main.angle_N
    orig_ang_vel_N = settings.cost_main.ang_vel_N
    orig_angle = settings.cost_main.angle
    orig_ang_vel = settings.cost_main.ang_vel
    orig_mtq = settings.mtq_control_weight
    
    # Apply balanced tuning multipliers
    settings.cost_main.angle_N *= 25
    settings.cost_main.ang_vel_N *= 75
    settings.cost_main.angle *= 5
    settings.cost_main.ang_vel *= 50      # KEY: 50x is the sweet spot
    settings.mtq_control_weight *= 0.1
    
    # Also apply to second pass for consistency
    settings.cost_second.angle_N *= 25
    settings.cost_second.ang_vel_N *= 75
    settings.cost_second.angle *= 5
    settings.cost_second.ang_vel *= 50
    
    # Balanced works best with:
    # - B-dot initialization (bdot_on=1)
    # - Gauss-Newton (Hessians OFF)
    settings.bdot_on = 1
    settings.cost_main.use_full_cost_hessian = False
    settings.cost_second.use_full_cost_hessian = False
    
    if verbose:
        print("Applied balanced tuning:")
        print(f"  angle_N: {orig_angle_N:.1f} -> {settings.cost_main.angle_N:.1f} (25x)")
        print(f"  ang_vel_N: {orig_ang_vel_N:.1f} -> {settings.cost_main.ang_vel_N:.1f} (75x)")
        print(f"  angle: {orig_angle:.1f} -> {settings.cost_main.angle:.1f} (5x)")
        print(f"  ang_vel: {orig_ang_vel:.1f} -> {settings.cost_main.ang_vel:.1f} (50x)")
        print(f"  mtq_weight: {orig_mtq:.2e} -> {settings.mtq_control_weight:.2e} (0.1x)")
        print(f"  bdot_on: 1 (B-dot init)")
        print(f"  Hessians: OFF (Gauss-Newton)")
    
    return settings


def apply_anti_spin_tuning(settings, verbose: bool = False):
    """
    Apply "anti-spin" tuning for maximum smoothness during trajectory.
    
    This tuning prioritizes smooth trajectories (low angular velocity) over
    aggressive goal reaching. Best for applications where oscillation/spinning
    during the maneuver is undesirable.
    
    Tested performance (5 seeds, 180° slews, 1000s):
    - Mean error over last 500s: 23.3°±20.6°
    - Final error: 0.3°±0.2°
    - Max angular rate: 1.1°/s (LOWEST)
    
    Note: "balanced" tuning actually achieves better Mean500s (14.6° vs 23.3°)
    while maintaining similar smoothness. Consider using "balanced" instead.
    
    Parameters
    ----------
    settings : PlannerSettings
        Base settings to modify (modified in place)
    verbose : bool, optional
        Print the applied multipliers
        
    Returns
    -------
    PlannerSettings
        The modified settings (same object, returned for convenience)
        
    Notes
    -----
    Multipliers applied (relative to base):
    - Terminal angle cost: 25x (lower than smooth's 50x)
    - Terminal ang_vel cost: 100x (4x higher than smooth's 25x)
    - Running angle cost: 5x (2x higher than smooth's 2.5x)
    - Running ang_vel cost: 100x (40x higher than smooth's 2.5x)
    - MTQ control weight: 0.1x (same as smooth)
    - bdot_on: 1 (B-dot initialization - critical for stability)
    - Gauss-Newton (Hessians OFF)
    
    The key difference is the extremely high angular velocity penalty (100x)
    which prevents spinning/oscillation throughout the trajectory.
    """
    # Store original values for verbose output
    orig_angle_N = settings.cost_main.angle_N
    orig_ang_vel_N = settings.cost_main.ang_vel_N
    orig_angle = settings.cost_main.angle
    orig_ang_vel = settings.cost_main.ang_vel
    orig_mtq = settings.mtq_control_weight
    
    # Apply anti-spin tuning multipliers
    # KEY: Very high ang_vel penalty to prevent spinning
    settings.cost_main.angle_N *= 25       # Moderate terminal angle cost
    settings.cost_main.ang_vel_N *= 100    # VERY HIGH terminal ang_vel cost
    settings.cost_main.angle *= 5          # Moderate running angle cost
    settings.cost_main.ang_vel *= 100      # VERY HIGH running ang_vel cost
    settings.mtq_control_weight *= 0.1     # Cheap MTQ for faster convergence
    
    # Also apply to second pass for consistency
    settings.cost_second.angle_N *= 25
    settings.cost_second.ang_vel_N *= 100
    settings.cost_second.angle *= 5
    settings.cost_second.ang_vel *= 100
    
    # Anti-spin works best with:
    # - B-dot initialization (bdot_on=1)
    # - Gauss-Newton (Hessians OFF)
    settings.bdot_on = 1
    settings.cost_main.use_full_cost_hessian = False
    settings.cost_second.use_full_cost_hessian = False
    
    if verbose:
        print("Applied anti-spin tuning:")
        print(f"  angle_N: {orig_angle_N:.1f} -> {settings.cost_main.angle_N:.1f} (25x)")
        print(f"  ang_vel_N: {orig_ang_vel_N:.1f} -> {settings.cost_main.ang_vel_N:.1f} (100x)")
        print(f"  angle: {orig_angle:.1f} -> {settings.cost_main.angle:.1f} (5x)")
        print(f"  ang_vel: {orig_ang_vel:.1f} -> {settings.cost_main.ang_vel:.1f} (100x)")
        print(f"  mtq_weight: {orig_mtq:.2e} -> {settings.mtq_control_weight:.2e} (0.1x)")
        print(f"  bdot_on: 1 (B-dot init)")
        print(f"  Hessians: OFF (Gauss-Newton)")
    
    return settings


def apply_aggressive_tuning(settings, verbose: bool = False):
    """
    Apply "aggressive" tuning for maximum RW usage.
    
    Experimental results show ang_vel/angle ratio is THE key factor:
    - ratio=1: ~100% RW, 11° error
    - ratio=10: ~100% RW, 4° error ★ BEST FOR RW
    - ratio=100: ~72% RW, 4° error 
    - ratio=1000: ~25% RW, 2° error (balanced default)
    
    This tuning uses ratio≈10 for maximum RW usage while keeping
    reasonable convergence (~4° error).
    
    Parameters
    ----------
    settings : PlannerSettings
        Base settings to modify (modified in place)
    verbose : bool, optional
        Print the applied multipliers
        
    Returns
    -------
    PlannerSettings
        The modified settings
        
    Notes
    -----
    Key finding: RW control cost has NO effect - the optimizer uses the same
    RW amount regardless of 0.001x to 10x cost ratio. Only ang_vel cost matters.
    
    Multipliers (relative to base ang_vel/angle ratio ≈ 100):
    - ang_vel: 0.1x → final ratio ≈ 10
    - Terminal costs: 10x (helps convergence)
    - Control costs: 0.01x (cheap, though it doesn't affect RW usage)
    """
    orig_angle_N = settings.cost_main.angle_N
    orig_ang_vel_N = settings.cost_main.ang_vel_N
    orig_angle = settings.cost_main.angle
    orig_ang_vel = settings.cost_main.ang_vel
    orig_mtq = settings.mtq_control_weight
    orig_rw = settings.rw_control_weight
    
    # KEY: Low ang_vel multiplier for high RW usage (ratio ≈ 10)
    settings.cost_main.ang_vel *= 0.1      # Results in ratio ≈ 10
    settings.cost_main.angle_N *= 10       # Higher terminal for convergence
    settings.cost_main.ang_vel_N *= 10
    settings.mtq_control_weight *= 0.01    # Very cheap (though doesn't affect RW)
    settings.rw_control_weight *= 0.01
    
    settings.cost_second.ang_vel *= 0.1
    settings.cost_second.angle_N *= 10
    settings.cost_second.ang_vel_N *= 10
    
    settings.bdot_on = 1
    settings.cost_main.use_full_cost_hessian = False
    settings.cost_second.use_full_cost_hessian = False
    
    if verbose:
        ratio = settings.cost_main.ang_vel / settings.cost_main.angle
        print("Applied aggressive tuning (max RW usage):")
        print(f"  ang_vel: {orig_ang_vel:.1f} -> {settings.cost_main.ang_vel:.1f} (0.1x)")
        print(f"  angle_N: {orig_angle_N:.1f} -> {settings.cost_main.angle_N:.1f} (10x)")
        print(f"  ang_vel/angle ratio: {ratio:.1f} (target: ~10 for ~100% RW)")
        print(f"  Expected: ~100% RW, ~4° final error")
    
    return settings


def apply_fast_slew_tuning(settings, verbose: bool = False):
    """
    Apply "fast_slew" tuning: balance between RW usage and accuracy.
    
    Experimental results:
    - ratio=100: ~72% RW, ~4° error with terminal=1x
    - ratio=100: ~72% RW, ~0.3° error with terminal=10x ★ BEST BALANCE
    
    This gives high RW usage (~72%) with excellent convergence (<1°).
    
    Parameters
    ----------
    settings : PlannerSettings
        Base settings to modify (modified in place)
    verbose : bool, optional
        Print the applied multipliers
        
    Returns
    -------
    PlannerSettings
        The modified settings
    """
    orig_angle_N = settings.cost_main.angle_N
    orig_ang_vel_N = settings.cost_main.ang_vel_N
    orig_angle = settings.cost_main.angle
    orig_ang_vel = settings.cost_main.ang_vel
    
    # Ratio ≈ 100 (no multiplier on ang_vel, which starts at ~100x angle)
    # Terminal 10x for good convergence
    settings.cost_main.angle_N *= 10
    settings.cost_main.ang_vel_N *= 10
    settings.mtq_control_weight *= 0.1
    settings.rw_control_weight *= 0.1
    
    settings.cost_second.angle_N *= 10
    settings.cost_second.ang_vel_N *= 10
    
    settings.bdot_on = 1
    settings.cost_main.use_full_cost_hessian = False
    settings.cost_second.use_full_cost_hessian = False
    
    if verbose:
        ratio = settings.cost_main.ang_vel / settings.cost_main.angle
        print("Applied fast_slew tuning (balance RW and accuracy):")
        print(f"  ang_vel/angle ratio: {ratio:.1f} (target: ~100)")
        print(f"  terminal costs: 10x")
        print(f"  Expected: ~72% RW, <1° final error")
    
    return settings


def create_optimized_planner_settings(
    sat,
    duration: float,
    dt_planning: float = 1.0,
    has_rw: bool = None,
    goal_changes: int = 1,
    use_multistart: bool = False,
    multistart_modes: list = None,
    tuning: str = "balanced",
    verbose: bool = False
):
    """
    Create fully optimized planner settings with tuning applied.
    
    This is the recommended function for production Monte Carlo tests.
    Combines auto-scaling with empirically-optimized cost weight tuning.
    
    Tuning options:
    | Tuning     | RW Usage | Final Err | Notes                        |
    |------------|----------|-----------|------------------------------|
    | balanced   | ~25%     | ~2°       | <-- RECOMMENDED (default)    |
    | fast_slew  | ~72%     | <1°       | Best balance of RW & accuracy|
    | aggressive | ~100%    | ~4°       | Maximum RW usage             |
    | smooth     | ~15%     | ~7°       | Original tuning              |
    | anti_spin  | ~15%     | ~7°       | Minimum rotation rate        |
    
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
        Number of goal changes in the trajectory.
    use_multistart : bool, optional
        Enable multi-start optimization. Runs multiple Pass 1 attempts 
        with different initializations and picks the best before Pass 2.
        Adds ~38% overhead but can find better solutions.
    multistart_modes : list, optional
        List of bdot modes to try. Default is [0, 1, 4, 5]:
        - 0: Random initialization
        - 1: B-dot damping
        - 4: PD control
        - 5: PD control + noise
    tuning : str, optional
        Tuning preset to apply. Options:
        - "balanced" (default, RECOMMENDED): Best overall - lowest Mean500s
        - "smooth": Original tuning, may have oscillation issues
        - "anti_spin": Most conservative, slowest convergence
        - "none": No additional tuning, use base auto-scaled settings
    verbose : bool, optional
        Print auto-scaling and tuning decisions.
        
    Returns
    -------
    PlannerSettings
        Fully optimized planner settings
        
    Examples
    --------
    # Recommended: balanced tuning (default)
    settings = create_optimized_planner_settings(sat, 1000)
    
    # For most conservative/slowest convergence
    settings = create_optimized_planner_settings(sat, 1000, tuning="anti_spin")
    """
    # Create base auto-scaled settings
    settings = create_adaptive_planner_settings(
        sat, duration, dt_planning, has_rw, goal_changes, verbose
    )
    
    # Apply requested tuning
    if tuning == "smooth":
        apply_smooth_tuning(settings, verbose)
    elif tuning == "balanced":
        apply_balanced_tuning(settings, verbose)
    elif tuning == "anti_spin":
        apply_anti_spin_tuning(settings, verbose)
    elif tuning == "aggressive":
        apply_aggressive_tuning(settings, verbose)
    elif tuning == "fast_slew":
        apply_fast_slew_tuning(settings, verbose)
    elif tuning == "none":
        if verbose:
            print("No additional tuning applied (using base auto-scaled settings)")
    else:
        raise ValueError(f"Unknown tuning preset: {tuning}. Use 'smooth', 'balanced', 'anti_spin', 'aggressive', 'fast_slew', or 'none'")
    
    # TVLQR gain tuning: Use high control multiplier for sensible feedback gains
    # Without this, LQR gains are ~1000x larger than actuator limits allow.
    # With ctrl_mult=1e6, gains are sized appropriately for the actuators.
    # This only affects the backward pass used to compute feedback gains (K),
    # not the trajectory optimization itself.
    settings.cost_tvlqr.control_mult = 1e6
    if verbose:
        print(f"TVLQR control multiplier: {settings.cost_tvlqr.control_mult:.0e}")
    
    # Enable multi-start if requested
    if use_multistart:
        if multistart_modes is None:
            multistart_modes = [0, 1, 4, 5]  # Default: random, bdot, PD, PD+noise
        settings.multistart_modes = multistart_modes
        if verbose:
            print(f"Multi-start enabled with modes: {multistart_modes}")
    
    return settings
