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
