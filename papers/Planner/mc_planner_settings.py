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
from ADCS.controller.helpers.normalized_settings import PlannerPresets


def create_good_planner_settings(sat, dt_planning: float = 1.0, has_rw: bool = True):
    """
    Create well-conditioned planner settings.
    
    Uses the mtq_plus_rw_normalized preset which has scale normalization enabled
    for better Hessian conditioning. Tested to give:
    - ~2s planning time
    - ~30° mean error on 90° slews (with limited iterations for speed)
    
    Parameters
    ----------
    sat : Satellite
        The satellite object
    dt_planning : float
        Planning timestep in seconds
    has_rw : bool
        Whether the satellite has reaction wheels
        
    Returns
    -------
    PlannerSettings
        Well-conditioned planner settings
    """
    if has_rw:
        # Use the normalized preset with scale normalization
        config = PlannerPresets.mtq_plus_rw_normalized()
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
    
    # Convergence settings - balanced for speed and accuracy
    settings.pass1.convergence.max_outer_iter = 10
    settings.pass1.convergence.max_inner_iter = 15
    settings.pass2.convergence.max_outer_iter = 8
    settings.pass2.convergence.max_inner_iter = 15
    
    # Augmented Lagrangian settings - key for convergence
    # Pass 1: Low penalty for exploration
    settings.pass1.aug_lag.penalty_init = 1.0
    settings.pass1.aug_lag.penalty_max = 1e6
    
    # Pass 2: High penalty for constraint enforcement
    settings.pass2.aug_lag.penalty_init = 1e4
    settings.pass2.aug_lag.penalty_max = 1e16
    
    return settings


def create_high_accuracy_planner_settings(sat, dt_planning: float = 1.0, has_rw: bool = True):
    """
    Create planner settings optimized for accuracy over speed.
    
    Uses more iterations for better convergence at the cost of longer planning time.
    
    Parameters
    ----------
    sat : Satellite
        The satellite object
    dt_planning : float
        Planning timestep in seconds
    has_rw : bool
        Whether the satellite has reaction wheels
        
    Returns
    -------
    PlannerSettings
        High-accuracy planner settings
    """
    if has_rw:
        config = PlannerPresets.mtq_plus_rw_normalized()
    else:
        config = PlannerPresets.mtq_only_normalized()
    
    settings = create_planner_settings(sat, config)
    
    settings.bdot_on = 0
    settings.dt_tp = 10
    settings.dt_tvlqr = dt_planning
    settings.verbosity = False
    
    if has_rw:
        settings.rw_AM_weight = 1e4
        settings.RWh_ok_mult = 0.5
    
    # More iterations for better convergence
    settings.pass1.convergence.max_outer_iter = 15
    settings.pass1.convergence.max_inner_iter = 25
    settings.pass2.convergence.max_outer_iter = 12
    settings.pass2.convergence.max_inner_iter = 25
    
    settings.pass1.aug_lag.penalty_init = 1.0
    settings.pass1.aug_lag.penalty_max = 1e6
    settings.pass2.aug_lag.penalty_init = 1e4
    settings.pass2.aug_lag.penalty_max = 1e16
    
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
