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


def create_good_planner_settings(sat, dt_planning: float = 1.0, has_rw: bool = True):
    """
    Create well-conditioned planner settings.
    
    These settings were tuned to provide:
    - Good numerical conditioning (Quu condition ~37k vs 100k with legacy)
    - Fast convergence (3x speedup over legacy)
    - Sub-degree pointing accuracy for 90° slews
    
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
        config = NormalizedPlannerConfig(
            actuator_costs=NormalizedActuatorCosts(
                mtq_cost=1.0,
                rw_torque_cost=5.0,
                rw_momentum_cost=10.0,
                rw_stiction_cost=1.0,
            ),
            state_costs=NormalizedStateCosts(
                angle_cost=1000.0,
                angle_terminal_cost=1000000.0,
                ang_vel_cost=1000.0,
                ang_vel_terminal_cost=100000.0,
            ),
        )
    else:
        # MTQ-only: no RW costs needed
        config = NormalizedPlannerConfig(
            actuator_costs=NormalizedActuatorCosts(
                mtq_cost=1.0,
                rw_torque_cost=1.0,  # Won't be used
                rw_momentum_cost=1.0,
                rw_stiction_cost=1.0,
            ),
            state_costs=NormalizedStateCosts(
                angle_cost=1000.0,
                angle_terminal_cost=1000000.0,
                ang_vel_cost=1000.0,
                ang_vel_terminal_cost=100000.0,
            ),
        )
    
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
    settings.pass1.convergence.max_outer_iter = 8
    settings.pass1.convergence.max_inner_iter = 30
    settings.pass2.convergence.max_outer_iter = 4
    settings.pass2.convergence.max_inner_iter = 15
    
    # Use default augmented Lagrangian settings
    # Note: Soft constraint violations in the plan are handled by actuator saturation at runtime
    
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
