"""
Factory functions for creating PlannerSettings from normalized configurations.

This module bridges the gap between the user-friendly NormalizedPlannerConfig
and the raw PlannerSettings expected by the optimizer.
"""

import numpy as np
from typing import Optional

from .planner_settings import PlannerSettings
from .planner_subsettings import CostWeights, SolverPassConfig
from .normalized_settings import (
    NormalizedPlannerConfig,
    NormalizedSettingsConverter,
    PlannerPresets,
)


def create_planner_settings(
    satellite,
    config: Optional[NormalizedPlannerConfig] = None,
    preset: Optional[str] = None,
    verbose: bool = False,
    **kwargs
) -> PlannerSettings:
    """
    Create PlannerSettings from a normalized configuration.
    
    This is the recommended way to create planner settings for new code.
    It automatically scales weights based on actuator hardware for well-conditioned
    optimization.
    
    Parameters
    ----------
    satellite : Satellite
        The satellite object with actuator definitions.
    config : NormalizedPlannerConfig, optional
        Normalized configuration. If None, uses preset or defaults.
    preset : str, optional
        Name of preset to use: 'mtq_only', 'mtq_plus_rw', 'rw_primary', 
        'agile_imaging', or 'dissertation'.
    verbose : bool
        If True, print diagnostic information about scaling.
    **kwargs
        Additional arguments passed directly to PlannerSettings constructor.
        These override any values computed from the normalized config.
    
    Returns
    -------
    PlannerSettings
        Configured planner settings with auto-scaled weights.
    
    Examples
    --------
    >>> # Using a preset
    >>> settings = create_planner_settings(satellite, preset='mtq_plus_rw')
    
    >>> # Using custom normalized config
    >>> config = NormalizedPlannerConfig(
    ...     actuator_costs=NormalizedActuatorCosts(mtq_cost=1.0, rw_torque_cost=5.0),
    ...     constraints=NormalizedConstraints(max_angular_velocity_deg_s=15.0)
    ... )
    >>> settings = create_planner_settings(satellite, config=config)
    
    >>> # With overrides
    >>> settings = create_planner_settings(
    ...     satellite, preset='mtq_only', 
    ...     dt_tp=20.0, bdot_on=0
    ... )
    """
    # Resolve configuration
    if config is None:
        if preset is not None:
            config = _get_preset(preset)
        else:
            # Auto-detect based on satellite configuration
            config = _auto_detect_config(satellite)
    
    # Convert to raw weights
    converter = NormalizedSettingsConverter(satellite, config)
    raw_weights = converter.compute_raw_weights()
    
    if verbose:
        _print_diagnostics(converter)
    
    # Build cost weights
    cost_main = CostWeights(
        angle=raw_weights.get('angle_weight', 1e3),
        angle_N=raw_weights.get('angle_weight', 1e3),
        ang_vel=raw_weights.get('angvel_weight', 1e3),
        ang_vel_N=raw_weights.get('angvel_weight', 1e3),
        control_mult=1.0,  # Already scaled in actuator weights
        ang_cost_func_type=raw_weights.get('ang_cost_func_type', 2),
    )
    
    cost_second = CostWeights(
        angle=raw_weights.get('angle_weight', 1e3) * 10,  # Higher for refinement
        angle_N=raw_weights.get('angle_weight_N', 1e6),
        ang_vel=raw_weights.get('angvel_weight', 1e3) * 0.1,  # Lower rate cost
        ang_vel_N=raw_weights.get('angvel_weight_N', 1e5),
        control_mult=10.0,  # Higher control cost in refinement
        ang_cost_func_type=raw_weights.get('ang_cost_func_type', 2),
    )
    
    cost_tvlqr = CostWeights(
        angle=raw_weights.get('angle_weight', 1e3) * 10,
        angle_N=raw_weights.get('angle_weight_N', 1e6) * 10,
        ang_vel=raw_weights.get('angvel_weight', 1e3) * 100,
        ang_vel_N=raw_weights.get('angvel_weight_N', 1e5) * 100,
        control_mult=10.0,
        ang_cost_func_type=raw_weights.get('ang_cost_func_type', 2),
    )
    
    # Create base settings
    planner_settings = PlannerSettings(
        est_sat=satellite,
        cost_main=cost_main,
        cost_second=cost_second,
        cost_tvlqr=cost_tvlqr,
        **{k: v for k, v in kwargs.items() if k in [
            'dt_control', 'pass1_config', 'pass2_config', 'init_traj',
            'dt_tvlqr', 'tvlqr_len', 'tvlqr_overlap', 'dt_tp',
            'precalculation_time', 'traj_overlap', 'bdot_on', 'debug_plot_on',
            'include_gg', 'include_resdipole', 'include_prop',
            'include_drag', 'include_srp', 'include_gendist'
        ]}
    )
    
    # Apply raw weights
    planner_settings.mtq_control_weight = raw_weights.get('mtq_control_weight', 1e3)
    planner_settings.rw_control_weight = raw_weights.get('rw_control_weight', 1e8)
    planner_settings.rw_AM_weight = raw_weights.get('rw_AM_weight', 1e4)
    planner_settings.rw_stic_weight = raw_weights.get('rw_stic_weight', 1e0)
    planner_settings.RWh_max_mult = raw_weights.get('RWh_max_mult', 0.5)
    planner_settings.RWh_stiction_mult = raw_weights.get('RWh_stiction_mult', 0.01)
    planner_settings.RWh_ok_mult = raw_weights.get('RWh_ok_mult', 0.15)
    planner_settings.wmax = raw_weights.get('wmax', 20 * np.pi / 180)
    planner_settings.control_limit_scale = raw_weights.get('control_limit_scale', 0.75)
    
    if 'sun_limit_angle' in raw_weights:
        planner_settings.sun_limit_angle = raw_weights['sun_limit_angle']
    if 'camera_axis' in raw_weights:
        planner_settings.camera_axis = raw_weights['camera_axis']
    
    # Store reference to normalized config for debugging
    planner_settings._normalized_config = config
    planner_settings._scaling_info = converter.get_diagnostic_info()
    
    return planner_settings


def _get_preset(name: str) -> NormalizedPlannerConfig:
    """Get a preset configuration by name."""
    presets = {
        'mtq_only': PlannerPresets.mtq_only,
        'mtq_plus_rw': PlannerPresets.mtq_plus_rw_assist,
        'rw_primary': PlannerPresets.rw_primary,
        'agile_imaging': PlannerPresets.agile_imaging,
        'dissertation': PlannerPresets.dissertation_equivalent,
    }
    
    if name not in presets:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(presets.keys())}")
    
    return presets[name]()


def _auto_detect_config(satellite) -> NormalizedPlannerConfig:
    """Auto-detect appropriate configuration based on satellite hardware."""
    
    num_mtq = sum(1 for a in satellite.actuators if type(a).__name__ == 'MTQ')
    num_rw = sum(1 for a in satellite.actuators if type(a).__name__ == 'RW')
    
    if num_rw == 0:
        return PlannerPresets.mtq_only()
    elif num_rw <= 2:
        return PlannerPresets.mtq_plus_rw_assist()
    else:
        return PlannerPresets.rw_primary()


def _print_diagnostics(converter: NormalizedSettingsConverter):
    """Print diagnostic information about scaling."""
    info = converter.get_diagnostic_info()
    
    print("=" * 60)
    print("Normalized Planner Settings - Diagnostic Info")
    print("=" * 60)
    print(f"Number of actuators: {info['num_actuators']}")
    print(f"Estimated Quu condition number: {info['estimated_Quu_condition']:.1f}")
    print()
    print("Scaling factors:")
    print(f"  Control scales: {info['scaling']['control_scales']}")
    print(f"  Angle scale: {info['scaling']['angle_scale']:.4f} rad")
    print(f"  Ang vel scale: {info['scaling']['ang_vel_scale']:.4f} rad/s")
    print()
    print("Computed raw weights:")
    for key, value in info['raw_weights'].items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4e}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)


def estimate_conditioning(satellite, config: NormalizedPlannerConfig = None) -> dict:
    """
    Estimate the conditioning of the optimization problem.
    
    This is useful for diagnosing convergence issues or comparing
    different configurations.
    
    Parameters
    ----------
    satellite : Satellite
        The satellite object.
    config : NormalizedPlannerConfig, optional
        Configuration to analyze. If None, uses auto-detected config.
    
    Returns
    -------
    dict
        Dictionary with conditioning metrics.
    """
    if config is None:
        config = _auto_detect_config(satellite)
    
    converter = NormalizedSettingsConverter(satellite, config)
    return converter.get_diagnostic_info()
