"""
Normalized planner settings for well-conditioned optimization.

This module provides a normalized formulation where costs and constraints are
specified in intuitive, hardware-agnostic terms. The normalization ensures:

1. Control Hessian (Quu) has bounded condition number regardless of actuator sizing
2. Same tuning parameters work across different satellites  
3. Regularization has uniform effect across all actuators
4. Constraint penalties scale appropriately

Key Concept:
-----------
Instead of raw quadratic costs `w * u²`, we use normalized costs:
    cost = w_normalized * (u / u_max)²

This means w_normalized represents "cost of using actuator at full capacity",
which is independent of the actuator's physical limits.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np


@dataclass
class NormalizedActuatorCosts:
    """
    Actuator costs in normalized (unit capacity) space.
    
    Each cost represents the contribution to the objective when that actuator
    is commanded at 100% of its capacity. This makes the costs independent of
    actuator sizing.
    
    Example:
        mtq_cost=1.0, rw_torque_cost=10.0 means using RW at full capacity is
        10x more expensive than using MTQ at full capacity.
    
    Parameters
    ----------
    mtq_cost : float
        Cost of MTQ at full dipole moment (per axis). Default 1.0 (baseline).
    rw_torque_cost : float
        Cost of RW at max torque. Default 10.0 (prefer MTQ for momentum exchange).
    rw_momentum_cost : float
        Cost of RW momentum relative to max. Default 1.0.
    rw_stiction_cost : float
        Cost of RW near zero-crossing (stiction avoidance). Default 0.1.
    thruster_cost : float
        Cost of thruster at max force. Default 1000.0 (fuel is expensive).
    magic_cost : float
        Cost of magic actuator at max torque. Default 1.0.
    """
    mtq_cost: float = 1.0
    rw_torque_cost: float = 10.0
    rw_momentum_cost: float = 1.0
    rw_stiction_cost: float = 0.1
    thruster_cost: float = 1000.0
    magic_cost: float = 1.0


@dataclass
class NormalizedStateCosts:
    """
    State costs in normalized space.
    
    Costs are specified per unit of normalized error:
    - Angle errors normalized by angle_scale (default π radians)
    - Angular velocity errors normalized by ang_vel_scale (default wmax)
    
    Parameters
    ----------
    angle_cost : float
        Cost of 1 radian attitude error (or angle_scale if specified).
    angle_terminal_cost : float
        Terminal cost multiplier for attitude error.
    ang_vel_cost : float
        Cost of 1 rad/s angular velocity error (or ang_vel_scale if specified).
    ang_vel_terminal_cost : float
        Terminal cost multiplier for angular velocity error.
    ang_cost_func_type : int
        0: 1-cos(angle) - smooth near zero
        2: acos geodesic - true geodesic distance
        4: squared MRP - fast computation
    """
    angle_cost: float = 100.0
    angle_terminal_cost: float = 1000.0
    ang_vel_cost: float = 100.0
    ang_vel_terminal_cost: float = 1000.0
    ang_cost_func_type: int = 2  # acos geodesic (dissertation default)


@dataclass 
class NormalizedConstraints:
    """
    Constraints in physical units with automatic scaling.
    
    Parameters
    ----------
    max_angular_velocity_deg_s : float
        Maximum allowed angular velocity in degrees/second.
    control_margin : float
        Fraction of actuator capacity to reserve (0.0 = use 100%, 0.25 = use 75%).
    sun_exclusion_angle_deg : float
        Minimum angle from sun for protected axis. 0 = no constraint.
    sun_exclusion_body_axis : array-like
        Body-frame axis to protect from sun (e.g., [0,0,1] for +Z).
    rw_momentum_margin : float
        Fraction of RW momentum capacity to stay within.
    """
    max_angular_velocity_deg_s: float = 20.0
    control_margin: float = 0.25
    sun_exclusion_angle_deg: float = 0.0
    sun_exclusion_body_axis: Optional[np.ndarray] = None
    rw_momentum_margin: float = 0.5


@dataclass
class ScalingFactors:
    """
    Physical scales used for normalization.
    
    These can be computed automatically from satellite hardware and constraints,
    or specified manually for custom behavior.
    
    Parameters
    ----------
    control_scales : np.ndarray
        Effective max for each control input (u_max * (1 - margin)).
    angle_scale : float
        Normalization scale for angles (default π).
    ang_vel_scale : float
        Normalization scale for angular velocity (default wmax).
    time_scale : float
        Characteristic time scale (default trajectory duration).
    """
    control_scales: Optional[np.ndarray] = None
    angle_scale: float = np.pi
    ang_vel_scale: Optional[float] = None  # Computed from constraints if None
    time_scale: Optional[float] = None


@dataclass
class NormalizedPlannerConfig:
    """
    Complete normalized planner configuration.
    
    This is the main user-facing configuration class. It combines actuator costs,
    state costs, and constraints into a single configuration that can be used
    to create well-conditioned planner settings.
    
    Parameters
    ----------
    actuator_costs : NormalizedActuatorCosts
        Normalized costs for each actuator type.
    state_costs : NormalizedStateCosts
        Normalized state error costs.
    constraints : NormalizedConstraints
        Physical constraints.
    global_control_scale : float
        Global multiplier on all control costs (for state/control tradeoff tuning).
    global_state_scale : float
        Global multiplier on all state costs.
    
    Example
    -------
    >>> config = NormalizedPlannerConfig(
    ...     actuator_costs=NormalizedActuatorCosts(mtq_cost=1.0, rw_torque_cost=5.0),
    ...     state_costs=NormalizedStateCosts(angle_cost=100.0, ang_vel_cost=100.0),
    ...     constraints=NormalizedConstraints(max_angular_velocity_deg_s=15.0)
    ... )
    """
    actuator_costs: NormalizedActuatorCosts = field(default_factory=NormalizedActuatorCosts)
    state_costs: NormalizedStateCosts = field(default_factory=NormalizedStateCosts)
    constraints: NormalizedConstraints = field(default_factory=NormalizedConstraints)
    global_control_scale: float = 1.0
    global_state_scale: float = 1.0


class NormalizedSettingsConverter:
    """
    Converts normalized settings to raw planner weights.
    
    This class handles the conversion from user-friendly normalized settings
    to the raw weight values expected by the C++ optimizer. It also computes
    scaling factors and estimates the resulting condition number.
    
    Parameters
    ----------
    satellite : Satellite
        The satellite object with actuator definitions.
    config : NormalizedPlannerConfig
        The normalized configuration to convert.
    """
    
    def __init__(self, satellite, config: NormalizedPlannerConfig):
        self.satellite = satellite
        self.config = config
        self._scaling = None
        self._raw_weights = None
        
    def compute_scaling_factors(self) -> ScalingFactors:
        """Compute normalization scales from satellite hardware."""
        if self._scaling is not None:
            return self._scaling
            
        scaling = ScalingFactors()
        
        # Control scales = effective actuator limits
        control_scales = []
        margin = self.config.constraints.control_margin
        
        for act in self.satellite.actuators:
            u_eff = act.u_max * (1.0 - margin)
            control_scales.append(u_eff)
        
        scaling.control_scales = np.array(control_scales)
        
        # Angular velocity scale from constraint
        scaling.ang_vel_scale = (
            self.config.constraints.max_angular_velocity_deg_s * np.pi / 180.0
        )
        
        # Angle scale is π by default
        scaling.angle_scale = np.pi
        
        self._scaling = scaling
        return scaling
    
    def compute_raw_weights(self) -> Dict[str, float]:
        """
        Convert normalized costs to raw weights for C++ optimizer.
        
        The conversion formula is:
            w_raw = w_normalized * global_scale / u_max²
        
        This ensures that when u = u_max, the cost contribution equals
        w_normalized * global_scale.
        
        Returns
        -------
        dict
            Dictionary with raw weight values for PlannerSettings.
        """
        if self._raw_weights is not None:
            return self._raw_weights
            
        scaling = self.compute_scaling_factors()
        act_costs = self.config.actuator_costs
        state_costs = self.config.state_costs
        g_ctrl = self.config.global_control_scale
        g_state = self.config.global_state_scale
        
        raw = {}
        
        # MTQ control weight
        mtq_actuators = [a for a in self.satellite.actuators 
                        if type(a).__name__ == 'MTQ']
        if mtq_actuators:
            # Use first MTQ's limit (assume all same)
            u_max = mtq_actuators[0].u_max * (1.0 - self.config.constraints.control_margin)
            raw['mtq_control_weight'] = act_costs.mtq_cost * g_ctrl / (u_max ** 2)
        
        # RW control weights
        rw_actuators = [a for a in self.satellite.actuators 
                       if type(a).__name__ == 'RW']
        if rw_actuators:
            u_max = rw_actuators[0].u_max * (1.0 - self.config.constraints.control_margin)
            raw['rw_control_weight'] = act_costs.rw_torque_cost * g_ctrl / (u_max ** 2)
            
            # Momentum weight - scale by h_max
            h_max = rw_actuators[0].h_max * self.config.constraints.rw_momentum_margin
            raw['rw_AM_weight'] = act_costs.rw_momentum_cost * g_ctrl / (h_max ** 2)
            
            # Stiction weight
            raw['rw_stic_weight'] = act_costs.rw_stiction_cost * g_ctrl
        
        # State weights
        raw['angle_weight'] = state_costs.angle_cost * g_state
        raw['angle_weight_N'] = state_costs.angle_terminal_cost * g_state
        raw['angvel_weight'] = state_costs.ang_vel_cost * g_state
        raw['angvel_weight_N'] = state_costs.ang_vel_terminal_cost * g_state
        raw['ang_cost_func_type'] = state_costs.ang_cost_func_type
        
        # Constraints in radians
        raw['wmax'] = self.config.constraints.max_angular_velocity_deg_s * np.pi / 180.0
        raw['control_limit_scale'] = 1.0 - self.config.constraints.control_margin
        
        if self.config.constraints.sun_exclusion_angle_deg > 0:
            raw['sun_limit_angle'] = self.config.constraints.sun_exclusion_angle_deg * np.pi / 180.0
            if self.config.constraints.sun_exclusion_body_axis is not None:
                raw['camera_axis'] = np.array(self.config.constraints.sun_exclusion_body_axis).reshape(3, 1)
        
        # RW momentum multipliers
        raw['RWh_max_mult'] = self.config.constraints.rw_momentum_margin
        raw['RWh_stiction_mult'] = 0.01  # Small zone near zero
        raw['RWh_ok_mult'] = 0.15  # Comfortable operating zone
        
        self._raw_weights = raw
        return raw
    
    def estimate_control_hessian_condition(self) -> float:
        """
        Estimate the condition number of the control cost Hessian.
        
        A well-conditioned problem has condition number close to 1.
        Values above 1000 may cause numerical issues.
        
        Returns
        -------
        float
            Estimated condition number of Quu.
        """
        act_costs = self.config.actuator_costs
        
        # Collect normalized costs for each actuator
        costs = []
        
        for act in self.satellite.actuators:
            if type(act).__name__ == 'MTQ':
                costs.append(act_costs.mtq_cost)
            elif type(act).__name__ == 'RW':
                costs.append(act_costs.rw_torque_cost)
            else:
                costs.append(act_costs.magic_cost)
        
        if len(costs) == 0:
            return 1.0
        
        return max(costs) / max(min(costs), 1e-10)
    
    def get_diagnostic_info(self) -> Dict[str, any]:
        """
        Get diagnostic information about the scaling and conditioning.
        
        Returns
        -------
        dict
            Diagnostic information including scales, raw weights, and condition estimate.
        """
        scaling = self.compute_scaling_factors()
        raw = self.compute_raw_weights()
        
        return {
            'scaling': {
                'control_scales': scaling.control_scales,
                'angle_scale': scaling.angle_scale,
                'ang_vel_scale': scaling.ang_vel_scale,
            },
            'raw_weights': raw,
            'estimated_Quu_condition': self.estimate_control_hessian_condition(),
            'num_actuators': len(self.satellite.actuators),
        }


# =============================================================================
# Preset Configurations
# =============================================================================

class PlannerPresets:
    """Factory methods for common planner configurations."""
    
    @staticmethod
    def mtq_only() -> NormalizedPlannerConfig:
        """
        Configuration for MTQ-only satellites.
        
        Emphasizes smooth control since MTQs are the only actuators.
        """
        return NormalizedPlannerConfig(
            actuator_costs=NormalizedActuatorCosts(
                mtq_cost=1.0,
            ),
            state_costs=NormalizedStateCosts(
                angle_cost=100.0,
                angle_terminal_cost=1000.0,
                ang_vel_cost=1000.0,  # Higher rate cost for stability
                ang_vel_terminal_cost=1000.0,
            ),
            constraints=NormalizedConstraints(
                max_angular_velocity_deg_s=10.0,  # Conservative for MTQ-only
                control_margin=0.25,
            ),
        )
    
    @staticmethod
    def mtq_plus_rw_assist() -> NormalizedPlannerConfig:
        """
        Configuration for satellites with MTQs and 1-2 RWs.
        
        MTQs are primary, RWs assist with axes perpendicular to B-field.
        """
        return NormalizedPlannerConfig(
            actuator_costs=NormalizedActuatorCosts(
                mtq_cost=1.0,
                rw_torque_cost=5.0,  # Slightly prefer MTQ
                rw_momentum_cost=10.0,  # Penalize momentum buildup
                rw_stiction_cost=0.1,
            ),
            state_costs=NormalizedStateCosts(
                angle_cost=100.0,
                angle_terminal_cost=1000.0,
                ang_vel_cost=100.0,
                ang_vel_terminal_cost=1000.0,
            ),
            constraints=NormalizedConstraints(
                max_angular_velocity_deg_s=20.0,
                control_margin=0.25,
                rw_momentum_margin=0.5,
            ),
        )
    
    @staticmethod
    def rw_primary() -> NormalizedPlannerConfig:
        """
        Configuration for RW-primary satellites (3+ RWs with full authority).
        
        RWs handle attitude control, MTQs used for momentum dumping.
        """
        return NormalizedPlannerConfig(
            actuator_costs=NormalizedActuatorCosts(
                mtq_cost=10.0,  # Discourage MTQ use during slews
                rw_torque_cost=1.0,  # RW is preferred
                rw_momentum_cost=1.0,
                rw_stiction_cost=0.5,
            ),
            state_costs=NormalizedStateCosts(
                angle_cost=100.0,
                angle_terminal_cost=10000.0,  # High precision at end
                ang_vel_cost=10.0,  # Allow faster slews
                ang_vel_terminal_cost=10000.0,
            ),
            constraints=NormalizedConstraints(
                max_angular_velocity_deg_s=30.0,  # Faster slews possible
                control_margin=0.2,
                rw_momentum_margin=0.7,
            ),
        )
    
    @staticmethod
    def agile_imaging() -> NormalizedPlannerConfig:
        """
        Configuration for agile imaging satellites.
        
        Optimized for fast slews with precise settling.
        """
        return NormalizedPlannerConfig(
            actuator_costs=NormalizedActuatorCosts(
                mtq_cost=5.0,
                rw_torque_cost=1.0,
                rw_momentum_cost=0.5,  # Allow momentum buildup during slew
                rw_stiction_cost=1.0,  # Avoid stiction for smooth pointing
            ),
            state_costs=NormalizedStateCosts(
                angle_cost=10.0,  # Low during slew
                angle_terminal_cost=100000.0,  # Very high at end
                ang_vel_cost=1.0,  # Allow fast rates
                ang_vel_terminal_cost=100000.0,  # Must be stationary at end
            ),
            constraints=NormalizedConstraints(
                max_angular_velocity_deg_s=45.0,
                control_margin=0.1,  # Use more actuator authority
                rw_momentum_margin=0.8,
            ),
            global_state_scale=1.0,
        )
    
    @staticmethod
    def dissertation_equivalent() -> NormalizedPlannerConfig:
        """
        Configuration that approximates dissertation cost ratios.
        
        The dissertation used:
        - MTQ cost ratio at max: 1000 * 1.0² = 1000
        - RW cost ratio at max: 1e5 * 0.005² = 2.5
        - Ratio: 400:1
        
        This preset creates similar ratios in normalized space.
        """
        return NormalizedPlannerConfig(
            actuator_costs=NormalizedActuatorCosts(
                mtq_cost=400.0,  # MTQ is "expensive" at capacity
                rw_torque_cost=1.0,  # RW is "cheap"
                rw_momentum_cost=10.0,
                rw_stiction_cost=1.0,
            ),
            state_costs=NormalizedStateCosts(
                angle_cost=100.0,
                angle_terminal_cost=100.0,
                ang_vel_cost=1000.0,
                ang_vel_terminal_cost=1000.0,
                ang_cost_func_type=2,  # acos geodesic
            ),
            constraints=NormalizedConstraints(
                max_angular_velocity_deg_s=20.0,
                control_margin=0.25,
            ),
            global_control_scale=1.0,
        )
