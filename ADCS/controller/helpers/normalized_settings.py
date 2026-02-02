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
        Note: This penalizes the magnetic dipole moment, not the produced torque.
        This is intentional because MTQs draw power proportional to dipole,
        regardless of whether that dipole produces useful torque.
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
    use_torque_effective_mtq_scaling : bool
        If True, scale MTQ cost to encourage full utilization despite low torque
        authority. Since MTQs produce much less torque than RWs (τ = m × B is
        typically 100-1000x smaller than RW torque), this scaling makes MTQs
        "cheap" to use at full capacity.
        
        The scaling ensures that if MTQ and RW have the same normalized cost,
        using MTQ at full capacity costs the same as using RW at full capacity,
        even though MTQ produces much less torque. This encourages the optimizer
        to readily use MTQs for what little torque they can provide.
        
        Default False (dipole-based cost that penalizes power draw uniformly).
        
    expected_B_field_uT : float
        Expected magnetic field magnitude in μT for torque-effective scaling.
        Only used when use_torque_effective_mtq_scaling=True.
        Typical LEO values: 20-50 μT. Default 30.0 μT.
    """
    mtq_cost: float = 1.0
    rw_torque_cost: float = 10.0
    rw_momentum_cost: float = 1.0
    rw_stiction_cost: float = 0.1
    thruster_cost: float = 1000.0
    magic_cost: float = 1.0
    use_torque_effective_mtq_scaling: bool = False
    expected_B_field_uT: float = 30.0  # Typical LEO value


def _compute_angle_cost_scale(
    ang_cost_func_type: int, 
    reference_angle_rad: float,
    is_quaternion_goal: bool
) -> float:
    """
    Compute the raw cost function value at a reference angle.
    
    This is used to normalize angle costs so that users can specify
    "cost at reference angle" regardless of the cost function type.
    
    Parameters
    ----------
    ang_cost_func_type : int
        Cost function type (0-4 for quaternion, 0-3 for vector).
    reference_angle_rad : float
        Reference angle in radians.
    is_quaternion_goal : bool
        True for quaternion (3-DOF) goals, False for vector (2-DOF) goals.
        
    Returns
    -------
    float
        The raw cost value at the reference angle.
    """
    theta = reference_angle_rad
    
    if is_quaternion_goal:
        # Quaternion goal: half-angle representation
        # For rotation θ, quaternion is [cos(θ/2), sin(θ/2)*axis]
        half_theta = theta / 2.0
        q_dot = np.cos(half_theta)  # q · q_goal = cos(θ/2)
        q_err_scalar = q_dot
        q_err_vec_norm = np.sin(half_theta)
        
        if ang_cost_func_type == 0:
            # 1 - |q·q_goal|
            return 1.0 - abs(q_dot)
        elif ang_cost_func_type == 1 or ang_cost_func_type == 3:
            # 0.5 * |q_err_vec|² (simplified quaternion error)
            return 0.5 * q_err_vec_norm ** 2
        elif ang_cost_func_type == 2:
            # 0.5 * |q_err_vec|² / q_err_scalar² ≈ 0.5 * tan²(θ/2)
            if abs(q_err_scalar) > 1e-10:
                return 0.5 * (q_err_vec_norm / q_err_scalar) ** 2
            else:
                return 1e6  # Very large for 180° rotation
        elif ang_cost_func_type == 4:
            # 1 - (q·q_goal)²
            return 1.0 - q_dot ** 2
        else:
            raise ValueError(f"Unknown quaternion cost function type: {ang_cost_func_type}")
    else:
        # Vector goal: uses dot product of aligned vectors
        cos_theta = np.cos(theta)
        
        if ang_cost_func_type == 0:
            # 1 - cos(θ)
            return 1.0 - cos_theta
        elif ang_cost_func_type == 1:
            # 0.5 * (1 - cos(θ))²
            return 0.5 * (1.0 - cos_theta) ** 2
        elif ang_cost_func_type == 2:
            # acos(cos(θ)) = θ (geodesic)
            return theta
        elif ang_cost_func_type == 3:
            # 0.5 * θ²
            return 0.5 * theta ** 2
        else:
            raise ValueError(f"Unknown vector cost function type: {ang_cost_func_type}")


@dataclass
class NormalizedStateCosts:
    """
    State costs in normalized space.
    
    Two modes are available:
    
    1. **Legacy mode** (use_scale_normalization=False, default):
       Costs are raw weights applied directly to states in radians/rad/s.
       - angle_cost: weight on attitude error (radians)
       - ang_vel_cost: weight on angular velocity (rad/s)
    
    2. **Scale-normalized mode** (use_scale_normalization=True):
       Costs are specified per reference scale for better conditioning.
       States are internally normalized: J = weight * (state / scale)²
       This improves Hessian conditioning when states have different magnitudes.
    
    Parameters
    ----------
    angle_cost : float
        Running attitude error weight. In legacy mode, this is the raw weight.
        In normalized mode, this is the cost when error equals angle_scale_deg.
    angle_terminal_cost : float
        Terminal attitude error weight or multiplier (see use_scale_normalization).
    ang_vel_cost : float
        Running angular velocity weight.
    ang_vel_terminal_cost : float
        Terminal angular velocity weight or multiplier.
    ang_cost_func_type : int
        Attitude cost function type:
        
        **Vector goals (2-DOF pointing):**
        - 0: 1-cos(θ) - smooth near zero, range [0, 2]
        - 1: 0.5*(1-cos(θ))² - smoother, range [0, 2]
        - 2: θ (geodesic) - linear in angle, range [0, π]
        - 3: 0.5*θ² - quadratic geodesic, range [0, π²/2]
        
        **Quaternion goals (3-DOF attitude):**
        - 0: 1-|q·q_goal| - range [0, 1]
        - 1: 0.5*|q_err_vec|² - quaternion error, range [0, 0.5]
        - 2: 0.5*tan²(θ/2) - scaled by half-angle, range [0, ∞)
        - 3: same as 1
        - 4: 1-(q·q_goal)² - range [0, 1]
        
    use_scale_normalization : bool
        If True, use scale-normalized costs for better conditioning.
        If False (default), use legacy raw weights.
    auto_scale_angle_cost : bool
        If True, automatically scale angle costs based on the cost function type
        and goal type so that the specified cost is achieved at the reference angle.
        This ensures consistent behavior across different cost functions.
    angle_scale_deg : float
        Reference angle in degrees for normalization (only used if use_scale_normalization=True).
    ang_vel_scale_deg_s : float
        Reference angular velocity in deg/s for normalization.
    """
    # Cost weights (interpretation depends on use_scale_normalization)
    angle_cost: float = 100.0
    angle_terminal_cost: float = 1000.0
    ang_vel_cost: float = 100.0
    ang_vel_terminal_cost: float = 1000.0
    
    # Cost function type
    ang_cost_func_type: int = 2  # acos geodesic (dissertation default)
    
    # Scale normalization (optional, off by default)
    use_scale_normalization: bool = False
    auto_scale_angle_cost: bool = False  # Auto-scale based on cost function type
    angle_scale_deg: float = 90.0  # Reference angle in degrees
    ang_vel_scale_deg_s: float = 10.0  # Reference angular velocity in deg/s
    
    @property
    def angle_scale_rad(self) -> float:
        """Angle scale in radians."""
        return self.angle_scale_deg * np.pi / 180.0
    
    @property
    def ang_vel_scale_rad_s(self) -> float:
        """Angular velocity scale in rad/s."""
        return self.ang_vel_scale_deg_s * np.pi / 180.0
    
    def get_cost_scaling_info(self) -> dict:
        """
        Get information about how angle costs scale for different goal types.
        
        Returns a dictionary with the raw cost values at the reference angle
        for both vector and quaternion goals, and the scaling factors that
        would be applied when auto_scale_angle_cost=True.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'ref_angle_deg': Reference angle in degrees
            - 'cost_func_type': The cost function type
            - 'vector_raw_cost': Raw cost at ref angle for vector goals
            - 'vector_scale_factor': Scaling factor for vector goals
            - 'quat_raw_cost': Raw cost at ref angle for quaternion goals
            - 'quat_scale_factor': Scaling factor for quaternion goals
        """
        ref_angle_rad = self.angle_scale_rad
        
        vec_cost = _compute_angle_cost_scale(self.ang_cost_func_type, ref_angle_rad, False)
        quat_cost = _compute_angle_cost_scale(self.ang_cost_func_type, ref_angle_rad, True)
        
        return {
            'ref_angle_deg': self.angle_scale_deg,
            'cost_func_type': self.ang_cost_func_type,
            'vector_raw_cost': vec_cost,
            'vector_scale_factor': 1.0 / vec_cost if vec_cost > 1e-10 else 1.0,
            'quat_raw_cost': quat_cost,
            'quat_scale_factor': 1.0 / quat_cost if quat_cost > 1e-10 else 1.0,
        }
    
    def to_raw_weights(
        self, 
        global_scale: float = 1.0,
        is_quaternion_goal: bool = None
    ) -> dict:
        """
        Convert to raw weights for C++ optimizer.
        
        In legacy mode: returns weights directly (scaled by global_scale).
        In normalized mode: returns weights divided by scale² so that
            cost = w_raw * state² = w_normalized * (state/scale)²
            
        Parameters
        ----------
        global_scale : float
            Global scaling factor applied to all weights.
        is_quaternion_goal : bool, optional
            If provided and auto_scale_angle_cost=True, scales angle costs
            so the specified cost is achieved at the reference angle regardless
            of the cost function type. True for 4D quaternion goals, False for
            3D vector goals.
        """
        # Compute angle cost scaling factor if auto-scaling enabled
        angle_scale_factor = 1.0
        if self.auto_scale_angle_cost and is_quaternion_goal is not None:
            # Get the raw cost value at the reference angle
            ref_angle_rad = self.angle_scale_rad
            raw_cost_at_ref = _compute_angle_cost_scale(
                self.ang_cost_func_type,
                ref_angle_rad,
                is_quaternion_goal
            )
            # Scale so that: w_raw * raw_cost_at_ref = user_specified_cost
            # Therefore: w_raw = user_specified_cost / raw_cost_at_ref
            if raw_cost_at_ref > 1e-10:
                angle_scale_factor = 1.0 / raw_cost_at_ref
            else:
                angle_scale_factor = 1.0
        
        if self.use_scale_normalization:
            # Scale-normalized mode
            angle_scale = self.angle_scale_rad
            ang_vel_scale = self.ang_vel_scale_rad_s
            
            # When using scale normalization, we divide by angle_scale²
            # When also using auto_scale, we additionally scale by the cost function
            return {
                'angle_weight': self.angle_cost * global_scale * angle_scale_factor / (angle_scale ** 2),
                'angle_weight_N': self.angle_terminal_cost * global_scale * angle_scale_factor / (angle_scale ** 2),
                'angvel_weight': self.ang_vel_cost * global_scale / (ang_vel_scale ** 2),
                'angvel_weight_N': self.ang_vel_terminal_cost * global_scale / (ang_vel_scale ** 2),
                'ang_cost_func_type': self.ang_cost_func_type,
            }
        else:
            # Legacy mode - direct weights (but still apply auto-scaling if enabled)
            return {
                'angle_weight': self.angle_cost * global_scale * angle_scale_factor,
                'angle_weight_N': self.angle_terminal_cost * global_scale * angle_scale_factor,
                'angvel_weight': self.ang_vel_cost * global_scale,
                'angvel_weight_N': self.ang_vel_terminal_cost * global_scale,
                'ang_cost_func_type': self.ang_cost_func_type,
            }


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
    
    def _get_reference_torque(
        self, 
        exclude_types: list = None,
        default: float = 0.004
    ) -> float:
        """
        Get the reference torque for cost scaling.
        
        Finds the maximum torque capability among all actuators (excluding
        specified types) to use as a reference for relative cost scaling.
        
        Parameters
        ----------
        exclude_types : list of str, optional
            Actuator type names to exclude (e.g., ['MTQ'] to exclude weak actuators).
        default : float, optional
            Default reference torque if no suitable actuators found. Default 4 mN·m.
            
        Returns
        -------
        float
            Reference torque in N·m.
        """
        if exclude_types is None:
            exclude_types = []
            
        margin = 1.0 - self.config.constraints.control_margin
        max_torque = default
        
        for actuator in self.satellite.actuators:
            type_name = type(actuator).__name__
            if type_name in exclude_types:
                continue
                
            # Use the actuator's estimate_torque_capability method
            try:
                tau = actuator.estimate_torque_capability(
                    expected_field_uT=self.config.actuator_costs.expected_B_field_uT
                ) * margin
                max_torque = max(max_torque, tau)
            except AttributeError:
                # Fallback for actuators without the method
                tau = actuator.u_max * margin
                max_torque = max(max_torque, tau)
        
        return max_torque
    
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
            m_max = mtq_actuators[0].u_max * (1.0 - self.config.constraints.control_margin)
            
            if act_costs.use_torque_effective_mtq_scaling:
                # Scale MTQ cost DOWN to account for its weak torque authority.
                # Since τ_mtq = m × B is tiny compared to RW torque, we make MTQ
                # "cheap" so the optimizer uses it freely at full capacity.
                #
                # Use the actuator's estimate_torque_capability() method for
                # generic handling of environment-dependent actuators.
                tau_mtq_max = mtq_actuators[0].estimate_torque_capability(
                    expected_field_uT=act_costs.expected_B_field_uT
                )
                
                # Get reference torque: max torque among all actuators
                tau_ref = self._get_reference_torque(
                    exclude_types=['MTQ'],  # Don't include MTQs in reference
                    default=0.004  # 4 mN·m fallback
                )
                
                # Scale factor: (τ_mtq/τ_ref)² makes MTQ cheaper when it's weaker
                weakness_ratio = (tau_mtq_max / tau_ref) ** 2
                raw['mtq_control_weight'] = act_costs.mtq_cost * g_ctrl * weakness_ratio / (m_max ** 2)
            else:
                # Default: cost on magnetic dipole moment (penalizes power draw)
                raw['mtq_control_weight'] = act_costs.mtq_cost * g_ctrl / (m_max ** 2)
        
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
        
        # State weights - use the new normalized interface
        state_raw = state_costs.to_raw_weights(global_scale=g_state)
        raw['angle_weight'] = state_raw['angle_weight']
        raw['angle_weight_N'] = state_raw['angle_weight_N']
        raw['angvel_weight'] = state_raw['angvel_weight']
        raw['angvel_weight_N'] = state_raw['angvel_weight_N']
        raw['ang_cost_func_type'] = state_raw['ang_cost_func_type']
        
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
    
    def estimate_state_hessian_condition(self) -> float:
        """
        Estimate the condition number of the state cost Hessian.
        
        With proper normalization, this should be close to 1.
        
        Returns
        -------
        float
            Estimated condition number of Qxx.
        """
        state_costs = self.config.state_costs
        raw = state_costs.to_raw_weights(global_scale=self.config.global_state_scale)
        
        # The raw weights are already scaled by 1/scale²
        # So the diagonal of Qxx is approximately [w_ang, w_ang, w_ang, w_av, w_av, w_av, ...]
        weights = [
            raw['angle_weight'],
            raw['angvel_weight'],
        ]
        
        # Add RW momentum weights if present
        rw_actuators = [a for a in self.satellite.actuators 
                       if type(a).__name__ == 'RW']
        if rw_actuators:
            act_costs = self.config.actuator_costs
            h_max = rw_actuators[0].h_max * self.config.constraints.rw_momentum_margin
            rw_h_weight = act_costs.rw_momentum_cost * self.config.global_control_scale / (h_max ** 2)
            weights.append(rw_h_weight)
        
        if len(weights) == 0:
            return 1.0
        
        return max(weights) / max(min(weights), 1e-10)
    
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
        
        state_costs = self.config.state_costs
        
        info = {
            'scaling': {
                'control_scales': scaling.control_scales,
            },
            'raw_weights': raw,
            'estimated_Quu_condition': self.estimate_control_hessian_condition(),
            'estimated_Qxx_condition': self.estimate_state_hessian_condition(),
            'num_actuators': len(self.satellite.actuators),
            'use_scale_normalization': state_costs.use_scale_normalization,
        }
        
        if state_costs.use_scale_normalization:
            info['scaling']['angle_scale_deg'] = state_costs.angle_scale_deg
            info['scaling']['ang_vel_scale_deg_s'] = state_costs.ang_vel_scale_deg_s
        
        return info


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
        Uses conservative angular velocity limits.
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
    def mtq_only_normalized() -> NormalizedPlannerConfig:
        """
        MTQ-only configuration with scale normalization for better conditioning.
        
        This is an experimental option that normalizes states by reference scales
        to improve Hessian conditioning.
        """
        return NormalizedPlannerConfig(
            actuator_costs=NormalizedActuatorCosts(
                mtq_cost=1.0,
            ),
            state_costs=NormalizedStateCosts(
                angle_cost=100.0,
                angle_terminal_cost=1000.0,  # 10x running
                ang_vel_cost=100.0,
                ang_vel_terminal_cost=1000.0,
                use_scale_normalization=True,
                angle_scale_deg=90.0,
                ang_vel_scale_deg_s=10.0,
            ),
            constraints=NormalizedConstraints(
                max_angular_velocity_deg_s=10.0,
                control_margin=0.25,
            ),
        )
    
    @staticmethod
    def mtq_plus_rw_normalized() -> NormalizedPlannerConfig:
        """
        MTQ+RW configuration with scale normalization for better conditioning.
        
        This is an experimental option that normalizes states by reference scales
        to improve Hessian conditioning.
        """
        return NormalizedPlannerConfig(
            actuator_costs=NormalizedActuatorCosts(
                mtq_cost=1.0,
                rw_torque_cost=5.0,
                rw_momentum_cost=10.0,
                rw_stiction_cost=0.1,
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
