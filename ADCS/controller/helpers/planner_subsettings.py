"""
Configuration dataclasses for ALTRO trajectory planner.

This module defines the configuration structures for the Augmented Lagrangian iLQR
optimizer, including solver settings, cost weights, and convergence criteria.
"""

__all__ = ["LineSearchConfig", "AugLagConfig", "RegularizationConfig", "ConvergenceConfig", "SolverPassConfig", "CostWeights", "InitTrajConfig"]

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class LineSearchConfig:
    """
    Configuration for backtracking line search in the forward pass.

    The line search finds a step size alpha that ensures sufficient cost decrease.
    It checks that the ratio z = (actual_decrease) / (expected_decrease) satisfies
    beta1 < z < beta2.

    Attributes:
        max_iters: Maximum line search iterations before giving up
        beta1: Lower bound for acceptable cost decrease ratio (prevents tiny steps)
        beta2: Upper bound for acceptable cost decrease ratio (prevents model trust issues)
    """
    max_iters: int = 20
    beta1: float = 1e-10
    beta2: float = 500.0

    def to_tuple(self) -> Tuple[int, float, float]:
        return (self.max_iters, self.beta1, self.beta2)
    
@dataclass
class AugLagConfig:
    """
    Configuration for Augmented Lagrangian constraint handling.

    The augmented Lagrangian method converts constrained optimization into a sequence
    of unconstrained problems by adding penalty terms: L_A = L + lambda*c + (mu/2)*c^2
    where c is the constraint violation.

    Attributes:
        lag_mult_init: Initial value for Lagrange multipliers (lambda)
        lag_mult_max: Maximum allowed Lagrange multiplier magnitude
        penalty_init: Initial penalty parameter (mu). Higher values enforce
            constraints more strictly but can cause ill-conditioning.
        penalty_max: Maximum penalty parameter
        penalty_scale: Factor to increase penalty when constraints not satisfied
    """
    lag_mult_init: float = 0.0
    lag_mult_max: float = 1e20
    penalty_init: float = 1e-1
    penalty_max: float = 1e16
    penalty_scale: float = 10.0

    def to_tuple(self) -> Tuple[float, float, float, float, float]:
        return (self.lag_mult_init, self.lag_mult_max, self.penalty_init, self.penalty_max, self.penalty_scale)
    
@dataclass
class RegularizationConfig:
    """
    Configuration for Levenberg-Marquardt style regularization in the backward pass.

    Regularization ensures the control Hessian Quu is positive definite, which is
    required for computing valid feedback gains. When Quu is ill-conditioned or
    indefinite, regularization adds rho*I to make it invertible.

    Attributes:
        reg_init: Initial regularization parameter (rho)
        reg_min: Minimum regularization (floor value)
        reg_max: Maximum regularization (triggers failure if exceeded)
        reg_scale: Factor to increase/decrease regularization adaptively
        reg_bump: Additional increase when line search fails
    """
    reg_init: float = 1e-2
    reg_min: float = 1e-8
    reg_max: float = 1e30
    reg_scale: float = 1.6
    reg_bump: float = 10

    # Conditional logic flags
    reg_min_cond: int = 2         # 1: Reg >= regMin, 0: Ignored
    rand_add_ratio: float = 0.0   # Random noise addition
    use_ev_magic: int = 0         # Use Eigendecomposition?
    spd_ev_reg: int = 1           # Regularize even if SPD?
    spd_ev_reg_all: int = 0       # Reg SPD by adding rho*I?
    rho_ev_reg_test: int = 1      # Test reset against rho?
    ev_reg_test_pre_abs: int = 1  # Test reset before abs?
    ev_add_reg: int = 0           # Add value vs clamp?
    ev_reg_is_rho: int = 1        # Clamp to rho vs regMin?
    ev_rho_add: int = 0           # Add to values < rho?

    use_dynamics_hess: int = 0
    use_constraint_hess: int = 0

    def to_tuple(self) -> Tuple[float, float, float, float, float, int, float, int, int, int, int, int, int, int, int, int, int]:
        return (self.reg_init, self.reg_min, self.reg_max, self.reg_scale, 
                self.reg_bump, self.reg_min_cond, self.rand_add_ratio, 
                self.use_ev_magic, self.spd_ev_reg, self.spd_ev_reg_all, 
                self.rho_ev_reg_test, self.ev_reg_test_pre_abs, self.ev_add_reg, 
                self.ev_reg_is_rho, self.ev_rho_add, self.use_dynamics_hess, 
                self.use_constraint_hess)
    
@dataclass
class ConvergenceConfig:
    """
    Configuration for convergence criteria and iteration limits.

    The optimizer has nested loops: outer loop (augmented Lagrangian updates) and
    inner loop (iLQR iterations). Convergence is declared when cost changes are
    small and constraints are satisfied.

    Attributes:
        max_outer_iter: Maximum augmented Lagrangian iterations
        max_inner_iter: Maximum iLQR iterations per outer iteration
        max_total_iter: Absolute maximum iterations across all loops
        grad_tol: Gradient norm tolerance for declaring convergence
        ilqr_cost_tol: Cost change tolerance for iLQR inner loop
        total_cost_tol: Cost change tolerance for outer loop convergence
        z_count_lim: Number of small-z iterations before giving up
        c_max: Maximum constraint violation for feasibility
        max_cost: Cost threshold for detecting divergence
        xmax_val: State magnitude threshold for detecting divergence
    """
    max_outer_iter: int = 30
    max_inner_iter: int = 250
    max_total_iter: int = 7000
    grad_tol: float = 1e-3
    ilqr_cost_tol: float = 1e-1
    total_cost_tol: float = 1e-2
    z_count_lim: int = 10
    c_max: float = 0.002
    max_cost: float = 1e40

    # State bound for divergence check
    xmax_val: float = 10.0

    def to_tuple(self, state_len) -> Tuple[int, int, int, float, float, float, int, float, float, np.ndarray]:
        xmax_vec = self.xmax_val * np.ones((state_len, 1))
        return (self.max_outer_iter, self.max_inner_iter, self.max_total_iter, 
                self.grad_tol, self.ilqr_cost_tol, self.total_cost_tol, 
                self.z_count_lim, self.c_max, self.max_cost, xmax_vec)

@dataclass
class SolverPassConfig:
    line_search: LineSearchConfig = field(default_factory=LineSearchConfig)
    aug_lag: AugLagConfig = field(default_factory=AugLagConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)

@dataclass
class CostWeights:
    """
    Cost function weights for the trajectory optimization problem.

    The cost function is: J = sum_k [state_cost + control_cost] + terminal_cost

    Terminal costs (*_N) should typically be higher than running costs to
    prioritize reaching the goal state.

    Attributes:
        angle: Running cost weight on attitude error
        ang_vel: Running cost on angular velocity squared: 0.5*w'*w*w_av
        control_mult: Multiplier for actuator-specific control costs
        control_mag: UNUSED - extracted but not applied in C++ cost function
        ang_vel_mag: Cost on ang vel component along B-field: |dot(w, R'*B)|
        ang_accel: Misnamed - actually cost on ang vel along attitude error
            direction: dot(w, cross(R'*e_goal, boresight))
        angle_N: Terminal attitude error cost weight
        ang_vel_N: Terminal angular velocity cost weight
        ang_vel_mag_N: Terminal cost on ang vel along B-field
        ang_accel_N: Terminal cost on ang vel along error direction
        ang_cost_func_type: Attitude cost formulation:
            0=(1-dot), 1=0.5*(1-dot)^2, 2=acos(dot), 3=0.5*acos(dot)^2
        use_raw_control_cost: If True, use control values directly in cost
        consider_vector_in_tvlqr: Flag for TVLQR vector tracking mode
    """
    angle: float = 1e3
    ang_vel: float = 1e4
    control_mult: float = 1.0
    control_mag: float = 0.0
    ang_vel_mag: float = 0.0
    ang_accel: float = 0.0

    # Terminal costs (should be >= running costs to prioritize reaching goal)
    angle_N: float = 1e4
    ang_vel_N: float = 1e5
    ang_vel_mag_N: float = 0.0
    ang_accel_N: float = 0.0

    # Flags
    # 0=(1-dot), 1=0.5*(1-dot)^2, 2=acos(dot), 3=0.5*acos(dot)^2
    ang_cost_func_type: int = 3
    use_raw_control_cost: bool = True
    consider_vector_in_tvlqr: int = 0 # Specifically for TVLQR pass

    def to_tuple(self, tracking_formulation=None):
        # The last arg is specific to TVLQR settings (tracking formulation)
        # For main/second pass, use use_raw_control_cost as the last arg usually
        # or the vector flag. The original code has slight variations.
        
        # Mapping for standard OptMainCostSettings
        base = (self.angle, self.ang_vel, self.control_mult, 
                self.control_mag, self.ang_vel_mag, self.ang_accel,
                self.angle_N, self.ang_vel_N, self.ang_vel_mag_N, self.ang_accel_N)
        
        if tracking_formulation is not None:
             # This matches optTVLQRCostSettings
            return base + (self.consider_vector_in_tvlqr, self.use_raw_control_cost, tracking_formulation)
        
        # This matches optMainCostSettings / optSecondCostSettings
        return base + (self.ang_cost_func_type, self.use_raw_control_cost)

@dataclass
class InitTrajConfig:
    # Settings for generating the initial guess
    bdot_gain: float = 1000.0
    hl_angle_limit: float = 10.0 * np.pi / 180.0
    
    # (gyro, damp, vel, quat, rand, umax)
    high_settings: tuple = (0, -2e0, 0, -0.005, 0.1, 0.5)
    low_settings: tuple = (0, -1e-4, 0, -0.00001, 0.1, 0.5)

    def to_tuple(self) -> Tuple[float, float, tuple, tuple]:
        return (self.bdot_gain, self.hl_angle_limit, self.high_settings, self.low_settings)