__all__ = ["LineSearchConfig", "AugLagConfig", "RegularizationConfig", "ConvergenceConfig", "SolverPassConfig", "CostWeights", "InitTrajConfig"]

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class LineSearchConfig:
    # Settings for the backtracking line search
    max_iters: int = 20
    beta1: float = 1e-10
    beta2: float = 20.0

    def to_tuple(self) -> Tuple[int, float, float]:
        return (self.max_iters, self.beta1, self.beta2)
    
@dataclass
class AugLagConfig:
    # Settings for the Augmented Lagrangian (constraint enforcement)
    lag_mult_init: float = 0.0
    lag_mult_max: float = 1e10
    penalty_init: float = 1.0
    penalty_max: float = 1e10
    penalty_scale: float = 10.0

    def to_tuple(self) -> Tuple[float, float, float, float, float]:
        return (self.lag_mult_init, self.lag_mult_max, self.penalty_init, self.penalty_max, self.penalty_scale)
    
@dataclass
class RegularizationConfig:
    # Settings for matrix regulatization (Levenberg-Marquardt)
    reg_init: float = 1e-10
    reg_min: float = 1e-10
    reg_max: float = 1e12
    reg_scale: float = 1.6
    reg_bump: float = 10.0

    # Conditional logic flags
    reg_min_cond: int = 1         # 1: Reg >= regMin, 0: Ignored
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
    # Settings for breaking loops (Inner/Outer iterations and tolerances)
    max_outer_iter: int = 25
    max_inner_iter: int = 250
    max_total_iter: int = 4500
    grad_tol: float = 1e-7
    ilqr_cost_tol: float = 1e-8
    total_cost_tol: float = 1e-9
    z_count_lim: int = 20
    c_max: float = 0.002
    max_cost: float = 1e10

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
    convergence: ConvergenceConfig = field(default=ConvergenceConfig)
    regularization: RegularizationConfig = field(default=RegularizationConfig)

@dataclass
class CostWeights:
    # Weights for the Q and R matrices
    angle: float = 10.0
    ang_vel: float = 100.0
    control_mult: float = 1.0
    control_mag: float = 0.0
    ang_vel_mag: float = 0.0
    ang_accel: float = 0.0

    # Terminal costs
    angle_N: float = 100.0
    ang_vel_N: float = 100.0
    ang_vel_mag_N: float = 0.0
    ang_accel_N: float = 0.0

    # Flags
    # 0=(1-dot), 1=0.5*(1-dot)^2, 2=acos(dot), 3=0.5*acos(dot)^2
    ang_cost_func_type: int = 0 
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
    bdot_gain: float = 1e7
    hl_angle_limit: float = 10.0 * np.pi / 180.0
    
    # (gyro, damp, vel, quat, rand, umax)
    high_settings: tuple = (0, -2000.0, -50.0, -2.0, 0.001, 1.5)
    low_settings: tuple = (0, -1000.0, -200.0, -0.001, 0.0, 1.5)

    def to_tuple(self) -> Tuple[float, float, tuple, tuple]:
        return (self.bdot_gain, self.hl_angle_limit, self.high_settings, self.low_settings)