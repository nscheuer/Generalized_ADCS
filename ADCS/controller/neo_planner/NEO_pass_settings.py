from __future__ import annotations

__all__ = ["PassConfig"]

import numpy as np
from typing import Tuple, Optional, List
from numpy.typing import NDArray
from dataclasses import dataclass, field

try:
    import sys
    import os
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    build_dir = os.path.join(parent_dir, "SALTRO", "build")
    if build_dir not in sys.path:
        sys.path.append(build_dir)
    import saltro_py
except ImportError:
    saltro_py = None

@dataclass
class CostConfig:
    # Running costs
    angle: float = 1e3
    ang_vel: float = 1e4
    ang_vel_mag: float = 0.0
    ang_vel_err_dir: float = 0.0
    control_mult: float = 1.0

    # Actuator Weights
    mtq_control_weight: float = 1e3
    rw_control_weight: float = 1e8
    magic_control_weight: float = 0.0001
    rw_AM_weight: float = 1e4
    rw_stic_weight: float = 1e0
    RWh_max_mult: float = 0.8
    RWh_stiction_mult: float = 0.01
    RWh_ok_mult: float = 0.5

    # Terminal costs
    angle_N: float = 1e4
    ang_vel_N: float = 1e5
    ang_vel_mag_N: float = 0.0
    ang_vel_err_dir_N: float = 0.0

    # Flags
    ang_cost_func_type: int = 2
    use_cost_hess: int = 0

    def to_cpp(self):
        """Convert to C++ CostConfig"""
        if saltro_py is None:
            raise ImportError("saltro_py not available")
        cpp_cost = saltro_py.CostConfig()
        cpp_cost.angle = self.angle
        cpp_cost.ang_vel = self.ang_vel
        cpp_cost.ang_vel_mag = self.ang_vel_mag
        cpp_cost.ang_vel_err_dir = self.ang_vel_err_dir
        cpp_cost.control_mult = self.control_mult
        cpp_cost.mtq_control_weight = self.mtq_control_weight
        cpp_cost.rw_control_weight = self.rw_control_weight
        cpp_cost.magic_control_weight = self.magic_control_weight
        cpp_cost.rw_AM_weight = self.rw_AM_weight
        cpp_cost.rw_stic_weight = self.rw_stic_weight
        cpp_cost.RWh_max_mult = self.RWh_max_mult
        cpp_cost.RWh_stiction_mult = self.RWh_stiction_mult
        cpp_cost.RWh_ok_mult = self.RWh_ok_mult
        cpp_cost.angle_N = self.angle_N
        cpp_cost.ang_vel_N = self.ang_vel_N
        cpp_cost.ang_vel_mag_N = self.ang_vel_mag_N
        cpp_cost.ang_vel_err_dir_N = self.ang_vel_err_dir_N
        cpp_cost.ang_cost_func_type = self.ang_cost_func_type
        cpp_cost.use_cost_hess = bool(self.use_cost_hess)
        return cpp_cost


@dataclass
class AugLagConfig:
    max_outer_iters: int = 30

    lag_mult_init: float = 0.0
    lag_mult_max: float = 1e20

    penalty_init: float = 1e-1
    penalty_max: float = 1e16
    penalty_scale: float = 10.0
    
    constraint_tol: float = 0.002
    total_cost_tol: float = 1e-2

    def to_cpp(self):
        """Convert to C++ AugLagConfig"""
        if saltro_py is None:
            raise ImportError("saltro_py not available")
        cpp_auglag = saltro_py.AugLagConfig()
        cpp_auglag.max_outer_iters = self.max_outer_iters
        cpp_auglag.lag_mult_init = self.lag_mult_init
        cpp_auglag.lag_mult_max = self.lag_mult_max
        cpp_auglag.penalty_init = self.penalty_init
        cpp_auglag.penalty_max = self.penalty_max
        cpp_auglag.penalty_scale = self.penalty_scale
        cpp_auglag.constraint_tol = self.constraint_tol
        cpp_auglag.total_cost_tol = self.total_cost_tol
        return cpp_auglag

@dataclass
class ILQRConfig:
    max_iters: int = 250
    grad_tol: float = 1e-3
    cost_tol: float = 1e-1
    z_count_lim: int = 10

    max_cost: float = 1e40
    state_bound: float = 10.0

    def to_cpp(self):
        """Convert to C++ ILQRConfig"""
        if saltro_py is None:
            raise ImportError("saltro_py not available")
        cpp_ilqr = saltro_py.ILQRConfig()
        cpp_ilqr.max_iters = self.max_iters
        cpp_ilqr.grad_tol = self.grad_tol
        cpp_ilqr.cost_tol = self.cost_tol
        cpp_ilqr.z_count_lim = self.z_count_lim
        cpp_ilqr.max_cost = self.max_cost
        cpp_ilqr.state_bound = self.state_bound
        return cpp_ilqr

@dataclass
class RegularizationConfig:
    reg_init: float = 1e-2
    reg_min: float = 1e-8
    reg_max: float = 1e30
    reg_scale: float = 1.6
    reg_bump: float = 10.0

    reg_min_cond: int = 2
    rand_add_ratio: float = 0.0

    use_dynamics_hess: int = 0
    use_constraint_hess: int = 0

    def to_cpp(self):
        """Convert to C++ RegularizationConfig"""
        if saltro_py is None:
            raise ImportError("saltro_py not available")
        cpp_reg = saltro_py.RegularizationConfig()
        cpp_reg.reg_init = self.reg_init
        cpp_reg.reg_min = self.reg_min
        cpp_reg.reg_max = self.reg_max
        cpp_reg.reg_scale = self.reg_scale
        cpp_reg.reg_bump = self.reg_bump
        cpp_reg.reg_min_cond = self.reg_min_cond
        cpp_reg.rand_add_ratio = self.rand_add_ratio
        cpp_reg.use_dynamics_hess = bool(self.use_dynamics_hess)
        cpp_reg.use_constraint_hess = bool(self.use_constraint_hess)
        return cpp_reg

@dataclass
class LineSearchConfig:
    max_iters: int = 20
    beta1: float = 1e-10
    beta2: float = 500.0

    def to_cpp(self):
        """Convert to C++ LineSearchConfig"""
        if saltro_py is None:
            raise ImportError("saltro_py not available")
        cpp_ls = saltro_py.LineSearchConfig()
        cpp_ls.max_iters = self.max_iters
        cpp_ls.beta1 = self.beta1
        cpp_ls.beta2 = self.beta2
        return cpp_ls

@dataclass
class PassConfig:
    # Cost Function
    cost: CostConfig = field(default_factory=CostConfig)

    # Outer Loop
    aug_lag: AugLagConfig = field(default_factory=AugLagConfig)

    # Middle Loop
    ilqr: ILQRConfig = field(default_factory=ILQRConfig)
    reg: RegularizationConfig = field(default_factory=RegularizationConfig)

    # Inner Loop
    linesearch: LineSearchConfig = field(default_factory=LineSearchConfig)

    # Timestep
    dt: float = 1.0

    def to_cpp(self):
        """Convert to C++ PassConfig"""
        if saltro_py is None:
            raise ImportError("saltro_py not available")
        cpp_pass = saltro_py.PassConfig()
        cpp_pass.cost = self.cost.to_cpp()
        cpp_pass.auglag = self.aug_lag.to_cpp()
        cpp_pass.ilqr = self.ilqr.to_cpp()
        cpp_pass.reg = self.reg.to_cpp()
        cpp_pass.linesearch = self.linesearch.to_cpp()
        cpp_pass.dt = self.dt
        return cpp_pass


