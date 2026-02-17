from __future__ import annotations

__all__ = ["PassConfig"]

import numpy as np
from typing import Tuple, Optional, List
from numpy.typing import NDArray
from dataclasses import dataclass, field

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

@dataclass
class ILQRConfig:
    max_iters: int = 250
    grad_tol: float = 1e-3
    cost_tol: float = 1e-1
    z_count_lim: int = 10

    max_cost: float = 1e40
    state_bound: float = 10.0

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

@dataclass
class LineSearchConfig:
    max_iters: int = 20
    beta1: float = 1e-10
    beta2: float = 500.0

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


