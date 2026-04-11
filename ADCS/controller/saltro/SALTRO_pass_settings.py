"""Pass-level SALTRO optimization settings.

This module defines Python configuration dataclasses for solver costs,
regularization, line search, and loop limits, with helpers to convert each
configuration block to the corresponding SALTRO C++ struct.
"""

from __future__ import annotations

__all__ = ["PassConfig"]

import numpy as np
from typing import Tuple, Optional, List
from numpy.typing import NDArray
from dataclasses import dataclass, field

def _get_saltro_py():
    """Import and return the ``saltro_py`` binding module.

    The loader appends ``SALTRO/build`` to ``sys.path`` if needed.

    :return: Imported ``saltro_py`` module.
    :rtype: module
    :raises ImportError: If the SALTRO Python extension is not available.
    """
    import os
    import sys

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    build_dir = os.path.join(parent_dir, "SALTRO", "build")
    if build_dir not in sys.path:
        sys.path.append(build_dir)

    try:
        import saltro_py
    except ImportError as exc:
        raise ImportError(f"saltro_py not available (expected in {build_dir})") from exc

    return saltro_py

@dataclass
class CostConfig:
    """Running and terminal cost weights for one optimization pass.

    :param angle: Running attitude error cost weight.
    :type angle: float
    :param ang_vel: Running angular velocity error cost weight.
    :type ang_vel: float
    :param control_mult: Global multiplier for actuator effort costs.
    :type control_mult: float
    :param angle_N: Terminal attitude error cost weight.
    :type angle_N: float
    :param ang_vel_N: Terminal angular velocity error cost weight.
    :type ang_vel_N: float
    """

    # Running costs
    angle: float = 1e2
    ang_vel: float = 1e5
    ang_vel_mag: float = 0.0
    ang_vel_err_dir: float = 0.0
    control_mult: float = 1.0

    # Actuator Weights
    mtq_control_weight: float = 1.0
    rw_control_weight: float = 1.0
    magic_control_weight: float = 0.0
    rw_AM_weight: float = 0.0
    rw_stic_weight: float = 0.0
    RWh_max_mult: float = 0.0
    RWh_stiction_mult: float = 0.0
    RWh_ok_mult: float = 0.0

    # Terminal costs
    angle_N: float = 0.0
    ang_vel_N: float = 0.0
    ang_vel_mag_N: float = 0.0
    ang_vel_err_dir_N: float = 0.0

    # Flags
    ang_cost_func_type: int = 2
    use_cost_hess: int = 1

    def to_cpp(self):
        """Convert Python costs to SALTRO C++ ``CostConfig``.

        :return: C++ cost config object.
        :rtype: Any
        """
        saltro_py = _get_saltro_py()
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
    """Augmented-Lagrangian outer-loop settings.

    :param max_outer_iters: Maximum number of outer iterations.
    :type max_outer_iters: int
    :param penalty_init: Initial constraint penalty.
    :type penalty_init: float
    :param penalty_max: Maximum constraint penalty.
    :type penalty_max: float
    :param constraint_tol: Constraint satisfaction tolerance.
    :type constraint_tol: float
    """

    max_outer_iters: int = 30

    lag_mult_init: float = 0.0
    lag_mult_max: float = 1e20

    penalty_init: float = 1e-1
    penalty_max: float = 1e16
    penalty_scale: float = 10.0
    
    constraint_tol: float = 1e-3
    total_cost_tol: float = 1e-2

    def to_cpp(self):
        """Convert Python settings to SALTRO C++ ``AugLagConfig``.

        :return: C++ augmented-Lagrangian config object.
        :rtype: Any
        """
        saltro_py = _get_saltro_py()
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
    """iLQR middle-loop settings.

    :param max_iters: Maximum iLQR iterations.
    :type max_iters: int
    :param grad_tol: Gradient norm tolerance.
    :type grad_tol: float
    :param cost_tol: Relative/absolute cost change tolerance.
    :type cost_tol: float
    """

    max_iters: int = 20
    grad_tol: float = 1e-3
    cost_tol: float = 1e-5
    z_count_lim: int = 10

    max_cost: float = 1e40
    state_bound: float = 10.0

    def to_cpp(self):
        """Convert Python settings to SALTRO C++ ``ILQRConfig``.

        :return: C++ iLQR config object.
        :rtype: Any
        """
        saltro_py = _get_saltro_py()
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
    """Regularization settings for iLQR backward passes.

    :param reg_init: Initial regularization value.
    :type reg_init: float
    :param reg_min: Lower bound on regularization.
    :type reg_min: float
    :param reg_max: Upper bound on regularization.
    :type reg_max: float
    """

    reg_init: float = 1e-6
    reg_min: float = 1e-8
    reg_max: float = 1e10
    reg_scale: float = 10.0
    reg_bump: float = 10.0

    reg_min_cond: int = 2
    rand_add_ratio: float = 0.0

    use_dynamics_hess: int = 0
    use_constraint_hess: int = 0

    def to_cpp(self):
        """Convert Python settings to SALTRO C++ ``RegularizationConfig``.

        :return: C++ regularization config object.
        :rtype: Any
        """
        saltro_py = _get_saltro_py()
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
    """Line-search settings for iLQR forward rollout.

    :param max_iters: Maximum line-search iterations.
    :type max_iters: int
    :param beta1: Minimum step scaling factor.
    :type beta1: float
    :param beta2: Maximum step scaling factor.
    :type beta2: float
    """

    max_iters: int = 24
    beta1: float = 1e-10
    beta2: float = 5000.0

    def to_cpp(self):
        """Convert Python settings to SALTRO C++ ``LineSearchConfig``.

        :return: C++ line-search config object.
        :rtype: Any
        """
        saltro_py = _get_saltro_py()
        cpp_ls = saltro_py.LineSearchConfig()
        cpp_ls.max_iters = self.max_iters
        cpp_ls.beta1 = self.beta1
        cpp_ls.beta2 = self.beta2
        return cpp_ls

@dataclass
class PassConfig:
    """Composite settings for one SALTRO optimization pass.

    A pass bundles cost, augmented-Lagrangian, iLQR, regularization, and
    line-search configuration with a fixed planning timestep ``dt``.

    :param cost: Cost-function configuration.
    :type cost: :class:`CostConfig`
    :param aug_lag: Augmented-Lagrangian outer-loop configuration.
    :type aug_lag: :class:`AugLagConfig`
    :param ilqr: iLQR middle-loop configuration.
    :type ilqr: :class:`ILQRConfig`
    :param reg: Regularization configuration.
    :type reg: :class:`RegularizationConfig`
    :param linesearch: Line-search configuration.
    :type linesearch: :class:`LineSearchConfig`
    :param dt: Planner timestep in seconds.
    :type dt: float
    """

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
    dt: float = 5.0

    def to_cpp(self):
        """Convert Python settings to SALTRO C++ ``PassConfig``.

        :return: C++ pass config object.
        :rtype: Any
        """
        saltro_py = _get_saltro_py()
        cpp_pass = saltro_py.PassConfig()
        cpp_pass.cost = self.cost.to_cpp()
        cpp_pass.auglag = self.aug_lag.to_cpp()
        cpp_pass.ilqr = self.ilqr.to_cpp()
        cpp_pass.reg = self.reg.to_cpp()
        cpp_pass.linesearch = self.linesearch.to_cpp()
        cpp_pass.dt = self.dt
        return cpp_pass


