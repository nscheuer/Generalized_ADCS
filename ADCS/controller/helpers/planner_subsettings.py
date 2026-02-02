"""
Configuration dataclasses for ALTRO trajectory planner.

This module defines the configuration structures for the Augmented Lagrangian iLQR
optimizer, including solver settings, cost weights, and convergence criteria.
"""
from __future__ import annotations

__all__ = ["LineSearchConfig", "AugLagConfig", "RegularizationConfig", "ConvergenceConfig", "SolverPassConfig", "CostWeights", "InitTrajConfig"]

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Union
from numpy.typing import NDArray

@dataclass
class LineSearchConfig:
    r"""
    Configuration for backtracking line search in the iLQR forward pass.

    During the forward rollout, a step size :math:`\alpha` is chosen to ensure
    sufficient decrease of the cost function. Let

    .. math::

        z = \frac{J_{\text{actual}} - J_{\text{new}}}{J_{\text{expected}}}

    where :math:`J_{\text{expected}}` is the quadratic model prediction.
    The step is accepted if

    .. math::

        \beta_1 < z < \beta_2

    which prevents both excessively small steps and steps that violate the
    local quadratic approximation.

    :param max_iters:
        Maximum number of line search iterations.

    :type max_iters:
        int

    :param beta1:
        Lower bound on acceptable cost decrease ratio.

    :type beta1:
        float

    :param beta2:
        Upper bound on acceptable cost decrease ratio.

    :type beta2:
        float

    """
    max_iters: int = 20
    beta1: float = 1e-10
    beta2: float = 500.0

    def to_tuple(self) -> Tuple[int, float, float]:
        return (self.max_iters, self.beta1, self.beta2)
    
@dataclass
class AugLagConfig:
    r"""
    Configuration for augmented Lagrangian constraint handling.

    The augmented Lagrangian formulation transforms a constrained problem

    .. math::

        \min_x J(x) \quad \text{s.t.} \quad c(x) = 0

    into an unconstrained sequence of problems by augmenting the cost:

    .. math::

        \mathcal{L}_A(x, \lambda, \mu) =
        J(x) + \lambda^\top c(x) + \frac{\mu}{2}\|c(x)\|^2

    where :math:`\lambda` are Lagrange multipliers and :math:`\mu` is the
    penalty parameter.

    :param lag_mult_init:
        Initial Lagrange multiplier value.

    :type lag_mult_init:
        float

    :param lag_mult_max:
        Maximum allowed magnitude of Lagrange multipliers.

    :type lag_mult_max:
        float

    :param penalty_init:
        Initial penalty parameter.

    :type penalty_init:
        float

    :param penalty_max:
        Maximum penalty parameter.

    :type penalty_max:
        float

    :param penalty_scale:
        Multiplicative factor used to increase the penalty.

    :type penalty_scale:
        float

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
    r"""
    Configuration for Levenberg–Marquardt regularization in the backward pass.

    To compute feedback gains, the control Hessian :math:`\mathbf{Q}_{uu}`
    must be positive definite. When it is ill-conditioned or indefinite,
    regularization modifies it as

    .. math::

        \mathbf{Q}_{uu}^{\text{reg}} =
        \mathbf{Q}_{uu} + \rho \mathbf{I}

    where :math:`\rho` is adapted during optimization.

    :param reg_init:
        Initial regularization parameter.

    :type reg_init:
        float

    :param reg_min:
        Minimum allowable regularization value.

    :type reg_min:
        float

    :param reg_max:
        Maximum allowable regularization value.

    :type reg_max:
        float

    :param reg_scale:
        Factor used to scale regularization up or down.

    :type reg_scale:
        float

    :param reg_bump:
        Additional increase applied after failed line search.

    :type reg_bump:
        float

    :param reg_min_cond:
        Condition flag controlling minimum regularization enforcement.

    :type reg_min_cond:
        int

    :param rand_add_ratio:
        Ratio of random noise added to escape local minima.

    :type rand_add_ratio:
        float

    :param use_dynamics_hess:
        Flag enabling second-order dynamics terms.

    :type use_dynamics_hess:
        int

    :param use_constraint_hess:
        Flag enabling second-order constraint terms.

    :type use_constraint_hess:
        int

    """
    reg_init: float = 1e-2
    reg_min: float = 1e-8
    reg_max: float = 1e30
    reg_scale: float = 1.6
    reg_bump: float = 10.0
    reg_min_cond: int = 2
    rand_add_ratio: float = 0.0
    use_dynamics_hess: int = 0
    use_constraint_hess: int = 0

    def to_tuple(self) -> Tuple[float, float, float, float, float, int, float, int, int]:
        return (self.reg_init, self.reg_min, self.reg_max, self.reg_scale,
                self.reg_bump, self.reg_min_cond, self.rand_add_ratio,
                self.use_dynamics_hess, self.use_constraint_hess)
    
@dataclass
class ConvergenceConfig:
    r"""
    Configuration for convergence criteria and iteration limits.

    The optimization uses nested iterations:

    - Outer loop for augmented Lagrangian updates
    - Inner loop for iLQR refinement

    Convergence is declared when cost reduction and constraint violations
    satisfy specified tolerances.

    :param max_outer_iter:
        Maximum number of augmented Lagrangian iterations.

    :type max_outer_iter:
        int

    :param max_inner_iter:
        Maximum number of iLQR iterations per outer loop.

    :type max_inner_iter:
        int

    :param max_total_iter:
        Absolute maximum number of iterations.

    :type max_total_iter:
        int

    :param grad_tol:
        Gradient norm tolerance.

    :type grad_tol:
        float

    :param ilqr_cost_tol:
        Cost change tolerance for inner loop convergence.

    :type ilqr_cost_tol:
        float

    :param total_cost_tol:
        Cost change tolerance for outer loop convergence.

    :type total_cost_tol:
        float

    :param z_count_lim:
        Maximum number of consecutive small-decrease iterations.

    :type z_count_lim:
        int

    :param c_max:
        Maximum allowable constraint violation.

    :type c_max:
        float

    :param max_cost:
        Cost threshold for divergence detection.

    :type max_cost:
        float

    :param xmax_val:
        State magnitude bound for divergence detection.

    :type xmax_val:
        float

    """
    max_outer_iter: int = 30
    max_inner_iter: int = 250
    max_total_iter: int = 7000
    grad_tol: float = 1e-4
    ilqr_cost_tol: float = 1e-2
    total_cost_tol: float = 1e-4
    z_count_lim: int = 10
    c_max: float = 0.0002
    max_cost: float = 1e40

    # State bound for divergence check
    xmax_val: float = 10.0

    def to_tuple(self, state_len: int) -> Tuple[int, int, int, float, float, float, int, float, float, NDArray[np.float64]]:
        xmax_vec = self.xmax_val * np.ones((state_len, 1))
        return (self.max_outer_iter, self.max_inner_iter, self.max_total_iter, 
                self.grad_tol, self.ilqr_cost_tol, self.total_cost_tol, 
                self.z_count_lim, self.c_max, self.max_cost, xmax_vec)

@dataclass
class SolverPassConfig:
    r"""
    Aggregate configuration for a single optimization pass.

    This class bundles line search, augmented Lagrangian, convergence,
    and regularization settings into a single structure used by the solver.

    :param line_search:
        Line search configuration.

    :type line_search:
        :class:`~ADCS.controller.helpers.planner_subsettings.LineSearchConfig`

    :param aug_lag:
        Augmented Lagrangian configuration.

    :type aug_lag:
        :class:`~ADCS.controller.helpers.planner_subsettings.AugLagConfig`

    :param convergence:
        Convergence configuration.

    :type convergence:
        :class:`~ADCS.controller.helpers.planner_subsettings.ConvergenceConfig`

    :param regularization:
        Regularization configuration.

    :type regularization:
        :class:`~ADCS.controller.helpers.planner_subsettings.RegularizationConfig`

    """
    line_search: LineSearchConfig = field(default_factory=LineSearchConfig)
    aug_lag: AugLagConfig = field(default_factory=AugLagConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)

@dataclass
class CostWeights:
    r"""
    Cost function weights for trajectory optimization.

    The discrete-time cost function minimized by the planner is

    .. math::

        J = \sum_{k=0}^{N-1}
        \left(
        \mathbf{x}_k^\top \mathbf{Q} \mathbf{x}_k +
        \mathbf{u}_k^\top \mathbf{R} \mathbf{u}_k
        \right)
        + \mathbf{x}_N^\top \mathbf{Q}_N \mathbf{x}_N

    where running and terminal costs may differ.

    :param angle:
        Running attitude error weight.

    :type angle:
        float

    :param ang_vel:
        Running angular velocity weight.

    :type ang_vel:
        float

    :param control_mult:
        Multiplier applied to actuator control costs.

    :type control_mult:
        float

    :param ang_vel_mag:
        Running cost on angular velocity projected onto magnetic field.

    :type ang_vel_mag:
        float

    :param ang_vel_err_dir:
        Running cost on angular velocity along error direction.

    :type ang_vel_err_dir:
        float

    :param angle_N:
        Terminal attitude error weight.

    :type angle_N:
        float

    :param ang_vel_N:
        Terminal angular velocity weight.

    :type ang_vel_N:
        float

    :param ang_vel_mag_N:
        Terminal magnetic-field angular velocity weight.

    :type ang_vel_mag_N:
        float

    :param ang_vel_err_dir_N:
        Terminal angular velocity error-direction weight.

    :type ang_vel_err_dir_N:
        float

    :param ang_cost_func_type:
        Attitude cost formulation selector.

    :type ang_cost_func_type:
        int

    :param use_raw_control_cost:
        Flag selecting raw or scaled control costs.

    :type use_raw_control_cost:
        bool

    :param consider_vector_in_tvlqr:
        Flag enabling vector tracking in TVLQR.

    :type consider_vector_in_tvlqr:
        int

    :param use_full_cost_hessian:
        Flag enabling full Newton cost Hessian.

    :type use_full_cost_hessian:
        bool

    """
    angle: float = 1e3
    ang_vel: float = 1e4
    control_mult: float = 1.0
    ang_vel_mag: float = 0.0
    ang_vel_err_dir: float = 0.0

    # Terminal costs (should be >= running costs if prioritizing reaching goal, should be = if all steps matter equally.)
    angle_N: float = 1e4
    ang_vel_N: float = 1e5
    ang_vel_mag_N: float = 0.0
    ang_vel_err_dir_N: float = 0.0

    # Flags
    # Attitude cost formulations:
    #   0 = (1 - q·q_goal)         - Linear in dot product, cheap but non-smooth at 180°
    #   1 = 0.5*(1 - q·q_goal)²    - Quadratic in dot product, smooth but weak gradient near goal
    #   2 = acos(|q·q_goal|)       - Geodesic angle (radians), true rotation distance [RECOMMENDED]
    #   3 = 0.5*acos(|q·q_goal|)²  - Quadratic geodesic, strong near goal but may over-penalize large errors
    ang_cost_func_type: int = 2  # Default to geodesic angle (type 2) for balanced behavior
    use_raw_control_cost: bool = True
    consider_vector_in_tvlqr: int = 0

    # Hessian computation mode:
    #   False (0): Gauss-Newton approximation - always PSD, numerically stable
    #   True (1):  Full Newton - includes second derivative terms (dphi*ddphi for angle,
    #              smoothstep*smoothstep'' for stiction), may be indefinite but can
    #              converge faster when the problem is well-behaved
    use_full_cost_hessian: bool = False

    def to_tuple(self, tracking_formulation: int | None = None) -> Tuple[float, ...]:
        """Convert to tuple for C++ interface.

        Args:
            tracking_formulation: If provided, returns TVLQR cost settings format.
                                  If None, returns main/second pass format.
        """
        # Base tuple: running costs then terminal costs
        # Note: control_mag was removed as it was unused in C++
        base = (self.angle, self.ang_vel, self.control_mult,
                self.ang_vel_mag, self.ang_vel_err_dir,
                self.angle_N, self.ang_vel_N, self.ang_vel_mag_N, self.ang_vel_err_dir_N)

        if tracking_formulation is not None:
            # TVLQR cost settings format - C++ expects int for last 3 args
            return base + (int(self.consider_vector_in_tvlqr), int(self.use_raw_control_cost), int(tracking_formulation))

        # Main/second pass cost settings format - C++ expects int
        # Format: (9 cost weights, ang_cost_func_type, use_raw_control_cost, use_full_cost_hessian)
        return base + (int(self.ang_cost_func_type), int(self.use_raw_control_cost), int(self.use_full_cost_hessian))

@dataclass
class InitTrajConfig:
    r"""
    Configuration for initial trajectory generation.

    The initial guess trajectory may use B-dot detumbling and heuristic
    saturation limits to provide a dynamically feasible starting point
    for optimization.

    :param bdot_gain:
        Gain used for B-dot detumbling initialization.

    :type bdot_gain:
        float

    :param hl_angle_limit:
        High-level angle threshold for switching strategies.

    :type hl_angle_limit:
        float

    :param high_settings:
        Tuple defining high-level initialization parameters.

    :type high_settings:
        tuple

    :param low_settings:
        Tuple defining low-level initialization parameters.

    :type low_settings:
        tuple

    """
    bdot_gain: float = 1000.0
    hl_angle_limit: float = 10.0 * np.pi / 180.0
    
    # (gyro, damp, vel, quat, rand, umax)
    high_settings: tuple = (0, -2e0, 0, -0.005, 0.1, 0.5)
    low_settings: tuple = (0, -1e-4, 0, -0.00001, 0.1, 0.5)

    def to_tuple(self) -> Tuple[float, float, tuple, tuple]:
        return (self.bdot_gain, self.hl_angle_limit, self.high_settings, self.low_settings)