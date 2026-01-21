"""
Configuration dataclass for TinyMPC tracking controller.

TinyMPC is a lightweight ADMM-based MPC solver used for real-time
trajectory tracking. It operates at higher frequency than ALTRO
for responsive disturbance rejection.
"""
from __future__ import annotations

__all__ = ["TinyMPCSettings"]

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TinyMPCSettings:
    """
    Configuration for TinyMPC tracking controller.

    TinyMPC uses ADMM (Alternating Direction Method of Multipliers) to
    efficiently solve tracking QP problems with actuator constraints.

    Attributes:
        max_iter: Maximum ADMM iterations per solve
        abs_tol: Absolute convergence tolerance for primal/dual residuals
        rel_tol: Relative convergence tolerance
        rho: ADMM penalty parameter (higher = faster but less accurate)
        rho_min: Minimum rho for adaptive scaling
        rho_max: Maximum rho for adaptive scaling
        adaptive_rho: Enable automatic rho adjustment based on residuals
        check_interval: Iterations between convergence checks

        track_horizon: MPC prediction horizon (5-10 steps recommended)
        track_dt: MPC timestep in seconds (typically matches dt_tvlqr)

        replan_enabled: Enable automatic re-planning triggers
        replan_attitude_threshold: Attitude error threshold to trigger replan (radians)
        replan_angvel_threshold: Angular velocity error threshold (rad/s)
        replan_min_interval: Minimum time between replans (seconds)

        verbose: Verbosity level (0=silent, 1=summary, 2=detailed)

    Example:
        # Default settings for responsive tracking
        settings = TinyMPCSettings()

        # Faster convergence, less accurate
        fast = TinyMPCSettings(max_iter=20, abs_tol=1e-3)

        # Higher accuracy, more compute
        accurate = TinyMPCSettings(max_iter=100, abs_tol=1e-6, rel_tol=1e-6)

        # Disable re-planning
        no_replan = TinyMPCSettings(replan_enabled=False)
    """

    # ADMM Solver Settings
    max_iter: int = 50
    abs_tol: float = 1e-4
    rel_tol: float = 1e-4
    rho: float = 1.0
    rho_min: float = 0.1
    rho_max: float = 10.0
    adaptive_rho: bool = True
    check_interval: int = 10

    # MPC Horizon Settings
    track_horizon: int = 10
    track_dt: float = 1.0

    # Re-planning Trigger Settings
    replan_enabled: bool = True
    replan_attitude_threshold: float = 10.0 * np.pi / 180  # 10 degrees
    replan_angvel_threshold: float = 5.0 * np.pi / 180     # 5 deg/s
    replan_min_interval: float = 10.0                       # seconds

    # Control Mode
    # use_altro_gains: If True, use ALTRO's pre-computed K gains with saturation.
    #                  This is NOT true MPC, just constrained TVLQR. Fast but limited.
    # use_true_mpc: If True, linearize about CURRENT state instead of reference.
    #               This can improve tracking when system has deviated from reference.
    #
    # Recommended settings:
    #   - True TinyMPC (default): use_altro_gains=False, use_true_mpc=False
    #     Linearizes at reference, computes fresh Riccati gains, runs ADMM
    #   - Adaptive TinyMPC: use_altro_gains=False, use_true_mpc=True
    #     Linearizes at current state for better far-from-reference performance
    #   - Fast constrained TVLQR: use_altro_gains=True
    #     Just saturates ALTRO's gains, no ADMM (fastest but not true MPC)
    use_altro_gains: bool = False  # Default: use actual TinyMPC algorithm
    use_true_mpc: bool = False     # Default: linearize at reference state

    # Verbosity
    verbose: int = 0

    def to_cpp_tuple(self) -> Tuple[int, float, float, float, float, float, bool, int, int, float, int]:
        """
        Convert to tuple for C++ TinyMPCSettings struct.

        Returns:
            Tuple matching C++ TinyMPCSettings field order:
            (max_iter, abs_tol, rel_tol, rho, rho_min, rho_max,
             adaptive_rho, check_interval, track_horizon, track_dt, verbose)
        """
        return (
            self.max_iter,
            self.abs_tol,
            self.rel_tol,
            self.rho,
            self.rho_min,
            self.rho_max,
            self.adaptive_rho,
            self.check_interval,
            self.track_horizon,
            self.track_dt,
            self.verbose
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/debugging."""
        return {
            "max_iter": self.max_iter,
            "abs_tol": self.abs_tol,
            "rel_tol": self.rel_tol,
            "rho": self.rho,
            "rho_min": self.rho_min,
            "rho_max": self.rho_max,
            "adaptive_rho": self.adaptive_rho,
            "check_interval": self.check_interval,
            "track_horizon": self.track_horizon,
            "track_dt": self.track_dt,
            "replan_enabled": self.replan_enabled,
            "replan_attitude_threshold": self.replan_attitude_threshold,
            "replan_angvel_threshold": self.replan_angvel_threshold,
            "replan_min_interval": self.replan_min_interval,
            "use_altro_gains": self.use_altro_gains,
            "verbose": self.verbose,
        }
