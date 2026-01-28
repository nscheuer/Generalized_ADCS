"""
Fast Planner Tuning Experiment - 3MTQ + 1RW Slew

Designed for rapid iteration on CostWeights and PlannerSettings.
Uses simpler orbit and fewer iterations for speed.

Usage:
    python fast_slew_experiment.py                    # Single run with defaults
    python fast_slew_experiment.py --sweep angle      # Sweep angle costs
    python fast_slew_experiment.py --sweep all        # All sweeps
"""

import sys
import os
import time
import argparse
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, List

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.CONOPS.goals import Fixed_Attitude_Goal, Nadir_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings
from ADCS.controller.helpers.planner_subsettings import CostWeights, ConvergenceConfig, AugLagConfig, SolverPassConfig
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize, quat_mult, quat_diff


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single planner experiment."""
    slew_angle_deg: float = 45.0
    slew_axis: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    duration_sec: float = 60.0
    dt_planning: float = 2.0
    
    # Solver iterations (keep low for speed)
    pass1_outer: int = 4
    pass1_inner: int = 15
    pass2_outer: int = 2
    pass2_inner: int = 8
    
    # Cost weights
    angle: float = 1e3
    angle_N: float = 1e4
    ang_vel: float = 1e3
    ang_vel_N: float = 1e4
    control_mult: float = 1.0
    ang_cost_func_type: int = 2
    
    use_quaternion_goal: bool = True
    seed: int = 42
    w0_scale: float = 0.01
    
    def __post_init__(self):
        if isinstance(self.slew_axis, list):
            self.slew_axis = np.array(self.slew_axis)


@dataclass 
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    success: bool
    planning_time_sec: float
    final_angle_error_deg: float
    final_ang_vel_norm: float
    error_msg: Optional[str] = None


# =============================================================================
# Simple Orbit (no ephemeris lookups - constant B and S)
# =============================================================================

def create_simple_orbit(dt: float, tf: float) -> Orbit:
    """
    Create a simple circular orbit with constant B and S fields.
    Much faster than real orbit propagation.
    """
    ephem = Ephemeris()
    
    R = 7000.0 * np.array([1, 0, 0])  # km
    V = 7.5 * np.array([0, 1, 0])      # km/s (roughly circular)
    
    start_time = 0.22 - TimeConstants.sec2cent
    end_time = 0.22 + tf * TimeConstants.sec2cent
    
    # Constant environment (simplified)
    B_const = np.array([0, 3e-5, 2e-5])  # Tesla (typical LEO)
    S_const = np.array([1.5e8, 0, 0])     # km (sun direction)
    
    os0 = Orbital_State(
        ephem=ephem,
        J2000=start_time,
        R=R,
        V=V,
        B=B_const,
        S=S_const,
        rho=5e-12
    )
    
    return Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=False, fast=False)


# =============================================================================
# Core Experiment Runner  
# =============================================================================

def run_experiment(config: ExperimentConfig, verbose: bool = False) -> ExperimentResult:
    """Run a single planner experiment."""
    np.random.seed(config.seed)
    
    # Fresh satellite each run
    sat = create_beavercube2_cubesat(estimated=False)
    
    # Initial state
    q0 = normalize(np.array([1, 0.1, 0.1, 0.1]))
    w0 = normalize(np.random.randn(3)) * config.w0_scale
    h0 = np.array([0.0])
    x0 = np.concatenate([w0, q0, h0])
    
    # Goal quaternion
    axis = normalize(config.slew_axis)
    angle_rad = config.slew_angle_deg * np.pi / 180
    q_rot = np.array([
        np.cos(angle_rad/2),
        axis[0] * np.sin(angle_rad/2),
        axis[1] * np.sin(angle_rad/2),
        axis[2] * np.sin(angle_rad/2)
    ])
    q_goal = normalize(quat_mult(q0, q_rot))
    
    # Cost weights
    cost_weights = CostWeights(
        angle=config.angle,
        angle_N=config.angle_N,
        ang_vel=config.ang_vel,
        ang_vel_N=config.ang_vel_N,
        control_mult=config.control_mult,
        ang_cost_func_type=config.ang_cost_func_type,
    )
    
    # Solver config
    pass1 = SolverPassConfig(
        convergence=ConvergenceConfig(
            max_outer_iter=config.pass1_outer,
            max_inner_iter=config.pass1_inner
        ),
        aug_lag=AugLagConfig(penalty_init=1e-3)
    )
    pass2 = SolverPassConfig(
        convergence=ConvergenceConfig(
            max_outer_iter=config.pass2_outer,
            max_inner_iter=config.pass2_inner
        ),
        aug_lag=AugLagConfig(penalty_init=1e4)
    )
    
    planner_settings = PlannerSettings(
        est_sat=sat,
        bdot_on=0,
        dt_tp=10,
        dt_tvlqr=config.dt_planning,
        pass1_config=pass1,
        pass2_config=pass2,
        cost_main=cost_weights,
        cost_second=cost_weights,
        cost_tvlqr=cost_weights,
    )
    planner_settings.verbosity = verbose
    
    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=planner_settings)
    
    if config.use_quaternion_goal:
        goals = GoalList({0.22: Fixed_Attitude_Goal(q_goal)})
    else:
        goals = GoalList({0.22: Nadir_Goal()})
    
    # Simple orbit
    orb = create_simple_orbit(config.dt_planning, config.duration_sec + 60)
    os0 = orb.get_os(0.22)
    
    t_start = time.time()
    try:
        traj = controller.calculate_trajectory(
            t_start=0.22,
            duration=config.duration_sec,
            x_0=x0,
            os_0=os0,
            goals=goals,
            verbose=verbose
        )
        planning_time = time.time() - t_start
        
        x_final = traj.get_state_at(traj.end_time)
        q_final = x_final[3:7]
        w_final = x_final[0:3]
        
        q_err = quat_diff(q_goal, q_final)
        final_angle_deg = 2 * np.arccos(np.clip(abs(q_err[0]), 0, 1)) * 180 / np.pi
        final_w_norm = np.linalg.norm(w_final) * 180 / np.pi
        
        return ExperimentResult(
            config=config,
            success=True,
            planning_time_sec=planning_time,
            final_angle_error_deg=final_angle_deg,
            final_ang_vel_norm=final_w_norm,
        )
        
    except Exception as e:
        return ExperimentResult(
            config=config,
            success=False,
            planning_time_sec=time.time() - t_start,
            final_angle_error_deg=float('inf'),
            final_ang_vel_norm=float('inf'),
            error_msg=str(e),
        )


# =============================================================================
# Sweeps
# =============================================================================

def print_header(title: str, col: str):
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    print(f"{col:<12} {'time(s)':<10} {'err(°)':<12} {'ω(°/s)':<10} {'status'}")
    print("-" * 60)

def print_row(val, r: ExperimentResult, fmt: str = ""):
    st = "OK" if r.success else f"FAIL"
    if fmt:
        print(f"{val:<12{fmt}} {r.planning_time_sec:<10.2f} {r.final_angle_error_deg:<12.2f} {r.final_ang_vel_norm:<10.4f} {st}")
    else:
        print(f"{val:<12} {r.planning_time_sec:<10.2f} {r.final_angle_error_deg:<12.2f} {r.final_ang_vel_norm:<10.4f} {st}")


def sweep_angle_weight(base: ExperimentConfig) -> List[ExperimentResult]:
    print_header("Sweeping angle weight", "angle")
    results = []
    for val in [1e1, 1e2, 1e3, 1e4, 1e5]:
        cfg = ExperimentConfig(**{**asdict(base), 'angle': val, 'angle_N': val * 10})
        r = run_experiment(cfg)
        results.append(r)
        print_row(val, r, ".0e")
    return results


def sweep_ang_vel_weight(base: ExperimentConfig) -> List[ExperimentResult]:
    print_header("Sweeping ang_vel weight", "ang_vel")
    results = []
    for val in [1e1, 1e2, 1e3, 1e4, 1e5]:
        cfg = ExperimentConfig(**{**asdict(base), 'ang_vel': val, 'ang_vel_N': val * 10})
        r = run_experiment(cfg)
        results.append(r)
        print_row(val, r, ".0e")
    return results


def sweep_cost_func(base: ExperimentConfig) -> List[ExperimentResult]:
    print_header("Sweeping cost_func_type", "type")
    results = []
    for val in [0, 1, 2, 3, 4]:
        cfg = ExperimentConfig(**{**asdict(base), 'ang_cost_func_type': val})
        r = run_experiment(cfg)
        results.append(r)
        print_row(val, r)
    return results


def sweep_slew(base: ExperimentConfig) -> List[ExperimentResult]:
    print_header("Sweeping slew angle", "slew(°)")
    results = []
    for val in [15, 30, 45, 60, 90, 120]:
        cfg = ExperimentConfig(**{**asdict(base), 'slew_angle_deg': val})
        r = run_experiment(cfg)
        results.append(r)
        print_row(val, r)
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fast planner tuning")
    parser.add_argument("--sweep", choices=["angle", "ang_vel", "cost_func", "slew", "all"])
    parser.add_argument("--angle", type=float, default=1e3)
    parser.add_argument("--ang_vel", type=float, default=1e3)
    parser.add_argument("--cost_func", type=int, default=2)
    parser.add_argument("--slew_angle", type=float, default=45)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    base = ExperimentConfig(
        angle=args.angle,
        angle_N=args.angle * 10,
        ang_vel=args.ang_vel,
        ang_vel_N=args.ang_vel * 10,
        ang_cost_func_type=args.cost_func,
        slew_angle_deg=args.slew_angle,
        duration_sec=args.duration,
    )
    
    if args.sweep == "angle":
        sweep_angle_weight(base)
    elif args.sweep == "ang_vel":
        sweep_ang_vel_weight(base)
    elif args.sweep == "cost_func":
        sweep_cost_func(base)
    elif args.sweep == "slew":
        sweep_slew(base)
    elif args.sweep == "all":
        sweep_angle_weight(base)
        sweep_ang_vel_weight(base)
        sweep_cost_func(base)
        sweep_slew(base)
    else:
        print(f"Single run: {base.slew_angle_deg}° slew, costs: angle={base.angle:.0e}, ang_vel={base.ang_vel:.0e}")
        r = run_experiment(base, verbose=args.verbose)
        print(f"\nResult: {'OK' if r.success else 'FAIL'}")
        print(f"  Time: {r.planning_time_sec:.2f}s")
        print(f"  Final error: {r.final_angle_error_deg:.2f}°")
        print(f"  Final ω: {r.final_ang_vel_norm:.4f} °/s")


if __name__ == "__main__":
    main()
