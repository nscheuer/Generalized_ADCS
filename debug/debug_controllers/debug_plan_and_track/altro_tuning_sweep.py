"""
ALTRO Tuning Sweep for Plan & Track Controller.

This script systematically tests different ALTRO planner configurations to find
the best balance of speed and trajectory quality for the BC2 satellite.

Goals:
- ALTRO solve time < 60s (ideally < 20s) for 500s trajectory
- Near-zero angular error at end
- Near-zero angular velocity at end  
- Smooth trajectories (minimal oscillations)
- Good actuator utilization (both RW and MTQs)
- Respect actuator constraints

Usage:
    python altro_tuning_sweep.py                    # Run default sweep
    python altro_tuning_sweep.py --quick            # Quick sweep (fewer configs)
    python altro_tuning_sweep.py --config baseline  # Run specific config
"""
import sys
import os
import numpy as np
import time as time_module
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

from ADCS.CONOPS.goals import ECI_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.controller.helpers.planner_subsettings import (
    CostWeights, ConvergenceConfig, SolverPassConfig, AugLagConfig, RegularizationConfig
)
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize, rot_mat


# ============================================================================
# Result Data Structures
# ============================================================================

@dataclass
class TrajectoryMetrics:
    """Metrics for evaluating trajectory quality."""
    # Timing
    altro_time_s: float = 0.0
    env_prop_time_s: float = 0.0
    total_time_s: float = 0.0
    
    # Final state errors
    final_ang_vel_deg_s: float = 0.0
    final_pointing_error_deg: float = 0.0
    
    # Trajectory smoothness
    max_ang_vel_deg_s: float = 0.0
    mean_ang_vel_deg_s: float = 0.0
    mtq_sign_changes: int = 0
    rw_sign_changes: int = 0
    
    # Convergence
    converged: bool = False
    constraint_violation: float = 0.0
    
    # Actuator usage
    max_mtq_dipole: float = 0.0
    max_rw_torque: float = 0.0
    mean_mtq_dipole: float = 0.0
    mean_rw_torque: float = 0.0
    
    def score(self) -> float:
        """
        Compute overall quality score (higher is better).
        
        Weights chosen to balance speed vs quality:
        - Fast solve time is important
        - Final errors are critical
        - Smoothness is nice to have
        """
        if not self.converged or self.altro_time_s > 120:
            return -1000.0  # Penalize non-convergence heavily
        
        score = 0.0
        
        # Time score: 100 points for <20s, 50 for <60s, 0 for >60s
        if self.altro_time_s < 20:
            score += 100
        elif self.altro_time_s < 60:
            score += 50 * (60 - self.altro_time_s) / 40
        
        # Final error score: 100 points for <0.1 deg, scaled down
        if self.final_pointing_error_deg < 0.1:
            score += 100
        elif self.final_pointing_error_deg < 1.0:
            score += 80 * (1.0 - self.final_pointing_error_deg) / 0.9
        elif self.final_pointing_error_deg < 5.0:
            score += 40 * (5.0 - self.final_pointing_error_deg) / 4.0
        
        # Final velocity score: 50 points for <0.1 deg/s
        if self.final_ang_vel_deg_s < 0.1:
            score += 50
        elif self.final_ang_vel_deg_s < 1.0:
            score += 30 * (1.0 - self.final_ang_vel_deg_s) / 0.9
        
        # Smoothness penalty: lose up to 30 points for oscillations
        oscillation_penalty = min(30, self.mtq_sign_changes / 10)
        score -= oscillation_penalty
        
        return score


@dataclass 
class SweepConfig:
    """Configuration for a single sweep trial."""
    name: str
    
    # Cost weights
    angle: float = 1e3
    angle_N: float = 1e4
    ang_vel: float = 1e4
    ang_vel_N: float = 1e5
    control_mult: float = 1.0
    
    # Actuator weights
    mtq_control_weight: float = 1e3
    rw_control_weight: float = 1e5
    
    # Convergence
    pass1_max_outer: int = 20
    pass1_max_inner: int = 150
    pass2_max_outer: int = 20
    pass2_max_inner: int = 75
    grad_tol: float = 1e-4
    ilqr_cost_tol: float = 1e-2
    c_max: float = 0.0002
    
    # Augmented Lagrangian
    pass1_penalty_init: float = 1e-3
    pass2_penalty_init: float = 1e4
    penalty_scale: float = 10.0
    
    # Hessians
    use_full_cost_hessian: bool = True
    use_dynamics_hess: int = 1
    
    # Timing
    dt_tp: float = 30.0
    
    # Other
    bdot_on: int = 1
    use_raw_control_cost: bool = True
    wmax_deg: float = 20.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'angle': self.angle,
            'angle_N': self.angle_N,
            'ang_vel': self.ang_vel,
            'ang_vel_N': self.ang_vel_N,
            'control_mult': self.control_mult,
            'mtq_control_weight': self.mtq_control_weight,
            'rw_control_weight': self.rw_control_weight,
            'pass1_max_outer': self.pass1_max_outer,
            'pass1_max_inner': self.pass1_max_inner,
            'pass2_max_outer': self.pass2_max_outer,
            'pass2_max_inner': self.pass2_max_inner,
            'grad_tol': self.grad_tol,
            'ilqr_cost_tol': self.ilqr_cost_tol,
            'c_max': self.c_max,
            'pass1_penalty_init': self.pass1_penalty_init,
            'pass2_penalty_init': self.pass2_penalty_init,
            'penalty_scale': self.penalty_scale,
            'use_full_cost_hessian': self.use_full_cost_hessian,
            'use_dynamics_hess': self.use_dynamics_hess,
            'dt_tp': self.dt_tp,
            'bdot_on': self.bdot_on,
            'use_raw_control_cost': self.use_raw_control_cost,
            'wmax_deg': self.wmax_deg,
        }


@dataclass
class SweepResult:
    """Result from a single sweep trial."""
    config: SweepConfig
    metrics: TrajectoryMetrics
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'config': self.config.to_dict(),
            'metrics': {
                'altro_time_s': self.metrics.altro_time_s,
                'env_prop_time_s': self.metrics.env_prop_time_s,
                'total_time_s': self.metrics.total_time_s,
                'final_ang_vel_deg_s': self.metrics.final_ang_vel_deg_s,
                'final_pointing_error_deg': self.metrics.final_pointing_error_deg,
                'max_ang_vel_deg_s': self.metrics.max_ang_vel_deg_s,
                'mean_ang_vel_deg_s': self.metrics.mean_ang_vel_deg_s,
                'mtq_sign_changes': self.metrics.mtq_sign_changes,
                'rw_sign_changes': self.metrics.rw_sign_changes,
                'converged': self.metrics.converged,
                'max_mtq_dipole': self.metrics.max_mtq_dipole,
                'max_rw_torque': self.metrics.max_rw_torque,
                'score': self.metrics.score(),
            },
            'success': self.success,
            'error_message': self.error_message,
        }


# ============================================================================
# Predefined Configurations
# ============================================================================

def get_baseline_config() -> SweepConfig:
    """Current settings from debug_plan_and_track_bc2.py."""
    return SweepConfig(
        name="baseline",
        angle=1e10,
        angle_N=1e15,
        ang_vel=1e4,
        ang_vel_N=1e8,
        control_mult=1.0,
        mtq_control_weight=1e0,
        rw_control_weight=1e-6,
        pass1_max_outer=30,
        pass1_max_inner=150,
        pass2_max_outer=20,
        pass2_max_inner=60,
        grad_tol=0.005,
        ilqr_cost_tol=0.01,
        c_max=0.001,
        pass1_penalty_init=100.0,
        pass2_penalty_init=100.0,
        penalty_scale=10.0,
        use_full_cost_hessian=False,
        use_dynamics_hess=0,
        dt_tp=30.0,
        bdot_on=2,
        wmax_deg=10.0,
    )


def get_fast_config() -> SweepConfig:
    """Optimized for speed - fewer iterations, coarser dt."""
    return SweepConfig(
        name="fast",
        angle=1e8,
        angle_N=1e12,
        ang_vel=1e4,
        ang_vel_N=1e6,
        control_mult=1.0,
        pass1_max_outer=15,
        pass1_max_inner=100,
        pass2_max_outer=10,
        pass2_max_inner=50,
        grad_tol=0.01,
        ilqr_cost_tol=0.05,
        c_max=0.005,
        pass1_penalty_init=10.0,
        pass2_penalty_init=1000.0,
        use_full_cost_hessian=False,
        use_dynamics_hess=0,
        dt_tp=50.0,  # Coarser timestep
        bdot_on=2,
        wmax_deg=15.0,
    )


def get_quality_config() -> SweepConfig:
    """Optimized for trajectory quality - more iterations, full Hessians."""
    return SweepConfig(
        name="quality",
        angle=1e6,
        angle_N=1e10,
        ang_vel=1e6,
        ang_vel_N=1e10,
        control_mult=10.0,  # Higher control cost = smoother
        pass1_max_outer=40,
        pass1_max_inner=200,
        pass2_max_outer=30,
        pass2_max_inner=100,
        grad_tol=1e-4,
        ilqr_cost_tol=1e-3,
        c_max=0.0001,
        pass1_penalty_init=1.0,
        pass2_penalty_init=1e4,
        use_full_cost_hessian=True,
        use_dynamics_hess=1,
        dt_tp=20.0,  # Finer timestep
        bdot_on=1,
        wmax_deg=10.0,
    )


def get_balanced_config() -> SweepConfig:
    """Balanced speed and quality."""
    return SweepConfig(
        name="balanced",
        angle=1e8,
        angle_N=1e12,
        ang_vel=1e5,
        ang_vel_N=1e8,
        control_mult=1.0,
        pass1_max_outer=25,
        pass1_max_inner=150,
        pass2_max_outer=15,
        pass2_max_inner=75,
        grad_tol=0.001,
        ilqr_cost_tol=0.005,
        c_max=0.0005,
        pass1_penalty_init=10.0,
        pass2_penalty_init=1e4,
        use_full_cost_hessian=False,
        use_dynamics_hess=0,
        dt_tp=30.0,
        bdot_on=2,
        wmax_deg=12.0,
    )


def generate_sweep_configs() -> List[SweepConfig]:
    """Generate configurations for parameter sweep."""
    configs = []
    
    # Start with predefined configs
    configs.append(get_baseline_config())
    configs.append(get_fast_config())
    configs.append(get_quality_config())
    configs.append(get_balanced_config())
    
    # Sweep dt_tp (trajectory planner timestep)
    base = get_balanced_config()
    for dt in [10, 20, 30, 50]:
        c = SweepConfig(**{**base.to_dict(), 'name': f'dt_tp_{dt}', 'dt_tp': float(dt)})
        configs.append(c)
    
    # Sweep convergence iterations
    for outer, inner in [(10, 75), (20, 100), (30, 150), (40, 200)]:
        c = SweepConfig(
            **{**base.to_dict(), 
               'name': f'iter_{outer}x{inner}',
               'pass1_max_outer': outer,
               'pass1_max_inner': inner,
               'pass2_max_outer': outer // 2,
               'pass2_max_inner': inner // 2}
        )
        configs.append(c)
    
    # Sweep Hessian settings
    for full_hess, dyn_hess in [(False, 0), (False, 1), (True, 0), (True, 1)]:
        c = SweepConfig(
            **{**base.to_dict(),
               'name': f'hess_full{int(full_hess)}_dyn{dyn_hess}',
               'use_full_cost_hessian': full_hess,
               'use_dynamics_hess': dyn_hess}
        )
        configs.append(c)
    
    # Sweep angle vs ang_vel cost ratio
    for angle_exp, vel_exp in [(6, 4), (8, 4), (8, 6), (10, 6), (10, 8)]:
        c = SweepConfig(
            **{**base.to_dict(),
               'name': f'cost_ang{angle_exp}_vel{vel_exp}',
               'angle': 10**angle_exp,
               'angle_N': 10**(angle_exp + 4),
               'ang_vel': 10**vel_exp,
               'ang_vel_N': 10**(vel_exp + 4)}
        )
        configs.append(c)
    
    # Sweep bdot_on mode
    for bdot in [0, 1, 2]:
        c = SweepConfig(**{**base.to_dict(), 'name': f'bdot_{bdot}', 'bdot_on': bdot})
        configs.append(c)
    
    # Sweep penalty settings
    for p1_init, p2_init in [(1, 1e4), (10, 1e4), (100, 1e4), (10, 1e3), (10, 1e5)]:
        c = SweepConfig(
            **{**base.to_dict(),
               'name': f'penalty_{p1_init}_{p2_init:.0e}',
               'pass1_penalty_init': float(p1_init),
               'pass2_penalty_init': float(p2_init)}
        )
        configs.append(c)
    
    return configs


def generate_quick_sweep_configs() -> List[SweepConfig]:
    """Generate smaller set of configs for quick testing."""
    return [
        get_baseline_config(),
        get_fast_config(),
        get_balanced_config(),
        # Just a few key variations
        SweepConfig(**{**get_balanced_config().to_dict(), 'name': 'dt_tp_20', 'dt_tp': 20.0}),
        SweepConfig(**{**get_balanced_config().to_dict(), 'name': 'dt_tp_50', 'dt_tp': 50.0}),
        SweepConfig(**{**get_balanced_config().to_dict(), 'name': 'no_hess', 
                      'use_full_cost_hessian': False, 'use_dynamics_hess': 0}),
    ]


# ============================================================================
# Core Evaluation Functions
# ============================================================================

def create_planner_settings(config: SweepConfig, sat) -> PlannerSettings:
    """Create PlannerSettings from a SweepConfig."""
    
    # Cost weights for main pass
    cost_main = CostWeights(
        angle=config.angle,
        angle_N=config.angle_N,
        ang_vel=config.ang_vel,
        ang_vel_N=config.ang_vel_N,
        control_mult=config.control_mult,
        use_raw_control_cost=config.use_raw_control_cost,
        use_full_cost_hessian=config.use_full_cost_hessian,
    )
    
    # Cost weights for second pass (same structure, can adjust if needed)
    cost_second = CostWeights(
        angle=config.angle,
        angle_N=config.angle_N,
        ang_vel=config.ang_vel,
        ang_vel_N=config.ang_vel_N,
        control_mult=config.control_mult * 10,  # Often want tighter second pass
        use_raw_control_cost=config.use_raw_control_cost,
        use_full_cost_hessian=config.use_full_cost_hessian,
    )
    
    # Convergence settings
    conv1 = ConvergenceConfig(
        max_outer_iter=config.pass1_max_outer,
        max_inner_iter=config.pass1_max_inner,
        grad_tol=config.grad_tol,
        ilqr_cost_tol=config.ilqr_cost_tol,
        c_max=config.c_max,
    )
    conv2 = ConvergenceConfig(
        max_outer_iter=config.pass2_max_outer,
        max_inner_iter=config.pass2_max_inner,
        grad_tol=config.grad_tol / 10,  # Tighter for pass 2
        ilqr_cost_tol=config.ilqr_cost_tol / 10,
        c_max=config.c_max / 10,
    )
    
    # Augmented Lagrangian settings  
    aug1 = AugLagConfig(
        penalty_init=config.pass1_penalty_init,
        penalty_scale=config.penalty_scale,
    )
    aug2 = AugLagConfig(
        penalty_init=config.pass2_penalty_init,
        penalty_scale=config.penalty_scale * 2,
    )
    
    # Regularization
    reg1 = RegularizationConfig(use_dynamics_hess=config.use_dynamics_hess)
    reg2 = RegularizationConfig(use_dynamics_hess=config.use_dynamics_hess)
    
    # Solver passes
    pass1 = SolverPassConfig(convergence=conv1, aug_lag=aug1, regularization=reg1)
    pass2 = SolverPassConfig(convergence=conv2, aug_lag=aug2, regularization=reg2)
    
    # Create PlannerSettings
    ps = PlannerSettings(
        est_sat=sat,
        bdot_on=config.bdot_on,
        dt_tp=config.dt_tp,
        dt_tvlqr=1.0,
        pass1_config=pass1,
        pass2_config=pass2,
        cost_main=cost_main,
        cost_second=cost_second,
    )
    
    # Set actuator weights
    ps.mtq_control_weight = config.mtq_control_weight
    ps.rw_control_weight = config.rw_control_weight
    ps.wmax = config.wmax_deg * np.pi / 180.0
    
    return ps


def evaluate_trajectory(
    times: np.ndarray, 
    states: np.ndarray, 
    controls: np.ndarray,
    goal_vec: np.ndarray
) -> TrajectoryMetrics:
    """Compute trajectory quality metrics."""
    metrics = TrajectoryMetrics()
    
    N = len(times)
    
    # Angular velocity metrics
    w = states[:, :3]
    w_mag_deg = np.rad2deg(np.linalg.norm(w, axis=1))
    metrics.final_ang_vel_deg_s = w_mag_deg[-1]
    metrics.max_ang_vel_deg_s = np.max(w_mag_deg)
    metrics.mean_ang_vel_deg_s = np.mean(w_mag_deg)
    
    # Pointing error
    errors_deg = []
    for i in range(N):
        q = states[i, 3:7]
        R = rot_mat(q)
        body_boresight = np.array([0, 0, 1])
        eci_boresight = R @ body_boresight
        error_rad = np.arccos(np.clip(np.dot(eci_boresight, goal_vec), -1, 1))
        errors_deg.append(np.rad2deg(error_rad))
    metrics.final_pointing_error_deg = errors_deg[-1]
    
    # Control metrics
    n_mtq = min(3, controls.shape[1])
    mtq = controls[:, :n_mtq]
    metrics.max_mtq_dipole = np.max(np.abs(mtq))
    metrics.mean_mtq_dipole = np.mean(np.abs(mtq))
    
    # Sign changes (oscillation indicator)
    metrics.mtq_sign_changes = int(np.sum(np.diff(np.sign(mtq), axis=0) != 0))
    
    if controls.shape[1] > 3:
        rw = controls[:, 3:]
        metrics.max_rw_torque = np.max(np.abs(rw))
        metrics.mean_rw_torque = np.mean(np.abs(rw))
        metrics.rw_sign_changes = int(np.sum(np.diff(np.sign(rw), axis=0) != 0))
    
    # Assume converged if final error is reasonable
    metrics.converged = (metrics.final_pointing_error_deg < 10.0 and 
                        metrics.final_ang_vel_deg_s < 5.0)
    
    return metrics


def run_single_trial(
    config: SweepConfig,
    sat,
    orbit: Orbit,
    x0: np.ndarray,
    goal_vec: np.ndarray,
    duration: float = 500.0,
    verbose: bool = False
) -> SweepResult:
    """Run a single trial with the given configuration."""
    
    metrics = TrajectoryMetrics()
    
    try:
        # Create planner
        ps = create_planner_settings(config, sat)
        controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=ps)
        
        # Setup goal
        goal = ECI_Goal(goal_vec)
        start_time = 0.22
        goals = GoalList({start_time: goal})
        os0 = orbit.get_os(start_time)
        
        # Time the trajectory calculation
        t_start = time_module.perf_counter()
        
        traj: Trajectory = controller.calculate_trajectory(
            t_start=start_time,
            duration=duration,
            x_0=x0,
            os_0=os0,
            goals=goals,
            verbose=verbose,
        )
        
        t_end = time_module.perf_counter()
        metrics.total_time_s = t_end - t_start
        
        # For now, total time includes env prop - we'll refine this
        # In debug_plan_and_track_bc2.py there's timing instrumentation
        metrics.altro_time_s = metrics.total_time_s  # Approximate
        
        # Extract trajectory data
        times = (traj.times - start_time) * TimeConstants.cent2sec
        states = traj.states.T if traj.states.shape[0] != len(traj.times) else traj.states
        controls = traj.controls.T if traj.controls.shape[0] != len(traj.times) - 1 else traj.controls
        
        # Pad controls to match state length for evaluation
        if len(controls) < len(states):
            controls = np.vstack([controls, controls[-1:]])
        
        # Evaluate trajectory quality
        qual_metrics = evaluate_trajectory(times, states, controls, goal_vec)
        
        # Copy quality metrics
        metrics.final_ang_vel_deg_s = qual_metrics.final_ang_vel_deg_s
        metrics.final_pointing_error_deg = qual_metrics.final_pointing_error_deg
        metrics.max_ang_vel_deg_s = qual_metrics.max_ang_vel_deg_s
        metrics.mean_ang_vel_deg_s = qual_metrics.mean_ang_vel_deg_s
        metrics.mtq_sign_changes = qual_metrics.mtq_sign_changes
        metrics.rw_sign_changes = qual_metrics.rw_sign_changes
        metrics.converged = qual_metrics.converged
        metrics.max_mtq_dipole = qual_metrics.max_mtq_dipole
        metrics.max_rw_torque = qual_metrics.max_rw_torque
        metrics.mean_mtq_dipole = qual_metrics.mean_mtq_dipole
        metrics.mean_rw_torque = qual_metrics.mean_rw_torque
        
        return SweepResult(config=config, metrics=metrics, success=True)
        
    except Exception as e:
        return SweepResult(
            config=config, 
            metrics=metrics, 
            success=False, 
            error_message=str(e)
        )


# ============================================================================
# Main Sweep Runner
# ============================================================================

def run_sweep(
    configs: List[SweepConfig],
    duration: float = 500.0,
    seed: int = 42,
    verbose: bool = False
) -> List[SweepResult]:
    """Run sweep over all configurations."""
    
    np.random.seed(seed)
    
    # Create satellite
    print("Creating BC2 satellite...")
    sat = create_beavercube2_cubesat(estimated=False)
    sat.rw_actuators[0].h = 0.0
    
    # Create initial state
    w0 = random_n_unit_vec(3) * np.random.uniform(0.5, 1.0) * np.pi / 180.0
    q0 = normalize(np.random.randn(4))
    h0 = np.array([0.0])
    x0 = np.concatenate([w0, q0, h0])
    
    print(f"Initial angular velocity: {np.rad2deg(np.linalg.norm(w0)):.2f} deg/s")
    
    # Create orbit (fast=True for speed - we're testing ALTRO, not orbit prop)
    print("Creating orbit (fast mode)...")
    ephem = Ephemeris()
    start_time = 0.22 - 1 * TimeConstants.sec2cent
    end_time = 0.22 + (duration + 100) * TimeConstants.sec2cent
    R = 7000 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    V = np.array([8, 0, 0])
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    orbit = Orbit(os0=os0, end_time=end_time, dt=1, use_J2=True, fast=True)
    
    # Goal
    goal_vec = normalize(np.array([0, 0, 1]))
    
    # Run trials
    results = []
    print(f"\nRunning {len(configs)} configurations...")
    print("=" * 80)
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing: {config.name}")
        
        result = run_single_trial(
            config=config,
            sat=sat,
            orbit=orbit,
            x0=x0.copy(),
            goal_vec=goal_vec,
            duration=duration,
            verbose=verbose
        )
        
        results.append(result)
        
        if result.success:
            m = result.metrics
            print(f"  Time: {m.altro_time_s:.1f}s | "
                  f"Final err: {m.final_pointing_error_deg:.2f}° | "
                  f"Final vel: {m.final_ang_vel_deg_s:.2f}°/s | "
                  f"Score: {m.score():.1f}")
        else:
            print(f"  FAILED: {result.error_message[:60]}")
    
    return results


def print_results_summary(results: List[SweepResult]):
    """Print summary of sweep results."""
    print("\n" + "=" * 80)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 80)
    
    # Sort by score
    sorted_results = sorted(results, key=lambda r: r.metrics.score(), reverse=True)
    
    print(f"\n{'Config':<30} {'Time(s)':<10} {'Error(°)':<10} {'Vel(°/s)':<10} {'Score':<10}")
    print("-" * 70)
    
    for r in sorted_results:
        if r.success:
            m = r.metrics
            print(f"{r.config.name:<30} {m.altro_time_s:<10.1f} "
                  f"{m.final_pointing_error_deg:<10.2f} {m.final_ang_vel_deg_s:<10.2f} "
                  f"{m.score():<10.1f}")
        else:
            print(f"{r.config.name:<30} FAILED")
    
    # Best configs
    print("\n" + "=" * 80)
    print("TOP 5 CONFIGURATIONS")
    print("=" * 80)
    
    for i, r in enumerate(sorted_results[:5]):
        if r.success:
            print(f"\n{i+1}. {r.config.name} (Score: {r.metrics.score():.1f})")
            print(f"   ALTRO time: {r.metrics.altro_time_s:.1f}s")
            print(f"   Final pointing error: {r.metrics.final_pointing_error_deg:.3f}°")
            print(f"   Final angular velocity: {r.metrics.final_ang_vel_deg_s:.3f}°/s")
            print(f"   MTQ sign changes: {r.metrics.mtq_sign_changes}")
            print(f"   Config: dt_tp={r.config.dt_tp}, bdot={r.config.bdot_on}, "
                  f"hess={r.config.use_full_cost_hessian}/{r.config.use_dynamics_hess}")


def save_results(results: List[SweepResult], filename: str = None):
    """Save results to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"altro_sweep_results_{timestamp}.json"
    
    output_dir = os.path.dirname(__file__)
    filepath = os.path.join(output_dir, filename)
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'results': [r.to_dict() for r in results]
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ALTRO Tuning Sweep")
    parser.add_argument("--quick", action="store_true", help="Run quick sweep with fewer configs")
    parser.add_argument("--config", type=str, help="Run specific config by name")
    parser.add_argument("--duration", type=float, default=500.0, help="Trajectory duration in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose planner output")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()
    
    # Select configs
    if args.config:
        # Run specific config
        all_configs = generate_sweep_configs()
        configs = [c for c in all_configs if c.name == args.config]
        if not configs:
            print(f"Config '{args.config}' not found. Available:")
            for c in all_configs:
                print(f"  - {c.name}")
            sys.exit(1)
    elif args.quick:
        configs = generate_quick_sweep_configs()
    else:
        configs = generate_sweep_configs()
    
    print(f"ALTRO Tuning Sweep")
    print(f"  Configs: {len(configs)}")
    print(f"  Duration: {args.duration}s")
    print(f"  Seed: {args.seed}")
    
    # Run sweep
    results = run_sweep(
        configs=configs,
        duration=args.duration,
        seed=args.seed,
        verbose=args.verbose
    )
    
    # Print summary
    print_results_summary(results)
    
    # Save if requested
    if args.save:
        save_results(results)
