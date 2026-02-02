#!/usr/bin/env python3
"""
Experiment with maximizing RW usage in trajectory planner.

Goal: Find settings that give highest RW usage while meeting constraints,
with sharp transitions between goals instead of U-shaped slow rotations.

Experiments:
1. Cost weight ratios (angle vs ang_vel)
2. Warm start with bang-bang trajectory  
3. Full Hessians vs Gauss-Newton
4. Control cost ratios (RW vs MTQ)
5. Terminal vs running cost balance

Run:
    python papers/Planner/experiment_rw_usage.py
"""

import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/pmckeen/Generalized_ADCS')
sys.path.insert(0, '/home/pmckeen/Generalized_ADCS/papers/Planner')

from ADCS.CONOPS.goals import Fixed_Attitude_Goal, ECI_Goal, No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.controller.plan_and_track_python_alilqr import Plan_and_Track_PythonALILQR
from ADCS.controller.helpers import (
    PlannerSettings, create_planner_settings,
    IterationData, LivePlannerViz,
)
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize, quat_mult, quat_diff
from ADCS.controller.helpers.build_csat import build_cpp_satellite
from mc_planner_settings import create_adaptive_planner_settings, create_optimized_planner_settings
import trajectory_planner.build.tplaunch as tplaunch
from scipy.spatial.transform import Rotation


def boresight_error(q, goal_vec, boresight=np.array([0, 1, 0])):
    """Compute boresight error in degrees."""
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    boresight_eci = R @ boresight
    return np.degrees(np.arccos(np.clip(np.dot(boresight_eci, goal_vec), -1, 1)))


@dataclass
class ExperimentResult:
    """Result of one experiment configuration."""
    name: str
    final_error: float
    min_error: float
    time_to_10deg: Optional[float]
    rw_max_pct: float
    rw_mean_pct: float
    mtq_max_pct: float
    cmax: float
    converged: bool
    iterations: int
    elapsed: float
    errors: List[float] = field(default_factory=list)
    rw_usage: List[float] = field(default_factory=list)
    

def create_scenario():
    """Create the test scenario - single 90° slew for clear results."""
    np.random.seed(42)
    sat = create_beavercube2_cubesat(estimated=False)
    
    tf = 60.0  # 60s horizon - enough time for 90° slew
    t_start = 0.22
    sec2cent = TimeConstants.sec2cent
    
    orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=tf+100, use_J2=True, fast=True)
    orb.populate_environment(compute_B=True, compute_S=True)
    os0 = orb.get_os(t_start)
    
    # 90° initial error - body-Y boresight, goal is ECI-X
    q0 = normalize(np.array([1.0, 0.0, 0.0, 0.0]))  # Identity
    w0 = np.array([0.0, 0.0, 0.0])
    h0 = np.array([0.0])
    x0 = np.concatenate([w0, q0, h0])
    
    goal_vec = np.array([1.0, 0.0, 0.0])  # Requires 90° yaw
    goals = GoalList({t_start: ECI_Goal(goal_vec)})
    
    for i, rw in enumerate(sat.rw_actuators):
        rw.h = h0[i]
    
    return sat, orb, os0, t_start, tf, x0, goals, goal_vec


def run_experiment(sat, os0, t_start, tf, x0, goals, goal_vec, settings, name, 
                   warm_start_controls=None, verbose=False):
    """Run one experiment configuration."""
    dt = settings.dt_tp
    N = int(tf / dt) + 1
    
    rw = sat.rw_actuators[0]
    rw_limit = rw.u_max * settings.control_limit_scale
    mtq = sat.mtq_actuators[0]
    mtq_limit = mtq.u_max * settings.control_limit_scale
    
    # Create controller
    controller = Plan_and_Track_PythonALILQR(est_sat=sat, planner_settings=settings, verbose=False)
    
    all_iters = []
    def callback(iter_data):
        all_iters.append(iter_data)
        if verbose and len(all_iters) % 10 == 1:
            q_f = iter_data.Xset[3:7, -1]
            err = boresight_error(q_f, goal_vec)
            rw_max = np.abs(iter_data.Uset[3, :]).max() / rw_limit * 100 if iter_data.Uset.shape[0] > 3 else 0
            print(f"  [{iter_data.pass_label}] iter {len(all_iters):3d}: err={err:5.1f}°, RW={rw_max:5.1f}%, cmax={iter_data.cmax:.1e}")
    
    controller.set_iteration_callback(callback)
    
    # Run optimization
    t0 = time.perf_counter()
    try:
        traj = controller.calculate_trajectory(
            t_start=t_start, duration=tf, x_0=x0, os_0=os0, 
            goals=goals, verbose=False
        )
        elapsed = time.perf_counter() - t0
        
        if traj is None:
            return ExperimentResult(name=name, final_error=180, min_error=180, time_to_10deg=None,
                                   rw_max_pct=0, rw_mean_pct=0, mtq_max_pct=0, cmax=1e10,
                                   converged=False, iterations=len(all_iters), elapsed=elapsed)
        
        # Compute metrics
        states = traj.states
        controls = traj.controls
        
        errors = []
        for k in range(states.shape[1]):
            q_k = states[3:7, k]
            errors.append(boresight_error(q_k, goal_vec))
        
        rw_usage = np.abs(controls[3, :]) / rw_limit * 100 if controls.shape[0] > 3 else np.zeros(controls.shape[1])
        mtq_usage = np.abs(controls[:3, :]).max(axis=0) / mtq_limit * 100
        
        time_to_10 = next((k * dt for k, e in enumerate(errors) if e < 10), None)
        
        # Get final cmax from last iteration
        cmax = all_iters[-1].cmax if all_iters else 1e10
        
        return ExperimentResult(
            name=name,
            final_error=errors[-1],
            min_error=min(errors),
            time_to_10deg=time_to_10,
            rw_max_pct=rw_usage.max(),
            rw_mean_pct=rw_usage.mean(),
            mtq_max_pct=mtq_usage.max(),
            cmax=cmax,
            converged=cmax < 0.01,
            iterations=len(all_iters),
            elapsed=elapsed,
            errors=errors,
            rw_usage=list(rw_usage)
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  ERROR: {e}")
        return ExperimentResult(name=name, final_error=180, min_error=180, time_to_10deg=None,
                               rw_max_pct=0, rw_mean_pct=0, mtq_max_pct=0, cmax=1e10,
                               converged=False, iterations=len(all_iters), elapsed=elapsed)


def experiment_1_cost_ratios(sat, os0, t_start, tf, x0, goals, goal_vec):
    """Test different ang_vel/angle ratios."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Angular Velocity / Angle Cost Ratio")
    print("="*70)
    
    results = []
    # Test ratios from very low (allow fast rotation) to high (damped)
    for ratio_mult in [0.01, 0.1, 1, 10, 50, 100]:
        settings = create_adaptive_planner_settings(sat, duration=tf, verbose=False)
        
        # Base ratio is about 100 from normalization
        # Apply multiplier to ang_vel
        settings.cost_main.ang_vel *= ratio_mult
        settings.cost_second.ang_vel *= ratio_mult
        
        # Cheap controls
        settings.mtq_control_weight *= 0.1
        settings.rw_control_weight *= 0.01
        settings.wmax = np.radians(60)
        settings.bdot_on = 1
        
        actual_ratio = settings.cost_main.ang_vel / settings.cost_main.angle
        name = f"ratio_{ratio_mult}x (actual={actual_ratio:.0f})"
        
        print(f"\nTesting {name}...")
        result = run_experiment(sat, os0, t_start, tf, x0, goals, goal_vec, settings, name, verbose=True)
        results.append(result)
        print(f"  Result: err={result.final_error:.1f}°, RW={result.rw_max_pct:.1f}%, cmax={result.cmax:.1e}")
    
    return results


def experiment_2_hessians(sat, os0, t_start, tf, x0, goals, goal_vec):
    """Test Gauss-Newton vs Full Newton."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Gauss-Newton vs Full Hessians")
    print("="*70)
    
    results = []
    for use_hessian in [False, True]:
        settings = create_adaptive_planner_settings(sat, duration=tf, verbose=False)
        
        settings.cost_main.use_full_cost_hessian = use_hessian
        settings.cost_second.use_full_cost_hessian = use_hessian
        
        # Moderate settings
        settings.cost_main.ang_vel *= 10  # Moderate damping
        settings.cost_second.ang_vel *= 10
        settings.mtq_control_weight *= 0.1
        settings.rw_control_weight *= 0.01
        settings.wmax = np.radians(60)
        settings.bdot_on = 1
        
        name = f"Hessian={'ON' if use_hessian else 'OFF'}"
        
        print(f"\nTesting {name}...")
        result = run_experiment(sat, os0, t_start, tf, x0, goals, goal_vec, settings, name, verbose=True)
        results.append(result)
        print(f"  Result: err={result.final_error:.1f}°, RW={result.rw_max_pct:.1f}%, cmax={result.cmax:.1e}")
    
    return results


def experiment_3_control_costs(sat, os0, t_start, tf, x0, goals, goal_vec):
    """Test different RW/MTQ cost ratios."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: RW vs MTQ Control Cost Ratio")
    print("="*70)
    
    results = []
    for rw_mult in [0.001, 0.01, 0.1, 1.0, 10.0]:
        settings = create_adaptive_planner_settings(sat, duration=tf, verbose=False)
        
        settings.mtq_control_weight *= 0.1
        settings.rw_control_weight = settings.mtq_control_weight * rw_mult
        
        settings.cost_main.ang_vel *= 10
        settings.cost_second.ang_vel *= 10
        settings.wmax = np.radians(60)
        settings.bdot_on = 1
        
        name = f"RW_cost={rw_mult}x_MTQ"
        
        print(f"\nTesting {name}...")
        result = run_experiment(sat, os0, t_start, tf, x0, goals, goal_vec, settings, name, verbose=True)
        results.append(result)
        print(f"  Result: err={result.final_error:.1f}°, RW={result.rw_max_pct:.1f}%, cmax={result.cmax:.1e}")
    
    return results


def experiment_4_terminal_costs(sat, os0, t_start, tf, x0, goals, goal_vec):
    """Test different terminal/running cost balances."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Terminal vs Running Cost Balance")
    print("="*70)
    
    results = []
    for term_mult in [1, 10, 100, 1000]:
        settings = create_adaptive_planner_settings(sat, duration=tf, verbose=False)
        
        # High terminal, variable running
        settings.cost_main.angle_N *= term_mult
        settings.cost_main.ang_vel_N *= term_mult
        settings.cost_second.angle_N *= term_mult
        settings.cost_second.ang_vel_N *= term_mult
        
        # Lower running cost for the higher terminal cases
        if term_mult >= 100:
            settings.cost_main.angle *= 0.1
            settings.cost_main.ang_vel *= 0.1
            settings.cost_second.angle *= 0.1
            settings.cost_second.ang_vel *= 0.1
        
        settings.mtq_control_weight *= 0.1
        settings.rw_control_weight *= 0.01
        settings.wmax = np.radians(60)
        settings.bdot_on = 1
        
        name = f"terminal={term_mult}x"
        
        print(f"\nTesting {name}...")
        result = run_experiment(sat, os0, t_start, tf, x0, goals, goal_vec, settings, name, verbose=True)
        results.append(result)
        print(f"  Result: err={result.final_error:.1f}°, RW={result.rw_max_pct:.1f}%, cmax={result.cmax:.1e}")
    
    return results


def experiment_5_init_modes(sat, os0, t_start, tf, x0, goals, goal_vec):
    """Test different initialization modes."""
    print("\n" + "="*70)
    print("EXPERIMENT 5: Initialization Modes")
    print("="*70)
    
    results = []
    for bdot_on in [0, 1, 4, 5]:
        settings = create_adaptive_planner_settings(sat, duration=tf, verbose=False)
        
        settings.cost_main.ang_vel *= 10
        settings.cost_second.ang_vel *= 10
        settings.mtq_control_weight *= 0.1
        settings.rw_control_weight *= 0.01
        settings.wmax = np.radians(60)
        settings.bdot_on = bdot_on
        
        mode_names = {0: "zero", 1: "bdot", 4: "PD", 5: "PD+noise"}
        name = f"init={mode_names.get(bdot_on, bdot_on)}"
        
        print(f"\nTesting {name}...")
        result = run_experiment(sat, os0, t_start, tf, x0, goals, goal_vec, settings, name, verbose=True)
        results.append(result)
        print(f"  Result: err={result.final_error:.1f}°, RW={result.rw_max_pct:.1f}%, cmax={result.cmax:.1e}")
    
    return results


def experiment_6_combined_best(sat, os0, t_start, tf, x0, goals, goal_vec):
    """Try combinations of best settings from other experiments."""
    print("\n" + "="*70)
    print("EXPERIMENT 6: Combined Best Settings")
    print("="*70)
    
    results = []
    
    configs = [
        # Name, ang_vel_mult, rw_mult, term_mult, hessian, bdot
        ("baseline", 50, 0.1, 25, False, 1),
        ("cheap_rw", 50, 0.001, 25, False, 1),
        ("low_angvel", 1, 0.01, 25, False, 1),
        ("high_terminal", 10, 0.01, 1000, False, 1),
        ("low_running", 0.1, 0.01, 100, False, 0),
        ("hessian_on", 10, 0.01, 100, True, 1),
        ("aggressive_combo", 0.1, 0.001, 1000, False, 0),
    ]
    
    for name, ang_vel_mult, rw_mult, term_mult, use_hessian, bdot in configs:
        settings = create_adaptive_planner_settings(sat, duration=tf, verbose=False)
        
        settings.cost_main.ang_vel *= ang_vel_mult
        settings.cost_main.angle_N *= term_mult
        settings.cost_main.ang_vel_N *= term_mult
        settings.cost_main.use_full_cost_hessian = use_hessian
        
        settings.cost_second.ang_vel *= ang_vel_mult
        settings.cost_second.angle_N *= term_mult
        settings.cost_second.ang_vel_N *= term_mult
        settings.cost_second.use_full_cost_hessian = use_hessian
        
        settings.mtq_control_weight *= 0.1
        settings.rw_control_weight = settings.mtq_control_weight * rw_mult
        settings.wmax = np.radians(60)
        settings.bdot_on = bdot
        
        print(f"\nTesting {name}...")
        result = run_experiment(sat, os0, t_start, tf, x0, goals, goal_vec, settings, name, verbose=True)
        results.append(result)
        print(f"  Result: err={result.final_error:.1f}°, RW={result.rw_max_pct:.1f}%, cmax={result.cmax:.1e}")
    
    return results


def plot_results(all_results: Dict[str, List[ExperimentResult]]):
    """Create summary plot of all experiments."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('RW Usage Experiments', fontsize=14, fontweight='bold')
    
    for ax, (exp_name, results) in zip(axes.flat, all_results.items()):
        names = [r.name.split('=')[-1] if '=' in r.name else r.name for r in results]
        rw_pct = [r.rw_max_pct for r in results]
        errors = [r.final_error for r in results]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax2 = ax.twinx()
        bars1 = ax.bar(x - width/2, rw_pct, width, label='RW max %', color='blue', alpha=0.7)
        bars2 = ax2.bar(x + width/2, errors, width, label='Final error °', color='red', alpha=0.7)
        
        ax.set_ylabel('RW Usage %', color='blue')
        ax2.set_ylabel('Error °', color='red')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_title(exp_name)
        ax.set_ylim(0, 150)
        ax2.set_ylim(0, 100)
        
        # Mark converged results
        for i, r in enumerate(results):
            if r.converged:
                ax.annotate('✓', (i, rw_pct[i]+5), ha='center', fontsize=10, color='green')
    
    plt.tight_layout()
    save_path = '/tmp/rw_experiments.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.close()  # Don't block


def print_summary(all_results: Dict[str, List[ExperimentResult]]):
    """Print summary table."""
    print("\n" + "="*90)
    print("SUMMARY - All Experiments")
    print("="*90)
    print(f"{'Experiment':<20} {'Config':<25} {'Error°':>8} {'RW%':>8} {'t<10°':>8} {'cmax':>10} {'Conv':>5}")
    print("-"*90)
    
    best_rw = None
    best_rw_val = 0
    
    for exp_name, results in all_results.items():
        for r in results:
            t10 = f"{r.time_to_10deg:.1f}" if r.time_to_10deg else "never"
            conv = "✓" if r.converged else "✗"
            print(f"{exp_name:<20} {r.name:<25} {r.final_error:>8.1f} {r.rw_max_pct:>8.1f} {t10:>8} {r.cmax:>10.1e} {conv:>5}")
            
            if r.converged and r.rw_max_pct > best_rw_val:
                best_rw_val = r.rw_max_pct
                best_rw = (exp_name, r)
    
    print("-"*90)
    if best_rw:
        exp, r = best_rw
        print(f"\nBEST CONVERGED: {exp} / {r.name}")
        print(f"  RW max: {r.rw_max_pct:.1f}%")
        print(f"  Final error: {r.final_error:.1f}°")
        print(f"  Time to <10°: {r.time_to_10deg}")


def main():
    print("="*70)
    print("RW USAGE EXPERIMENTS")
    print("="*70)
    
    # Create scenario
    sat, orb, os0, t_start, tf, x0, goals, goal_vec = create_scenario()
    initial_error = boresight_error(x0[3:7], goal_vec)
    print(f"\nScenario: {initial_error:.1f}° initial error, {tf}s horizon")
    print(f"Goal: Maximize RW usage while converging to <10° error")
    
    all_results = {}
    
    # Run experiments
    all_results['1_ratio'] = experiment_1_cost_ratios(sat, os0, t_start, tf, x0, goals, goal_vec)
    all_results['2_hessian'] = experiment_2_hessians(sat, os0, t_start, tf, x0, goals, goal_vec)
    all_results['3_control'] = experiment_3_control_costs(sat, os0, t_start, tf, x0, goals, goal_vec)
    all_results['4_terminal'] = experiment_4_terminal_costs(sat, os0, t_start, tf, x0, goals, goal_vec)
    all_results['5_init'] = experiment_5_init_modes(sat, os0, t_start, tf, x0, goals, goal_vec)
    all_results['6_combined'] = experiment_6_combined_best(sat, os0, t_start, tf, x0, goals, goal_vec)
    
    # Summary
    print_summary(all_results)
    plot_results(all_results)
    
    return all_results


if __name__ == '__main__':
    results = main()
