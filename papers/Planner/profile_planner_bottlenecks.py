#!/usr/bin/env python3
"""
Profile the trajectory planner to identify performance bottlenecks.

This script uses the Python ALILQR wrapper to time individual operations:
- Backward pass (Riccati recursion)
- Forward pass (line search + rollout)
- Cost evaluation
- Constraint violation check
- Augmented Lagrangian updates

Tests multiple configurations:
1. MTQ-only (3 actuators)
2. MTQ + 1 RW (4 actuators) 
3. Different goal types (vector vs quaternion)
4. Different horizon lengths
"""

import numpy as np
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from collections import defaultdict

sys.path.insert(0, '/home/pmckeen/Generalized_ADCS')

from ADCS.CONOPS.goals import Fixed_Attitude_Goal, Nadir_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import (
    PlannerSettings, create_planner_settings,
    NormalizedPlannerConfig, NormalizedActuatorCosts, NormalizedStateCosts,
)
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize, quat_mult, quat_diff

# Try to import the C++ planner directly for profiling
try:
    import tplaunch
    HAS_TPLAUNCH = True
except ImportError:
    HAS_TPLAUNCH = False
    print("Warning: tplaunch not available, using high-level profiling only")


@dataclass
class TimingStats:
    """Accumulator for timing statistics."""
    times: List[float] = field(default_factory=list)
    
    def add(self, t: float):
        self.times.append(t)
    
    @property
    def total(self) -> float:
        return sum(self.times)
    
    @property
    def mean(self) -> float:
        return np.mean(self.times) if self.times else 0.0
    
    @property
    def std(self) -> float:
        return np.std(self.times) if len(self.times) > 1 else 0.0
    
    @property
    def count(self) -> int:
        return len(self.times)


class ProfilingALILQR:
    """
    Instrumented ALILQR that times each operation.
    
    Wraps the C++ planner methods and records timing for:
    - backwardPass
    - forwardPass  
    - cost2Func
    - maxViol
    - incrementAugLag
    - generateInitialTrajectory
    """
    
    def __init__(self, planner):
        self.planner = planner
        self.timings: Dict[str, TimingStats] = defaultdict(TimingStats)
        
    def reset_timings(self):
        self.timings = defaultdict(TimingStats)
    
    def _timed_call(self, name: str, func, *args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        self.timings[name].add(elapsed)
        return result
    
    def generateInitialTrajectory(self, *args, **kwargs):
        return self._timed_call('generateInitialTrajectory', 
                                self.planner.generateInitialTrajectory, *args, **kwargs)
    
    def backwardPass(self, *args, **kwargs):
        return self._timed_call('backwardPass',
                                self.planner.backwardPass, *args, **kwargs)
    
    def forwardPass(self, *args, **kwargs):
        return self._timed_call('forwardPass',
                                self.planner.forwardPass, *args, **kwargs)
    
    def cost2Func(self, *args, **kwargs):
        return self._timed_call('cost2Func',
                                self.planner.cost2Func, *args, **kwargs)
    
    def maxViol(self, *args, **kwargs):
        return self._timed_call('maxViol',
                                self.planner.maxViol, *args, **kwargs)
    
    def incrementAugLag(self, *args, **kwargs):
        return self._timed_call('incrementAugLag',
                                self.planner.incrementAugLag, *args, **kwargs)
    
    def ilqrBreak(self, *args, **kwargs):
        return self._timed_call('ilqrBreak',
                                self.planner.ilqrBreak, *args, **kwargs)
    
    def outerBreak(self, *args, **kwargs):
        return self._timed_call('outerBreak',
                                self.planner.outerBreak, *args, **kwargs)
    
    def print_summary(self):
        print("\n" + "="*70)
        print("TIMING BREAKDOWN")
        print("="*70)
        
        total_time = sum(s.total for s in self.timings.values())
        
        # Sort by total time
        sorted_ops = sorted(self.timings.items(), key=lambda x: -x[1].total)
        
        print(f"{'Operation':<30} {'Total (s)':<12} {'%':<8} {'Calls':<8} {'Mean (ms)':<12}")
        print("-"*70)
        
        for name, stats in sorted_ops:
            pct = 100 * stats.total / total_time if total_time > 0 else 0
            print(f"{name:<30} {stats.total:<12.4f} {pct:<8.1f} {stats.count:<8} {stats.mean*1000:<12.3f}")
        
        print("-"*70)
        print(f"{'TOTAL':<30} {total_time:<12.4f}")
        
        return dict(self.timings)


def run_profiled_optimization(
    sat,
    x0: np.ndarray,
    h0: np.ndarray,
    os0,
    goals: GoalList,
    duration: float,
    settings: PlannerSettings,
    max_outer: int = 5,
    max_inner: int = 20,
) -> Tuple[float, Dict[str, TimingStats]]:
    """
    Run optimization with detailed timing.
    
    Returns total time and timing breakdown.
    """
    # Reset RW state
    for i, rw in enumerate(sat.rw_actuators):
        rw.h = h0[i] if i < len(h0) else 0.0
    
    # Modify settings for controlled iteration count
    settings.pass1.convergence.max_outer_iter = max_outer
    settings.pass1.convergence.max_inner_iter = max_inner
    settings.pass2.convergence.max_outer_iter = 3
    settings.pass2.convergence.max_inner_iter = 10
    
    # Create controller
    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=settings)
    
    # Time the full trajectory calculation
    t0 = time.perf_counter()
    traj = controller.calculate_trajectory(
        t_start=0.22, duration=duration, x_0=x0, os_0=os0, goals=goals, verbose=False
    )
    total_time = time.perf_counter() - t0
    
    return total_time, traj


def profile_with_python_alilqr(
    sat,
    x0: np.ndarray,
    h0: np.ndarray,
    os0,
    orb,
    goals: GoalList,
    duration: float,
    settings: PlannerSettings,
    max_outer: int = 3,
    max_inner: int = 10,
):
    """
    Profile using the Python ALILQR wrapper for detailed operation timing.
    """
    from ADCS.controller.plan_and_track_python_alilqr import Plan_and_Track_PythonALILQR
    from ADCS.controller.helpers.python_alilqr import PythonALILQR
    
    # Reset RW state
    for i, rw in enumerate(sat.rw_actuators):
        rw.h = h0[i] if i < len(h0) else 0.0
    
    # Modify settings
    settings.pass1.convergence.max_outer_iter = max_outer
    settings.pass1.convergence.max_inner_iter = max_inner
    settings.pass2.convergence.max_outer_iter = 2
    settings.pass2.convergence.max_inner_iter = 5
    
    # Create controller with verbose Python ALILQR
    controller = Plan_and_Track_PythonALILQR(
        est_sat=sat, 
        planner_settings=settings,
        verbose=False
    )
    
    # Collect timing per iteration
    iter_times = []
    
    def timing_callback(iter_data):
        # This is called after each iteration
        pass
    
    controller.set_iteration_callback(timing_callback)
    
    # Time components separately
    timings = defaultdict(TimingStats)
    
    # Time orbit propagation
    t0 = time.perf_counter()
    # Trigger orbit propagation by accessing internal method
    from ADCS.orbits.universal_constants import TimeConstants
    t_end = 0.22 + duration * TimeConstants.sec2cent
    dt_seconds = settings.dt_tp
    N = int(np.ceil(duration / dt_seconds)) + 1
    
    vecs = controller._propagate_environment(os0, 0.22, t_end, dt_seconds, N, goals)
    timings['orbit_propagation'].add(time.perf_counter() - t0)
    
    # Now time the actual optimization
    t0 = time.perf_counter()
    traj = controller.calculate_trajectory(
        t_start=0.22, duration=duration, x_0=x0, os_0=os0, goals=goals, verbose=False
    )
    timings['full_optimization'].add(time.perf_counter() - t0)
    
    # Get iteration-level data if available
    if hasattr(controller, 'last_optimization_result') and controller.last_optimization_result:
        result = controller.last_optimization_result
        timings['total_iterations'].add(float(result.total_inner_iters))
    
    return timings, traj


def run_test_suite():
    """Run profiling across different configurations."""
    
    np.random.seed(42)
    
    print("="*70)
    print("TRAJECTORY PLANNER PROFILING")
    print("="*70)
    
    # Create satellites
    print("\nCreating satellites...")
    sat_mtq_rw = create_beavercube2_cubesat(estimated=False)
    # For MTQ-only, use BeaverCube2 but we'll skip RW-specific tests
    sat_mtq_only = create_beavercube2_cubesat(estimated=False)  # Has RW but we can test MTQ behavior
    
    # Create orbit
    print("Creating orbit...")
    orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=300, use_J2=True, fast=True)
    orb.populate_environment(compute_B=True, compute_S=True)
    os0 = orb.get_os(0.22)
    
    # Initial conditions
    rng = np.random.default_rng(seed=1000)
    q0 = normalize(rng.standard_normal(4))
    w0 = normalize(rng.standard_normal(3)) * (0.5 * np.pi / 180.0)
    h0_rw = rng.uniform(-0.0001, 0.0001, size=1)
    h0_none = np.array([])
    
    # Goal quaternion (90 degree slew)
    half_angle = 45 * np.pi / 180
    q_rot = np.array([np.cos(half_angle), np.sin(half_angle), 0, 0])
    q_goal = normalize(quat_mult(q0, q_rot))
    
    x0_rw = np.concatenate([w0, q0, h0_rw])
    x0_mtq = np.concatenate([w0, q0])
    
    # Test configurations
    configs = [
        {
            'name': 'MTQ+RW, Quat Goal, 60s',
            'sat': sat_mtq_rw,
            'x0': x0_rw,
            'h0': h0_rw,
            'goals': GoalList({0.22: Fixed_Attitude_Goal(q_goal)}),
            'duration': 60,
        },
        {
            'name': 'MTQ+RW, Quat Goal, 120s',
            'sat': sat_mtq_rw,
            'x0': x0_rw,
            'h0': h0_rw,
            'goals': GoalList({0.22: Fixed_Attitude_Goal(q_goal)}),
            'duration': 120,
        },
        {
            'name': 'MTQ+RW, Nadir Goal, 60s',
            'sat': sat_mtq_rw,
            'x0': x0_rw,
            'h0': h0_rw,
            'goals': GoalList({0.22: Nadir_Goal()}),
            'duration': 60,
        },
        {
            'name': 'MTQ-only, Quat Goal, 60s',
            'sat': sat_mtq_only,
            'x0': x0_mtq,
            'h0': h0_none,
            'goals': GoalList({0.22: Fixed_Attitude_Goal(q_goal)}),
            'duration': 60,
        },
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"TEST: {cfg['name']}")
        print(f"{'='*70}")
        
        # Create settings
        if cfg['sat'].rw_actuators:
            settings = create_planner_settings(
                cfg['sat'],
                NormalizedPlannerConfig(
                    actuator_costs=NormalizedActuatorCosts(
                        mtq_cost=1.0, rw_torque_cost=5.0,
                        rw_momentum_cost=10.0, rw_stiction_cost=1.0,
                    ),
                    state_costs=NormalizedStateCosts(
                        angle_cost=1000.0, angle_terminal_cost=1000000.0,
                        ang_vel_cost=1000.0, ang_vel_terminal_cost=100000.0,
                    ),
                )
            )
            settings.rw_AM_weight = 1e4
            settings.RWh_ok_mult = 0.5
        else:
            settings = PlannerSettings(est_sat=cfg['sat'], bdot_on=0)
        
        # Run with C++ (fast)
        print("\nRunning C++ optimization...")
        t_cpp, traj_cpp = run_profiled_optimization(
            cfg['sat'], cfg['x0'], cfg['h0'], os0, cfg['goals'],
            cfg['duration'], settings, max_outer=5, max_inner=20
        )
        print(f"  C++ total time: {t_cpp:.3f}s")
        
        # Run with Python ALILQR (instrumented)
        print("\nRunning Python ALILQR (for breakdown)...")
        try:
            timings, traj_py = profile_with_python_alilqr(
                cfg['sat'], cfg['x0'], cfg['h0'], os0, orb, cfg['goals'],
                cfg['duration'], settings, max_outer=3, max_inner=10
            )
            
            print(f"  Orbit propagation: {timings['orbit_propagation'].total:.3f}s")
            print(f"  Full optimization: {timings['full_optimization'].total:.3f}s")
            
        except Exception as e:
            print(f"  Python ALILQR failed: {e}")
            timings = {}
        
        results.append({
            'name': cfg['name'],
            'cpp_time': t_cpp,
            'timings': dict(timings),
            'N': int(np.ceil(cfg['duration'] / settings.dt_tp)) + 1,
            'n_actuators': len(cfg['sat'].actuators),
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Config':<35} {'N':<6} {'Acts':<6} {'Time (s)':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['name']:<35} {r['N']:<6} {r['n_actuators']:<6} {r['cpp_time']:<10.3f}")
    
    return results


def profile_single_iteration():
    """
    Profile a single optimization iteration in detail.
    
    This gives per-operation timing within one backward+forward pass.
    """
    print("\n" + "="*70)
    print("SINGLE ITERATION PROFILING")
    print("="*70)
    
    np.random.seed(42)
    
    # Setup
    sat = create_beavercube2_cubesat(estimated=False)
    orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=120, use_J2=True, fast=True)
    orb.populate_environment(compute_B=True, compute_S=True)
    os0 = orb.get_os(0.22)
    
    rng = np.random.default_rng(seed=1000)
    q0 = normalize(rng.standard_normal(4))
    w0 = normalize(rng.standard_normal(3)) * (0.5 * np.pi / 180.0)
    h0 = rng.uniform(-0.0001, 0.0001, size=1)
    
    half_angle = 45 * np.pi / 180
    q_rot = np.array([np.cos(half_angle), np.sin(half_angle), 0, 0])
    q_goal = normalize(quat_mult(q0, q_rot))
    
    x0 = np.concatenate([w0, q0, h0])
    goals = GoalList({0.22: Fixed_Attitude_Goal(q_goal)})
    
    for i, rw in enumerate(sat.rw_actuators):
        rw.h = h0[i]
    
    settings = create_planner_settings(
        sat,
        NormalizedPlannerConfig(
            actuator_costs=NormalizedActuatorCosts(
                mtq_cost=1.0, rw_torque_cost=5.0,
            ),
            state_costs=NormalizedStateCosts(
                angle_cost=1000.0, angle_terminal_cost=1000000.0,
                ang_vel_cost=1000.0, ang_vel_terminal_cost=100000.0,
            ),
        )
    )
    settings.rw_AM_weight = 1e4
    settings.RWh_ok_mult = 0.5
    
    # Just run 1 outer, 5 inner iterations
    settings.pass1.convergence.max_outer_iter = 1
    settings.pass1.convergence.max_inner_iter = 5
    settings.pass2.convergence.max_outer_iter = 1
    settings.pass2.convergence.max_inner_iter = 3
    
    from ADCS.controller.plan_and_track_python_alilqr import Plan_and_Track_PythonALILQR
    
    # Track timing per iteration
    iter_timings = []
    last_time = [time.perf_counter()]
    
    def timing_callback(iter_data):
        now = time.perf_counter()
        elapsed = now - last_time[0]
        iter_timings.append({
            'outer': iter_data.outer_iter,
            'inner': iter_data.inner_iter,
            'elapsed': elapsed,
            'cost': iter_data.LA,
            'cmax': iter_data.cmax,
        })
        last_time[0] = now
    
    controller = Plan_and_Track_PythonALILQR(est_sat=sat, planner_settings=settings, verbose=True)
    controller.set_iteration_callback(timing_callback)
    
    t0 = time.perf_counter()
    traj = controller.calculate_trajectory(
        t_start=0.22, duration=60, x_0=x0, os_0=os0, goals=goals, verbose=True
    )
    total = time.perf_counter() - t0
    
    print(f"\nTotal time: {total:.3f}s")
    print(f"\nPer-iteration timing:")
    print(f"{'Outer':<8} {'Inner':<8} {'Time (ms)':<12} {'Cost':<15} {'Cmax':<12}")
    print("-"*60)
    for it in iter_timings:
        print(f"{it['outer']:<8} {it['inner']:<8} {it['elapsed']*1000:<12.2f} {it['cost']:<15.2e} {it['cmax']:<12.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', action='store_true', help='Profile single iteration in detail')
    parser.add_argument('--suite', action='store_true', help='Run full test suite')
    args = parser.parse_args()
    
    if args.single:
        profile_single_iteration()
    elif args.suite:
        run_test_suite()
    else:
        # Default: run both
        run_test_suite()
        profile_single_iteration()
