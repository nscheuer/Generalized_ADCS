"""
Performance benchmarks for the ALTRO trajectory planner.

This module provides timing benchmarks for the trajectory optimization pipeline
under various configurations. Use these benchmarks to:
- Profile planner performance on your hardware
- Compare different solver configurations
- Identify bottlenecks in the planning pipeline

Usage:
    python benchmark_planner.py              # Run all benchmarks
    python benchmark_planner.py --quick      # Run quick benchmarks only
    python benchmark_planner.py --detailed   # Include detailed per-iteration timing
"""
from __future__ import annotations

import sys
import os
import time
import argparse
import numpy as np
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass, field

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.CONOPS.goals import ECI_Goal, No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings, CostWeights
from ADCS.controller.helpers.planner_subsettings import ConvergenceConfig, SolverPassConfig, AugLagConfig
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import RW, MTQ
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize


@dataclass
class BenchmarkResult:
    """Container for benchmark timing results."""
    name: str
    duration_seconds: float
    iterations: int
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    config: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Total: {self.duration_seconds:.3f}s over {self.iterations} runs\n"
            f"  Mean: {self.mean_time*1000:.1f}ms, Std: {self.std_time*1000:.1f}ms\n"
            f"  Min: {self.min_time*1000:.1f}ms, Max: {self.max_time*1000:.1f}ms"
        )


def create_rw_satellite() -> Satellite:
    """Create a standard RW-only satellite for benchmarking."""
    rw_max_torque = 0.01
    rw_J = 0.001
    rw_hmax = 0.05
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=0.0, h_max=rw_hmax)
           for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    return Satellite(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.1, 0.1]),
        actuators=rws,
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )


def create_mtq_satellite() -> Satellite:
    """Create a MTQ-only satellite for benchmarking."""
    mtq_max = 0.5
    mtqs = [MTQ(axis=j, max_moment=mtq_max) for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    return Satellite(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.1, 0.1]),
        actuators=mtqs,
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )


def create_mixed_satellite() -> Satellite:
    """Create a mixed MTQ+RW satellite for benchmarking."""
    rw_max_torque = 0.01
    rw_J = 0.001
    rw_hmax = 0.05
    mtq_max = 0.5

    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=0.0, h_max=rw_hmax)
           for j in MathConstants.unitvecs]
    mtqs = [MTQ(axis=j, max_moment=mtq_max) for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    return Satellite(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.1, 0.1]),
        actuators=rws + mtqs,
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )


def create_test_orbit(duration: int = 200) -> Tuple[Orbit, Orbital_State]:
    """Create test orbit for benchmarking.

    Uses simple time-stepped copies instead of full propagation for speed.
    """
    ephem = Ephemeris()
    R = 6778 * np.array([1, 0, 0])
    V = np.array([0, 7.67, 0])
    os0 = Orbital_State(
        ephem=ephem,
        J2000=0.22,
        R=R, V=V,
        B=np.array([2e-5, 1e-5, 3e-5]),
        S=np.array([1e5, 0, 0]),
        rho=0.0
    )
    # Use simple copies with time updates (fast) instead of propagation
    orbs = [os0.copy() for _ in range(duration + 10)]
    for j in range(len(orbs)):
        orbs[j].J2000 = os0.J2000 + j * TimeConstants.sec2cent
    return Orbit(orbs), os0


def create_initial_state(w_mag: float = 0.0) -> np.ndarray:
    """Create initial state with optional angular velocity."""
    w0 = np.array([w_mag, 0.0, 0.0])
    q0 = normalize(np.array([0, 0, 0, 1]))
    h0 = np.array([0.0, 0.0, 0.0])
    return np.concatenate([w0, q0, h0])


def run_planning_benchmark(
    sat: Satellite,
    planner_settings: PlannerSettings,
    x0: np.ndarray,
    os0: Orbital_State,
    goal: ECI_Goal,
    duration: float,
    iterations: int = 5,
    warmup: int = 0
) -> BenchmarkResult:
    """Run trajectory planning benchmark with timing."""

    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=planner_settings)
    goals = GoalList({os0.J2000: goal})

    times = []

    # Warmup runs (skip in quick mode by default)
    for _ in range(warmup):
        controller.calculate_trajectory(
            t_start=os0.J2000,
            duration=duration,
            x_0=x0,
            os_0=os0,
            goals=goals,
            verbose=False
        )

    # Timed runs
    for _ in range(iterations):
        start = time.perf_counter()
        controller.calculate_trajectory(
            t_start=os0.J2000,
            duration=duration,
            x_0=x0,
            os_0=os0,
            goals=goals,
            verbose=False
        )
        end = time.perf_counter()
        times.append(end - start)

    times_arr = np.array(times)

    return BenchmarkResult(
        name="",  # Set by caller
        duration_seconds=sum(times),
        iterations=iterations,
        mean_time=np.mean(times_arr),
        std_time=np.std(times_arr),
        min_time=np.min(times_arr),
        max_time=np.max(times_arr)
    )


# ============================================================================
# Benchmark Configurations
# ============================================================================

def benchmark_trajectory_durations(iterations: int = 3, quick: bool = False) -> List[BenchmarkResult]:
    """Benchmark planning time vs trajectory duration."""
    results = []
    sat = create_rw_satellite()
    _, os0 = create_test_orbit(150)
    x0 = create_initial_state()
    goal = ECI_Goal(normalize(np.array([1, 0, 0])))

    durations = [20, 40] if quick else [30, 60, 90, 120]

    for dur in durations:
        planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=1.0)
        result = run_planning_benchmark(
            sat=sat,
            planner_settings=planner_settings,
            x0=x0,
            os0=os0,
            goal=goal,
            duration=float(dur),
            iterations=iterations
        )
        result.name = f"Duration {dur}s"
        result.config = {"duration": dur}
        results.append(result)

    return results


def benchmark_timestep_sizes(iterations: int = 3, quick: bool = False) -> List[BenchmarkResult]:
    """Benchmark planning time vs dt_tp (trajectory planner timestep)."""
    results = []
    sat = create_rw_satellite()
    _, os0 = create_test_orbit(100)
    x0 = create_initial_state()
    goal = ECI_Goal(normalize(np.array([1, 0, 0])))

    timesteps = [1.0, 2.0] if quick else [0.5, 1.0, 2.0, 5.0]

    for dt in timesteps:
        planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=dt)
        result = run_planning_benchmark(
            sat=sat,
            planner_settings=planner_settings,
            x0=x0,
            os0=os0,
            goal=goal,
            duration=30.0 if quick else 60.0,
            iterations=iterations
        )
        result.name = f"dt_tp={dt}s"
        result.config = {"dt_tp": dt}
        results.append(result)

    return results


def benchmark_actuator_types(iterations: int = 3, quick: bool = False) -> List[BenchmarkResult]:
    """Benchmark planning time for different actuator configurations."""
    results = []
    _, os0 = create_test_orbit(100)
    x0 = create_initial_state()
    goal = ECI_Goal(normalize(np.array([1, 0, 0])))

    configs = [
        ("RW-only", create_rw_satellite),
    ] if quick else [
        ("RW-only", create_rw_satellite),
        ("MTQ-only", create_mtq_satellite),
        ("Mixed RW+MTQ", create_mixed_satellite),
    ]

    for name, sat_factory in configs:
        sat = sat_factory()

        # Adjust initial state for different control dimensions
        if "RW" in name:
            x0 = create_initial_state()
        else:
            # MTQ-only doesn't have RW momentum states
            x0 = np.concatenate([np.zeros(3), normalize(np.array([0, 0, 0, 1]))])
            # But our satellite model still expects 10 states
            x0 = np.concatenate([x0, np.zeros(3)])

        planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=1.0)

        result = run_planning_benchmark(
            sat=sat,
            planner_settings=planner_settings,
            x0=x0,
            os0=os0,
            goal=goal,
            duration=30.0 if quick else 60.0,
            iterations=iterations
        )
        result.name = name
        result.config = {"actuator_type": name}
        results.append(result)

    return results


def benchmark_solver_iterations(iterations: int = 3) -> List[BenchmarkResult]:
    """Benchmark planning time vs max solver iterations."""
    results = []
    sat = create_rw_satellite()
    _, os0 = create_test_orbit(200)
    x0 = create_initial_state()
    goal = ECI_Goal(normalize(np.array([1, 0, 0])))

    iter_configs = [
        ("Light (10/50)", 10, 50),
        ("Medium (20/150)", 20, 150),
        ("Heavy (30/250)", 30, 250),
    ]

    for name, outer, inner in iter_configs:
        converge = ConvergenceConfig(max_outer_iter=outer, max_inner_iter=inner)
        pass_config = SolverPassConfig(convergence=converge)

        planner_settings = PlannerSettings(
            est_sat=sat,
            bdot_on=0,
            dt_tp=1.0,
            pass1_config=pass_config,
            pass2_config=pass_config
        )

        result = run_planning_benchmark(
            sat=sat,
            planner_settings=planner_settings,
            x0=x0,
            os0=os0,
            goal=goal,
            duration=60.0,
            iterations=iterations
        )
        result.name = name
        result.config = {"max_outer": outer, "max_inner": inner}
        results.append(result)

    return results


def benchmark_maneuver_sizes(iterations: int = 3) -> List[BenchmarkResult]:
    """Benchmark planning time for different maneuver sizes."""
    results = []
    sat = create_rw_satellite()
    _, os0 = create_test_orbit(200)

    maneuvers = [
        ("Small (10 deg)", normalize(np.array([0.17, 0.0, 1.0]))),  # ~10 deg
        ("Medium (45 deg)", normalize(np.array([1.0, 0.0, 1.0]))),  # ~45 deg
        ("Large (90 deg)", normalize(np.array([1.0, 0.0, 0.0]))),   # ~90 deg
        ("Very Large (180 deg)", normalize(np.array([0.0, 0.0, -1.0]))),  # ~180 deg
    ]

    for name, goal_vec in maneuvers:
        x0 = create_initial_state()
        goal = ECI_Goal(goal_vec)

        planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=1.0)

        result = run_planning_benchmark(
            sat=sat,
            planner_settings=planner_settings,
            x0=x0,
            os0=os0,
            goal=goal,
            duration=90.0,
            iterations=iterations
        )
        result.name = name
        result.config = {"maneuver": name, "goal_vec": goal_vec.tolist()}
        results.append(result)

    return results


# ============================================================================
# Main Benchmark Runner
# ============================================================================

def run_all_benchmarks(quick: bool = False, detailed: bool = False) -> Dict[str, List[BenchmarkResult]]:
    """Run all benchmark suites."""
    iterations = 1 if quick else 3

    print("=" * 70)
    print("ALTRO TRAJECTORY PLANNER BENCHMARKS")
    print("=" * 70)
    print(f"Running with {iterations} iterations per config")
    if quick:
        print("(Quick mode - reduced configs)")
    print()

    all_results = {}

    # Trajectory Duration Benchmarks
    print("Benchmarking trajectory durations...")
    results = benchmark_trajectory_durations(iterations, quick)
    all_results["Trajectory Duration"] = results
    print_benchmark_results("Trajectory Duration", results)

    # Timestep Size Benchmarks
    print("\nBenchmarking timestep sizes...")
    results = benchmark_timestep_sizes(iterations, quick)
    all_results["Timestep Size"] = results
    print_benchmark_results("Timestep Size", results)

    # Actuator Type Benchmarks
    print("\nBenchmarking actuator types...")
    results = benchmark_actuator_types(iterations, quick)
    all_results["Actuator Types"] = results
    print_benchmark_results("Actuator Types", results)

    if not quick:
        # Solver Iteration Benchmarks
        print("\nBenchmarking solver iterations...")
        results = benchmark_solver_iterations(iterations)
        all_results["Solver Iterations"] = results
        print_benchmark_results("Solver Iterations", results)

        # Maneuver Size Benchmarks
        print("\nBenchmarking maneuver sizes...")
        results = benchmark_maneuver_sizes(iterations)
        all_results["Maneuver Sizes"] = results
        print_benchmark_results("Maneuver Sizes", results)

    return all_results


def print_benchmark_results(suite_name: str, results: List[BenchmarkResult]) -> None:
    """Print formatted benchmark results."""
    print(f"\n--- {suite_name} ---")
    print(f"{'Config':<25} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-" * 73)
    for r in results:
        print(f"{r.name:<25} {r.mean_time*1000:>10.1f}  {r.std_time*1000:>10.1f}  "
              f"{r.min_time*1000:>10.1f}  {r.max_time*1000:>10.1f}")


def print_summary(all_results: Dict[str, List[BenchmarkResult]]) -> None:
    """Print summary of all benchmarks."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_runs = 0
    total_time = 0.0

    for suite_name, results in all_results.items():
        for r in results:
            total_runs += r.iterations
            total_time += r.duration_seconds

    print(f"Total benchmark runs: {total_runs}")
    print(f"Total benchmark time: {total_time:.1f}s")
    print()

    # Find slowest and fastest configs
    all_flat = [(r.name, r.mean_time) for results in all_results.values() for r in results]
    all_flat.sort(key=lambda x: x[1])

    print("Fastest configurations:")
    for name, t in all_flat[:3]:
        print(f"  {name}: {t*1000:.1f}ms")

    print("\nSlowest configurations:")
    for name, t in all_flat[-3:]:
        print(f"  {name}: {t*1000:.1f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ALTRO planner benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks only")
    parser.add_argument("--detailed", action="store_true", help="Include detailed timing")
    args = parser.parse_args()

    all_results = run_all_benchmarks(quick=args.quick, detailed=args.detailed)
    print_summary(all_results)
