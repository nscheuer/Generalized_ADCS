#!/usr/bin/env python3
"""
ALTRO Benchmark - Performance benchmarking of trajectory planner.

Tests performance across different:
- Horizon lengths (35s, 60s, 90s)
- Initial conditions (small angle, large angle, high omega)

Note: Duration must be > dt_tp (30s) to avoid edge case errors.

Usage:
    python benchmark_altro.py
    python benchmark_altro.py --quick      # Quick benchmark (fewer iterations)
    python benchmark_altro.py --save       # Save results to JSON
"""

import sys
import os
import numpy as np
import time
import argparse
from dataclasses import dataclass
from typing import List
import json

# Add project root to path BEFORE any ADCS imports
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))


@dataclass
class BenchmarkResult:
    name: str
    duration_s: float
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    iterations: int
    converged: int


def create_test_setup():
    """Create satellite, orbit, and controller for benchmarking."""
    from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
    from ADCS.orbits.orbit import Orbit
    from ADCS.orbits.orbital_state import Orbital_State
    from ADCS.orbits.ephemeris import Ephemeris
    from ADCS.orbits.universal_constants import TimeConstants
    from ADCS.controller.helpers import PlannerSettings
    from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR

    # Create satellite
    real_sat = create_beavercube2_cubesat(estimated=False)
    if hasattr(real_sat, 'rw_actuators') and len(real_sat.rw_actuators) > 0:
        real_sat.rw_actuators[0].h = 0.0

    # Create orbit - use_J2=True required
    ephem = Ephemeris()
    start_time = 0.22
    end_time = start_time + 300 * TimeConstants.sec2cent  # 5 min trajectory support
    R = 7000 * np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    orb = Orbit(os0=os0, end_time=end_time, dt=1, use_J2=True, fast=False)
    os_0 = orb.get_os(start_time)

    # Create controller
    planner_settings = PlannerSettings(
        est_sat=real_sat,
        bdot_on=2,
        dt_tp=30,
        dt_tvlqr=1,
    )
    controller = Plan_and_Track_LQR(
        est_sat=real_sat,
        planner_settings=planner_settings,
    )

    return real_sat, orb, os_0, controller, start_time


def get_default_goals(start_time):
    """Create default ECI pointing goal."""
    from ADCS.CONOPS.goallist import GoalList
    from ADCS.CONOPS.goals import ECI_Goal
    from ADCS.helpers.math_helpers import normalize

    goal_vec = normalize(np.array([0, 0, 1]))
    goal = ECI_Goal(goal_vec)
    goals = GoalList({start_time: goal})
    return goals


def benchmark_configuration(controller, os_0, start_time, goals,
                           duration: float, x0: np.ndarray, name: str,
                           iterations: int = 3) -> BenchmarkResult:
    """Benchmark a single configuration."""

    times = []
    converged = 0

    for i in range(iterations):
        # Add small noise for variety
        x0_test = x0.copy()
        x0_test[:3] += np.random.randn(3) * 0.001

        start = time.perf_counter()
        try:
            trajectory = controller.calculate_trajectory(
                t_start=start_time,
                duration=duration,
                x_0=x0_test,
                os_0=os_0,
                goals=goals,
                verbose=0
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

            if not np.any(np.isnan(trajectory.states)):
                converged += 1
        except Exception as e:
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

    return BenchmarkResult(
        name=name,
        duration_s=duration,
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times),
        iterations=iterations,
        converged=converged
    )


def run_benchmark(quick: bool = False, save: bool = False):
    """Run comprehensive benchmark."""

    print("=" * 70)
    print("ALTRO Performance Benchmark")
    print("=" * 70)

    iterations = 2 if quick else 3

    # Setup
    print("\nSetting up...")
    real_sat, orb, os_0, controller, start_time = create_test_setup()
    goals = get_default_goals(start_time)
    print(f"  Satellite: BeaverCube2")
    print(f"  MTQs: {len(real_sat.mtq_actuators)}, RWs: {len(real_sat.rw_actuators)}")

    # Initial conditions to test - state format: [omega(3), q(4), h(1)]
    initial_conditions = {
        "small_angle": np.array([0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 1.0, 0.0]),
        "large_angle": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # 180 deg about z
        "high_omega": np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0, 0.0]),   # ~6 deg/s
    }

    # Durations to test (must be > dt_tp=30)
    durations = [35, 60] if quick else [35, 60, 90]

    results: List[BenchmarkResult] = []

    # Run benchmarks
    total_tests = len(durations) * len(initial_conditions)
    test_num = 0

    for duration in durations:
        for ic_name, x0 in initial_conditions.items():
            test_num += 1
            name = f"{duration}s_{ic_name}"
            print(f"\n[{test_num}/{total_tests}] Benchmarking: {name}")

            result = benchmark_configuration(
                controller, os_0, start_time, goals,
                duration, x0, name, iterations
            )
            results.append(result)

            print(f"    Mean: {result.mean_time_ms:.1f}ms "
                  f"(std: {result.std_time_ms:.1f}ms) "
                  f"[{result.converged}/{result.iterations} converged]")

    # Print summary
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    print(f"{'Configuration':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)

    for r in results:
        print(f"{r.name:<25} {r.mean_time_ms:>10.1f} "
              f"{r.std_time_ms:>10.1f} {r.min_time_ms:>10.1f} {r.max_time_ms:>10.1f}")

    print("-" * 70)

    # Compute averages by duration
    print("\nAverage by duration:")
    for duration in durations:
        dur_results = [r for r in results if r.duration_s == duration]
        avg_time = np.mean([r.mean_time_ms for r in dur_results])
        print(f"  {duration}s: {avg_time:.1f}ms average")

    # Save results
    if save:
        output_file = os.path.join(
            os.path.dirname(__file__),
            f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        results_dict = [
            {
                "name": r.name,
                "duration_s": r.duration_s,
                "mean_time_ms": r.mean_time_ms,
                "std_time_ms": r.std_time_ms,
                "min_time_ms": r.min_time_ms,
                "max_time_ms": r.max_time_ms,
                "iterations": r.iterations,
                "converged": r.converged
            }
            for r in results
        ]
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 70)
    return results


def main():
    parser = argparse.ArgumentParser(description="ALTRO benchmark")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick benchmark (fewer iterations)")
    parser.add_argument("--save", "-s", action="store_true",
                        help="Save results to JSON file")
    args = parser.parse_args()

    run_benchmark(quick=args.quick, save=args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
