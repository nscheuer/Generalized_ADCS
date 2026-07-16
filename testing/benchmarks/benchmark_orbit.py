"""Fast Orbit performance checks.

This module intentionally does not depend on pytest-benchmark.  It can be run
directly in CI and compares each benchmark against a checked-in baseline after
normalizing by a small CPU/Numpy calibration workload:

    python testing/benchmarks/benchmark_orbit.py --compare

To refresh the baseline after an intentional performance change:

    python testing/benchmarks/benchmark_orbit.py --update-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants


BASELINE_PATH = REPO_ROOT / "testing" / "benchmarks" / "baselines" / "orbit.json"
RESULT_PATH = REPO_ROOT / "testing" / "artifacts" / "benchmarks" / "orbit_current.json"

DEFAULT_MIN_SECONDS = 0.015
DEFAULT_MAX_LOOPS = 20_000


@dataclass(frozen=True)
class Benchmark:
    name: str
    func: Callable[[], object]
    max_regression: float
    min_seconds: float = DEFAULT_MIN_SECONDS
    max_loops: int = DEFAULT_MAX_LOOPS


def _reference_state() -> Orbital_State:
    ephem = Ephemeris()
    r = np.array([7078.137, 0.0, 0.0])
    v = np.array([0.0, np.sqrt(EarthConstants.mu_e / np.linalg.norm(r)), 0.0])
    return Orbital_State(ephem=ephem, J2000=0.22, R=r, V=v, fast=True)


_BASE_STATE = _reference_state()


def _reference_orbit() -> Orbit:
    os0 = _BASE_STATE.copy()
    dt = 60.0
    end_time = os0.J2000 + TimeConstants.sec2cent * dt * 4
    return Orbit(os0=os0, end_time=end_time, dt=dt, fast=True, verbose=False, zonal_J=2)


def _warmup(benchmarks: list[Benchmark]) -> None:
    _calibration_work()
    for benchmark in benchmarks:
        benchmark.func()


def _calibration_work(iterations: int = 2_000) -> float:
    """Small deterministic workload used to reduce hardware-to-hardware noise."""
    a = np.array(
        [
            [1.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            [0.01, 1.02, 0.03, 0.04, 0.05, 0.06],
            [0.01, 0.02, 1.03, 0.04, 0.05, 0.06],
            [0.01, 0.02, 0.03, 1.04, 0.05, 0.06],
            [0.01, 0.02, 0.03, 0.04, 1.05, 0.06],
            [0.01, 0.02, 0.03, 0.04, 0.05, 1.06],
        ],
        dtype=float,
    )
    b = np.eye(6) * 0.999
    for _ in range(iterations):
        a = a @ b
        b = b + np.eye(6) * 1e-12
    return float(a[0, 0])


def _time_callable(func: Callable[[], object], min_seconds: float, max_loops: int) -> tuple[float, int]:
    loops = 1
    while True:
        start = time.perf_counter()
        for _ in range(loops):
            result = func()
        elapsed = time.perf_counter() - start

        if result is None:
            raise RuntimeError("benchmark function returned None")

        if elapsed >= min_seconds or loops >= max_loops:
            return elapsed / loops, loops
        loops *= 2


def _median_time(func: Callable[[], object], min_seconds: float, max_loops: int, samples: int) -> tuple[float, int]:
    timings = []
    loops_used = 0
    for _ in range(samples):
        elapsed, loops = _time_callable(func, min_seconds=min_seconds, max_loops=max_loops)
        timings.append(elapsed)
        loops_used = max(loops_used, loops)
    return statistics.median(timings), loops_used


def _make_benchmarks(orbit: Orbit) -> list[Benchmark]:
    exact_time = float(orbit.times[2])
    interp_time = float((orbit.times[1] + orbit.times[2]) * 0.5)
    start_time = float(orbit.times[1])
    end_time = float(orbit.times[-1])
    sampled_times = [float(orbit.times[0]), interp_time, float(orbit.times[-1])]
    b_geo = np.vstack([state.geocentric for state in orbit.states.values()])
    b_ecef = orbit.geocentric_to_ecef_orbit(b_geo)

    return [
        Benchmark(
            name="construct_propagated_orbit_small",
            func=_reference_orbit,
            max_regression=1.50,
            min_seconds=0.040,
            max_loops=8,
        ),
        Benchmark(
            name="get_os_exact_node",
            func=lambda: orbit.get_os(exact_time),
            max_regression=1.50,
        ),
        Benchmark(
            name="get_os_interpolated",
            func=lambda: orbit.get_os(interp_time),
            max_regression=1.50,
            min_seconds=0.025,
            max_loops=1_024,
        ),
        Benchmark(
            name="get_range_precreated_states",
            func=lambda: orbit.get_range(start_time, end_time),
            max_regression=1.50,
            min_seconds=0.020,
            max_loops=1_024,
        ),
        Benchmark(
            name="new_orbit_from_times_small",
            func=lambda: orbit.new_orbit_from_times(sampled_times),
            max_regression=1.50,
            min_seconds=0.020,
            max_loops=512,
        ),
        Benchmark(
            name="geocentric_to_ecef_orbit",
            func=lambda: orbit.geocentric_to_ecef_orbit(b_geo),
            max_regression=1.50,
        ),
        Benchmark(
            name="ecef_to_eci_orbit",
            func=lambda: orbit.ecef_to_eci_orbit(b_ecef),
            max_regression=1.50,
        ),
        Benchmark(
            name="get_b_eci_orbit",
            func=orbit.get_b_eci_orbit,
            max_regression=1.50,
        ),
        Benchmark(
            name="get_vecs",
            func=orbit.get_vecs,
            max_regression=1.50,
        ),
    ]


def run_benchmarks(samples: int) -> dict:
    orbit = _reference_orbit()
    benchmarks = _make_benchmarks(orbit)
    _warmup(benchmarks)

    calibration_seconds, calibration_loops = _median_time(
        _calibration_work,
        min_seconds=0.040,
        max_loops=64,
        samples=samples,
    )

    results = {
        "schema_version": 1,
        "description": "Orbit fast benchmark results normalized by calibration_seconds.",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "calibration_seconds": calibration_seconds,
        "calibration_loops": calibration_loops,
        "benchmarks": {},
    }

    for benchmark in benchmarks:
        seconds, loops = _median_time(
            benchmark.func,
            min_seconds=benchmark.min_seconds,
            max_loops=benchmark.max_loops,
            samples=samples,
        )
        results["benchmarks"][benchmark.name] = {
            "seconds": seconds,
            "loops": loops,
            "normalized_seconds": seconds / calibration_seconds,
            "max_regression": benchmark.max_regression,
        }

    return results


def compare_results(current: dict, baseline: dict) -> list[str]:
    failures = []
    baseline_benchmarks = baseline["benchmarks"]

    for name, current_entry in current["benchmarks"].items():
        if name not in baseline_benchmarks:
            failures.append(f"{name}: missing from baseline")
            continue

        baseline_entry = baseline_benchmarks[name]
        threshold = float(baseline_entry.get("max_regression", current_entry["max_regression"]))
        baseline_normalized = float(baseline_entry["normalized_seconds"])
        current_normalized = float(current_entry["normalized_seconds"])
        ratio = current_normalized / baseline_normalized
        if ratio > threshold:
            failures.append(
                f"{name}: {ratio:.2f}x baseline normalized time "
                f"(allowed {threshold:.2f}x; current {current_entry['seconds'] * 1e6:.1f} us)"
            )

    return failures


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _color_ratio(ratio: float, use_color: bool) -> str:
    label = f"{ratio:.2f}x"
    if not use_color:
        return label

    if 0.8 <= ratio <= 1.2:
        color = "\033[32m"
    elif 0.5 <= ratio < 0.8 or 1.2 < ratio <= 1.5:
        color = "\033[33m"
    else:
        color = "\033[31m"
    return f"{color}{label}\033[0m"


def print_summary(results: dict, baseline: dict | None = None, use_color: bool = True) -> None:
    print("\nOrbit benchmarks")
    print(f"calibration: {results['calibration_seconds']:.6f} s")
    print("-" * 88)
    print(f"{'benchmark':36} {'time':>12} {'normalized':>14} {'vs baseline':>14}")
    print("-" * 88)

    for name, entry in results["benchmarks"].items():
        normalized = entry["normalized_seconds"]
        ratio = ""
        if baseline is not None and name in baseline["benchmarks"]:
            ratio_value = normalized / baseline["benchmarks"][name]["normalized_seconds"]
            ratio = _color_ratio(ratio_value, use_color=use_color)
        print(f"{name:36} {entry['seconds'] * 1e6:10.1f} us {normalized:14.6e} {ratio:>14}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ADCS.orbits.orbit.Orbit.")
    parser.add_argument("--compare", action="store_true", help="fail if calibrated timings regress versus baseline")
    parser.add_argument("--update-baseline", action="store_true", help="replace the checked-in baseline with this run")
    parser.add_argument("--samples", type=int, default=3, help="median sample count per benchmark")
    parser.add_argument("--output", type=Path, default=RESULT_PATH, help="where to write current benchmark JSON")
    parser.add_argument("--no-color", action="store_true", help="disable colored ratio output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 3:
        raise SystemExit("--samples must be at least 3")

    current = run_benchmarks(samples=args.samples)
    write_json(args.output, current)

    baseline = None
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    if args.update_baseline:
        write_json(BASELINE_PATH, current)
        print_summary(current, use_color=not args.no_color and "NO_COLOR" not in os.environ)
        print(f"\nUpdated baseline: {BASELINE_PATH.relative_to(REPO_ROOT)}")
        return 0

    print_summary(current, baseline, use_color=not args.no_color and "NO_COLOR" not in os.environ)

    if args.compare:
        if baseline is None:
            print(f"\nMissing baseline: {BASELINE_PATH.relative_to(REPO_ROOT)}")
            return 2
        failures = compare_results(current, baseline)
        if failures:
            print("\nPerformance regressions detected:")
            for failure in failures:
                print(f"  - {failure}")
            return 1
        print("\nNo calibrated performance regressions detected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
