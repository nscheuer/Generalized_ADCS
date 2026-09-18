"""Fast goal performance checks.

This module intentionally does not depend on pytest-benchmark. It can be run
directly in CI and compares each benchmark against a checked-in baseline after
normalizing by a small CPU/Numpy calibration workload:

    python testing/benchmarks/benchmark_goals.py --compare

To refresh the baseline after an intentional performance change:

    python testing/benchmarks/benchmark_goals.py --update-baseline
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

from ADCS.CONOPS.goallist import GoalList
from ADCS.CONOPS.goals import AntiBField_Goal, AntiSun_Goal, AntiVelocity_Goal, BField_Goal, Coordinate_Goal, ECI_Goal, Fixed_Attitude_Goal, LVLH_Tangential_Goal, Nadir_Goal, No_Goal, PerpBField_Goal, Sun_Goal, Velocity_Goal, Zenith_Goal
from ADCS.helpers.math_helpers import normalize
from testing.test_goals._helpers import make_multi_boresight_satellite, make_test_orbital_state


BASELINE_PATH = REPO_ROOT / "testing" / "benchmarks" / "baselines" / "goals.json"
RESULT_PATH = REPO_ROOT / "testing" / "artifacts" / "benchmarks" / "goals_current.json"

DEFAULT_MIN_SECONDS = 0.020
DEFAULT_MAX_LOOPS = 8_192


@dataclass(frozen=True)
class Benchmark:
    name: str
    func: Callable[[], object]
    max_regression: float
    min_seconds: float = DEFAULT_MIN_SECONDS
    max_loops: int = DEFAULT_MAX_LOOPS


def _reference_case() -> dict[str, object]:
    os0 = make_test_orbital_state(
        R=np.array([7000.0, 20.0, -10.0]),
        V=np.array([0.2, 7.5, -0.1]),
        B=np.array([2.0e-5, -3.0e-5, 4.0e-5]),
        S=np.array([1.0e8, -2.0e8, 3.0e8]),
    )
    sat = make_multi_boresight_satellite()

    q = normalize(np.array([0.92, 0.17, -0.08, 0.34]))
    q_align_camera_to_x = normalize(np.array([1.0, 0.0, 1.0, 0.0]))
    camera = sat.get_boresight("camera")
    panel = sat.get_boresight("solar_panel")
    antenna = sat.get_boresight("antenna")

    no_goal = No_Goal()
    eci_goal = ECI_Goal(np.array([1.0, 0.0, 0.0]), boresight_name="camera")
    coordinate_goal = Coordinate_Goal(lat=40.0, lon=-40.0, alt=0.2, boresight_name="antenna")
    fixed_attitude_goal = Fixed_Attitude_Goal(np.array([0.4, -0.1, 0.2, 0.8]))

    orbital_goals = [
        Nadir_Goal(),
        Zenith_Goal(),
        Velocity_Goal(),
        AntiVelocity_Goal(),
        LVLH_Tangential_Goal(),
    ]
    environment_goals = [
        Sun_Goal(),
        AntiSun_Goal(),
        BField_Goal(),
        AntiBField_Goal(),
        PerpBField_Goal(),
    ]

    timeline = GoalList(
        {
            0.22 - 2.0e-6: no_goal,
            0.22: eci_goal,
            0.22 + 2.0e-6: coordinate_goal,
            0.22 + 4.0e-6: fixed_attitude_goal,
        }
    )

    return {
        "os0": os0,
        "q": q,
        "q_align_camera_to_x": q_align_camera_to_x,
        "camera": camera,
        "panel": panel,
        "antenna": antenna,
        "no_goal": no_goal,
        "eci_goal": eci_goal,
        "coordinate_goal": coordinate_goal,
        "fixed_attitude_goal": fixed_attitude_goal,
        "orbital_goals": orbital_goals,
        "environment_goals": environment_goals,
        "timeline": timeline,
    }


_REFERENCE = _reference_case()


def _warmup(benchmarks: list[Benchmark]) -> None:
    _calibration_work()
    for benchmark in benchmarks:
        benchmark.func()


def _calibration_work(iterations: int = 2_000) -> float:
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


def _orbital_goal_group() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    return tuple(goal.to_ref(_REFERENCE["os0"]) for goal in _REFERENCE["orbital_goals"])


def _environment_goal_group() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    return tuple(goal.to_ref(_REFERENCE["os0"]) for goal in _REFERENCE["environment_goals"])


def _make_benchmarks() -> list[Benchmark]:
    return [
        Benchmark(
            name="no_goal_to_ref",
            func=lambda: _REFERENCE["no_goal"].to_ref(_REFERENCE["os0"]),
            max_regression=1.50,
        ),
        Benchmark(
            name="no_goal_error",
            func=lambda: _REFERENCE["no_goal"].error(
                _REFERENCE["q"],
                _REFERENCE["camera"],
                _REFERENCE["os0"],
            ),
            max_regression=1.50,
        ),
        Benchmark(
            name="eci_goal_to_ref",
            func=lambda: _REFERENCE["eci_goal"].to_ref(_REFERENCE["os0"]),
            max_regression=1.50,
        ),
        Benchmark(
            name="eci_goal_error",
            func=lambda: _REFERENCE["eci_goal"].error(
                _REFERENCE["q_align_camera_to_x"],
                _REFERENCE["camera"],
                _REFERENCE["os0"],
            ),
            max_regression=1.50,
        ),
        Benchmark(
            name="coordinate_goal_to_ref",
            func=lambda: _REFERENCE["coordinate_goal"].to_ref(_REFERENCE["os0"]),
            max_regression=1.50,
            min_seconds=0.025,
            max_loops=4_096,
        ),
        Benchmark(
            name="coordinate_goal_error",
            func=lambda: _REFERENCE["coordinate_goal"].error(
                _REFERENCE["q"],
                _REFERENCE["antenna"],
                _REFERENCE["os0"],
            ),
            max_regression=1.50,
            min_seconds=0.025,
            max_loops=4_096,
        ),
        Benchmark(
            name="orbital_vector_goal_group",
            func=_orbital_goal_group,
            max_regression=1.50,
        ),
        Benchmark(
            name="environment_vector_goal_group",
            func=_environment_goal_group,
            max_regression=1.50,
            min_seconds=0.025,
            max_loops=4_096,
        ),
        Benchmark(
            name="fixed_attitude_goal_to_ref",
            func=lambda: _REFERENCE["fixed_attitude_goal"].to_ref(_REFERENCE["os0"]),
            max_regression=1.50,
        ),
        Benchmark(
            name="fixed_attitude_goal_error",
            func=lambda: _REFERENCE["fixed_attitude_goal"].error(
                _REFERENCE["q"],
                _REFERENCE["panel"],
                _REFERENCE["os0"],
            ),
            max_regression=1.50,
        ),
        Benchmark(
            name="goallist_get_active_goal",
            func=lambda: _REFERENCE["timeline"].get_active_goal(0.22 + 2.5e-6),
            max_regression=1.50,
        ),
        Benchmark(
            name="goallist_to_ref",
            func=lambda: _REFERENCE["timeline"].to_ref(0.22 + 2.5e-6, _REFERENCE["os0"]),
            max_regression=1.50,
            min_seconds=0.025,
            max_loops=4_096,
        ),
    ]


def run_benchmarks(samples: int) -> dict:
    benchmarks = _make_benchmarks()
    _warmup(benchmarks)
    calibration_seconds, calibration_loops = _median_time(_calibration_work, 0.040, 64, samples)
    results = {
        "schema_version": 1,
        "description": "Goals fast benchmark results normalized by calibration_seconds.",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "calibration_seconds": calibration_seconds,
        "calibration_loops": calibration_loops,
        "benchmarks": {},
    }
    for benchmark in benchmarks:
        seconds, loops = _median_time(benchmark.func, benchmark.min_seconds, benchmark.max_loops, samples)
        results["benchmarks"][benchmark.name] = {
            "seconds": seconds,
            "loops": loops,
            "normalized_seconds": seconds / calibration_seconds,
            "max_regression": benchmark.max_regression,
        }
    return results


def compare_results(current: dict, baseline: dict) -> list[str]:
    failures = []
    for name, current_entry in current["benchmarks"].items():
        if name not in baseline["benchmarks"]:
            failures.append(f"{name}: missing from baseline")
            continue
        baseline_entry = baseline["benchmarks"][name]
        threshold = float(baseline_entry.get("max_regression", current_entry["max_regression"]))
        ratio = float(current_entry["normalized_seconds"]) / float(baseline_entry["normalized_seconds"])
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
    print("\nGoals benchmarks")
    print(f"calibration: {results['calibration_seconds']:.6f} s")
    print("-" * 88)
    print(f"{'benchmark':36} {'time':>12} {'normalized':>14} {'vs baseline':>14}")
    print("-" * 88)
    for name, entry in results["benchmarks"].items():
        ratio = ""
        if baseline is not None and name in baseline["benchmarks"]:
            ratio = _color_ratio(entry["normalized_seconds"] / baseline["benchmarks"][name]["normalized_seconds"], use_color)
        print(f"{name:36} {entry['seconds'] * 1e6:10.1f} us {entry['normalized_seconds']:14.6e} {ratio:>14}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ADCS goal paths.")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--no-color", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 3:
        raise SystemExit("--samples must be at least 3")
    current = run_benchmarks(args.samples)
    write_json(args.output, current)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8")) if BASELINE_PATH.exists() else None
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
