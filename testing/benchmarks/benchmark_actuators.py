"""Fast actuator performance checks.

This module intentionally does not depend on pytest-benchmark. It can be run
directly in CI and compares each benchmark against a checked-in baseline after
normalizing by a small CPU/Numpy calibration workload:

    python testing/benchmarks/benchmark_actuators.py --compare

To refresh the baseline after an intentional performance change:

    python testing/benchmarks/benchmark_actuators.py --update-baseline
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

from ADCS.helpers.math_constants import MathConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.state import State


BASELINE_PATH = REPO_ROOT / "testing" / "benchmarks" / "baselines" / "actuators.json"
RESULT_PATH = REPO_ROOT / "testing" / "artifacts" / "benchmarks" / "actuators_current.json"

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
    os0 = Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 8.0, 0.0]),
        B=np.array([2.0e-5, -3.0e-5, 4.0e-5]),
        fast=True,
    )

    q = np.array([0.97, 0.15, -0.08, 0.16])
    x = State(w=[2.0e-3, -1.0e-3, 1.5e-3], q=q / np.linalg.norm(q))

    mtq = MTQ(
        axis=MathConstants.unitvecs[0],
        max_torque=0.4,
        bias=Bias(bias=0.0, std_bias=0.0),
        noise=Noise(noise=np.zeros(3), std_noise=np.zeros(3)),
    )
    rw = RW(
        axis=MathConstants.unitvecs[1],
        max_torque=7.0e-3,
        J=1.0e-3,
        h=5.0e-3,
        h_max=16.2e-3,
        bias=Bias(bias=0.0, std_bias=0.0),
        noise=Noise(noise=0.0, std_noise=0.0),
        h_meas_noise=Noise(noise=0.0, std_noise=0.0),
    )

    return {
        "os0": os0,
        "x": x,
        "mtq": mtq,
        "rw": rw,
        "u_mtq": 0.15,
        "u_rw": 2.5e-3,
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


def _make_benchmarks() -> list[Benchmark]:
    mtq = _REFERENCE["mtq"]
    rw = _REFERENCE["rw"]
    x = _REFERENCE["x"]
    os0 = _REFERENCE["os0"]
    u_mtq = _REFERENCE["u_mtq"]
    u_rw = _REFERENCE["u_rw"]
    return [
        Benchmark(
            name="mtq_torque",
            func=lambda: mtq.torque(u_mtq, x, os0),
            max_regression=1.50,
        ),
        Benchmark(
            name="mtq_storage_torque",
            func=lambda: mtq.storage_torque(u_mtq, x, os0),
            max_regression=1.50,
        ),
        Benchmark(
            name="mtq_jacobians",
            func=lambda: mtq.jacobians(u_mtq, x, os0),
            max_regression=1.50,
            min_seconds=0.025,
            max_loops=4_096,
        ),
        Benchmark(
            name="mtq_hessians",
            func=lambda: mtq.hessians(u_mtq, x, os0),
            max_regression=1.50,
            min_seconds=0.025,
            max_loops=2_048,
        ),
        Benchmark(
            name="rw_effective_command",
            func=lambda: rw._effective_command(u_rw, os0),
            max_regression=1.50,
        ),
        Benchmark(
            name="rw_torque",
            func=lambda: rw.torque(u_rw, x, os0),
            max_regression=1.50,
        ),
        Benchmark(
            name="rw_storage_torque",
            func=lambda: rw.storage_torque(u_rw, x, os0),
            max_regression=1.50,
        ),
        Benchmark(
            name="rw_measure_momentum",
            func=rw.measure_momentum,
            max_regression=1.50,
        ),
        Benchmark(
            name="rw_measure_momentum_noiseless",
            func=rw.measure_momentum_noiseless,
            max_regression=1.50,
        ),
        Benchmark(
            name="rw_dtorq__du",
            func=lambda: rw.dtorq__du(u_rw, x, os0),
            max_regression=1.50,
        ),
    ]


def run_benchmarks(samples: int) -> dict:
    benchmarks = _make_benchmarks()
    _warmup(benchmarks)
    calibration_seconds, calibration_loops = _median_time(_calibration_work, 0.040, 64, samples)
    results = {
        "schema_version": 1,
        "description": "Actuators fast benchmark results normalized by calibration_seconds.",
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
    print("\nActuators benchmarks")
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
    parser = argparse.ArgumentParser(description="Benchmark ADCS actuator paths.")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--samples", type=int, default=3)
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
