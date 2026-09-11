"""Microbenchmarks for the structured spacecraft state model."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.state import EstimatorState, State
from testing.test_estimators.ukf.helpers import make_baseline_sensors, make_satellites, make_ukf


BASELINE_PATH = REPO_ROOT / "testing" / "benchmarks" / "baselines" / "state.json"
RESULT_PATH = REPO_ROOT / "testing" / "artifacts" / "benchmarks" / "state_current.json"

MAX_REGRESSION = 1.5


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
        a = a @ b + 1.0e-6
    return float(a[0, 0])


def _measure_calibration(samples: int = 5) -> tuple[float, int]:
    timings = []
    loops_used = 1
    loops = 1
    while True:
        start = time.perf_counter()
        for _ in range(loops):
            result = _calibration_work()
        elapsed = time.perf_counter() - start
        if result is None:
            raise RuntimeError("calibration returned None")
        if elapsed >= 0.15 or loops >= 256:
            loops_used = loops
            break
        loops *= 2
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(loops_used):
            result = _calibration_work()
        elapsed = time.perf_counter() - start
        if result is None:
            raise RuntimeError("calibration returned None")
        timings.append(elapsed / loops_used)
    return statistics.median(timings), loops_used


def _time(func: Callable[[], object], samples: int) -> tuple[float, int]:
    timings = []
    loops_used = 1
    for _ in range(samples):
        loops = 1
        while True:
            start = time.perf_counter()
            for _ in range(loops):
                result = func()
            elapsed = time.perf_counter() - start
            if result is None:
                raise RuntimeError("benchmark returned None")
            if elapsed >= 0.025 or loops >= 65_536:
                timings.append(elapsed / loops)
                loops_used = loops
                break
            loops *= 2
    return statistics.median(timings), loops_used


def run_benchmarks(samples: int) -> dict:
    state = State(w=[0.01, -0.02, 0.03], q=[1.0, 0.0, 0.0, 0.0], h=[0.1, 0.2, 0.3])
    estimated = EstimatorState(
        w=state.w,
        q=state.q,
        h=state.h,
        act_bias=np.zeros(3),
        sens_bias=np.zeros(6),
        dist_param=np.zeros(3),
    )
    states = [state.copy() for _ in range(128)]
    satellite = Satellite()
    orbital_state = Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=[7000.0, 0.0, 0.0],
        V=[0.0, 7.5, 0.0],
        fast=True,
    )
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    ukf = make_ukf(est_sat)
    augmented = ukf.x_hat.as_estimator_array()
    manifold_delta = np.zeros(ukf.x_hat.cov.shape[0])
    reset_delta = np.zeros(estimated.tangent_size)
    reset_delta[estimated.slice("attitude", coordinates="tangent")] = np.array([0.8, -0.55, 0.35])

    cases = {
        "construct": lambda: State(w=state.w, q=state.q, h=state.h),
        "field_access": lambda: (state.w, state.q, state.h),
        "full_size": lambda: estimated.full_size,
        "tangent_size": lambda: estimated.tangent_size,
        "full_slices": lambda: estimated.slices(coordinates="full"),
        "tangent_slices": lambda: estimated.slices(coordinates="tangent"),
        "copy": state.copy,
        "physical_conversion": state.as_array,
        "augmented_conversion": estimated.as_estimator_array,
        "tangent_map": estimated.tangent_map,
        "tangent_pinv": estimated.tangent_pinv,
        "retraction_jacobian_quaternion_vector": lambda: estimated.retraction_jacobian(
            reset_delta,
            quaternion_mode="quaternion_vector",
        ),
        "retraction_jacobian_rotation_vector": lambda: estimated.retraction_jacobian(
            reset_delta,
            quaternion_mode="rotation_vector",
        ),
        "retraction_jacobian_mrp_fallback": lambda: estimated.retraction_jacobian(
            reset_delta,
            quaternion_mode="mrp",
        ),
        "stack_128": lambda: State.stack(states),
        "rk4_propagation": lambda: satellite.noiseless_rk4(
            State(w=state.w, q=state.q),
            np.empty(0),
            0.1,
            orbital_state,
            orbital_state,
            mid_orbital_state=orbital_state,
        ),
        "estimator_manifold_update": lambda: ukf.add_to_state(augmented, manifold_delta),
    }
    calibration_seconds, calibration_loops = _measure_calibration()
    return {
        "schema_version": 1,
        "description": "Structured State and EstimatorState microbenchmarks.",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "calibration_seconds": calibration_seconds,
        "calibration_loops": calibration_loops,
        "benchmarks": {
            name: {
                "seconds": seconds,
                "loops": loops,
                "normalized_seconds": seconds / calibration_seconds,
                "max_regression": MAX_REGRESSION,
            }
            for name, (seconds, loops) in (
                (name, _time(func, samples)) for name, func in cases.items()
            )
        },
    }


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
    print("\nState benchmarks")
    print(f"calibration: {results['calibration_seconds']:.6f} s")
    print("-" * 88)
    print(f"{'benchmark':36} {'time':>12} {'normalized':>14} {'vs baseline':>14}")
    print("-" * 88)
    for name, entry in results["benchmarks"].items():
        ratio = ""
        if baseline is not None and name in baseline["benchmarks"]:
            ratio = _color_ratio(
                entry["normalized_seconds"] / baseline["benchmarks"][name]["normalized_seconds"],
                use_color,
            )
        print(
            f"{name:36} {entry['seconds'] * 1e6:10.1f} us "
            f"{entry['normalized_seconds']:14.6e} {ratio:>14}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ADCS State and EstimatorState paths.")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--samples", type=int, default=7)
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
