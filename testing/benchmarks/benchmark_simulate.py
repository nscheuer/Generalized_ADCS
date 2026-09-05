"""Fast simulate() performance checks.

This module intentionally does not depend on pytest-benchmark.  It can be run
directly in CI and compares each benchmark against a checked-in baseline after
normalizing by a small CPU/Numpy calibration workload:

    python testing/benchmarks/benchmark_simulate.py --compare

To refresh the baseline after an intentional performance change:

    python testing/benchmarks/benchmark_simulate.py --update-baseline
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import statistics
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ADCS.CONOPS.goals import No_Goal
from ADCS.controller import BDot, MTQ_w_RW
from ADCS.estimators.old_attitude_estimators import UAKF
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite, Satellite
from ADCS.satellite_hardware.sensors import Gyro, MTM
from ADCS.simulate import simulate
from ADCS.state import EstimatorState, State


BASELINE_PATH = REPO_ROOT / "testing" / "benchmarks" / "baselines" / "simulate.json"
RESULT_PATH = REPO_ROOT / "testing" / "artifacts" / "benchmarks" / "simulate_current.json"

DEFAULT_MIN_SECONDS = 0.030
DEFAULT_MAX_LOOPS = 16


@dataclass(frozen=True)
class Benchmark:
    name: str
    func: Callable[[], object]
    max_regression: float
    min_seconds: float = DEFAULT_MIN_SECONDS
    max_loops: int = DEFAULT_MAX_LOOPS


def _build_reference_setup() -> dict[str, object]:
    unit_vectors = MathConstants.unitvecs
    np.random.seed(11)

    actuators = [MTQ(axis=unit_vectors[index], max_torque=0.1) for index in range(3)]
    actuators += [RW(axis=unit_vectors[index], max_torque=4.51, J=0.22, h=0.0, h_max=3.8) for index in range(3)]

    sensors = [
        *[
            MTM(
                axis=unit_vectors[index],
                noise=Noise(noise=0.0, std_noise=1e-8),
                bias=Bias(bias=1e-9, std_bias=1e-9),
            )
            for index in range(3)
        ],
        *[
            Gyro(
                axis=unit_vectors[index],
                noise=Noise(noise=0.0, std_noise=1e-4),
                bias=Bias(bias=2e-3, std_bias=4e-4 * np.pi / 180.0),
            )
            for index in range(3)
        ],
    ]

    satellite = Satellite(
        mass=4.0,
        J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=actuators,
        sensors=sensors,
    )
    estimated_satellite = EstimatedSatellite.from_satellite(satellite)

    orbital_state = Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=-7000.0 * np.array([0.0, np.sqrt(0.5), np.sqrt(0.5)]),
        V=np.array([8.0, 0.0, 0.0]),
        B=np.array([0.0, 0.1, 0.0]),
        S=np.array([1.0e5 + 1.0, 0.0, 0.0]),
        rho=5.0e-12,
    )

    state_length = satellite.state_len
    initial_rate = random_n_unit_vec(3) * np.random.uniform(1.0, 2.0) * np.pi / 180.0
    initial_quaternion = random_n_unit_vec(4)
    x0 = State(w=initial_rate, q=initial_quaternion, h=np.zeros(state_length - 7))

    x_hat0 = EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], h=np.zeros(state_length - 7))
    reduced_length = state_length - 1
    covariance0 = np.diag(np.concatenate([[1e-3] * 3, [1e-2] * 3, [1e-4] * (reduced_length - 6)]))
    process_noise0 = np.eye(reduced_length) * 1e-8

    return {
        "satellite": satellite,
        "estimated_satellite": estimated_satellite,
        "orbital_state": orbital_state,
        "x0": x0,
        "x_hat0": x_hat0,
        "covariance0": covariance0,
        "process_noise0": process_noise0,
    }


def _run_quietly(func: Callable[[], object]) -> object:
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return func()


_REFERENCE = _build_reference_setup()


def _simulate_open_loop_one_step() -> object:
    return _run_quietly(
        lambda: simulate(
            x=_REFERENCE["x0"].copy(),
            satellite=_REFERENCE["satellite"],
            os0=_REFERENCE["orbital_state"],
            dt=1.0,
            tf=1.0,
        )
    )


def _simulate_bdot_one_step() -> object:
    controller = BDot(est_sat=_REFERENCE["estimated_satellite"], gain=100.0)
    return _run_quietly(
        lambda: simulate(
            x=_REFERENCE["x0"].copy(),
            satellite=_REFERENCE["satellite"],
            est_satellite=_REFERENCE["estimated_satellite"],
            controller=controller,
            goal=No_Goal(),
            os0=_REFERENCE["orbital_state"],
            dt=1.0,
            tf=1.0,
        )
    )


def _simulate_uakf_mtq_rw_two_steps() -> object:
    estimator = UAKF(
        est_sat=_REFERENCE["estimated_satellite"],
        J2000=_REFERENCE["orbital_state"].J2000,
        x_hat=_REFERENCE["x_hat0"].copy(),
        P_hat=np.array(_REFERENCE["covariance0"], copy=True),
        Q_hat=np.array(_REFERENCE["process_noise0"], copy=True),
        dt=1.0,
        cross_term=True,
        quat_as_vec=False,
    )
    controller = MTQ_w_RW(
        est_sat=_REFERENCE["estimated_satellite"],
        p_gain=0.0,
        d_gain=1.0,
        c_gain=0.0,
        h_target=np.zeros(3),
    )
    return _run_quietly(
        lambda: simulate(
            x=_REFERENCE["x0"].copy(),
            satellite=_REFERENCE["satellite"],
            est_satellite=_REFERENCE["estimated_satellite"],
            controller=controller,
            estimator=estimator,
            goal=No_Goal(),
            os0=_REFERENCE["orbital_state"],
            dt=1.0,
            tf=2.0,
        )
    )


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


def _make_benchmarks() -> list[Benchmark]:
    return [
        Benchmark(
            name="simulate_open_loop_one_step",
            func=_simulate_open_loop_one_step,
            max_regression=1.50,
        ),
        Benchmark(
            name="simulate_bdot_one_step",
            func=_simulate_bdot_one_step,
            max_regression=1.50,
        ),
        Benchmark(
            name="simulate_uakf_mtq_rw_two_steps",
            func=_simulate_uakf_mtq_rw_two_steps,
            max_regression=1.50,
            min_seconds=0.040,
            max_loops=4,
        ),
    ]


def run_benchmarks(samples: int) -> dict:
    benchmarks = _make_benchmarks()
    _warmup(benchmarks)

    calibration_seconds, calibration_loops = _median_time(
        _calibration_work,
        min_seconds=0.040,
        max_loops=64,
        samples=samples,
    )

    results = {
        "schema_version": 1,
        "description": "simulate() fast benchmark results normalized by calibration_seconds.",
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
    print("\nsimulate() benchmarks")
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
    parser = argparse.ArgumentParser(description="Benchmark ADCS.simulate.simulate.")
    parser.add_argument("--compare", action="store_true", help="fail if calibrated timings regress versus baseline")
    parser.add_argument("--update-baseline", action="store_true", help="replace the checked-in baseline with this run")
    parser.add_argument("--samples", type=int, default=5, help="median sample count per benchmark")
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
