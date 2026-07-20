"""Fast controller performance checks.

This module intentionally does not depend on pytest-benchmark. It can be run
directly in CI and compares each benchmark against a checked-in baseline after
normalizing by a small CPU/Numpy calibration workload:

    python testing/benchmarks/benchmark_controllers.py --compare

To refresh the baseline after an intentional performance change:

    python testing/benchmarks/benchmark_controllers.py --update-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ADCS.CONOPS.goals import ECI_Goal, No_Goal
from ADCS.controller import BDot, MTQ_Lovera, MTQ_Wisniewski, MTQ_w_RW, MTQ_w_RW_LP, MTQ_w_RW_QP, MTQ_w_RW_QPC, MTQ_w_RW_QPG, MTQ_w_RW_QPW
from ADCS.helpers.math_constants import MathConstants
from ADCS.orbits.universal_constants import TimeConstants
from testing.test_controllers._mtq_rw_qp_test_helpers import make_controller as make_qp_controller
from testing.test_controllers._mtq_rw_qp_test_helpers import make_orbital_state, make_satellite
from testing.test_controllers.test_controller_mtq_w_rw import StaticGoal as RWStaticGoal
from testing.test_controllers.test_controller_mtq_w_rw_lp import StaticGoal as LPStaticGoal


BASELINE_PATH = REPO_ROOT / "testing" / "benchmarks" / "baselines" / "controllers.json"
RESULT_PATH = REPO_ROOT / "testing" / "artifacts" / "benchmarks" / "controllers_current.json"

DEFAULT_MIN_SECONDS = 0.020
DEFAULT_MAX_LOOPS = 1_024


@dataclass(frozen=True)
class Benchmark:
    name: str
    func: Callable[[], object]
    max_regression: float
    min_seconds: float = DEFAULT_MIN_SECONDS
    max_loops: int = DEFAULT_MAX_LOOPS


def _reference_case() -> dict[str, object]:
    sat_mtq = make_satellite(include_rw=False, mtq_max_torque=0.4)
    sat_mixed = make_satellite(include_rw=True, mtq_max_torque=0.4, rw_max_torque=7.0e-3, rw_h=5.0e-3)

    os0 = make_orbital_state(b_body=np.array([2.0e-5, -3.0e-5, 4.0e-5]))
    os1 = make_orbital_state(b_body=np.array([2.2e-5, -2.7e-5, 3.6e-5]))
    os1.J2000 = os0.J2000 + TimeConstants.sec2cent

    q = np.array([0.985, 0.12, -0.08, 0.09], dtype=float)
    q = q / np.linalg.norm(q)

    x_mtq = np.concatenate([np.array([0.012, -0.017, 0.009]), q])
    x_mixed = np.concatenate([np.array([0.012, -0.017, 0.009]), q, np.array([5.0e-3])])

    sens_mtq0 = sat_mtq.sensor_readings(x_mtq, os0)
    sens_mtq1 = sat_mtq.sensor_readings(x_mtq, os1)
    sens_mixed = sat_mixed.sensor_readings(x_mixed, os0)

    bdot = BDot(est_sat=sat_mtq, gain=100.0)
    bdot.find_u(x_mtq, sens_mtq0, sat_mtq, os0, No_Goal())

    lovera = MTQ_Lovera(est_sat=sat_mixed, p_gain=20.0, d_gain=80.0, eps=1.0e-2)
    wisniewski = MTQ_Wisniewski(
        est_sat=sat_mixed,
        lambda_s=np.diag([8.0e-3, 8.0e-3, 8.0e-3]),
        lambda_q=np.diag([5.0e-3, 5.0e-3, 5.0e-3]),
    )
    mtq_rw = MTQ_w_RW(est_sat=sat_mixed, p_gain=0.1, d_gain=0.7, c_gain=0.1, h_target=np.zeros(3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mtq_rw_lp = MTQ_w_RW_LP(
            est_sat=sat_mixed,
            p_gain=5.0e-5,
            d_gain=1.0e-3,
            c_gain=0.0,
            h_target=np.array([0.004, 0.0, 0.0]),
        )

    qp_sat = make_satellite(include_rw=True, mtq_max_torque=0.4, rw_axes=[MathConstants.unitvecs[0]], rw_max_torque=7.0e-3, rw_h=5.0e-3)
    qp_x = np.concatenate([np.array([0.01, -0.015, 0.008]), q, np.array([5.0e-3])])
    qp_sens = qp_sat.sensor_readings(qp_x, os0)
    mtq_rw_qp = make_qp_controller(MTQ_w_RW_QP, qp_sat, 5.0e-5, 1.0e-3, 0.0, np.array([0.004, 0.0, 0.0]))
    mtq_rw_qpc = make_qp_controller(MTQ_w_RW_QPC, qp_sat, 5.0e-5, 1.0e-3, 0.0, np.array([0.004, 0.0, 0.0]))
    mtq_rw_qpg = make_qp_controller(MTQ_w_RW_QPG, qp_sat, 5.0e-5, 1.0e-3, 10.0, 0.0, np.array([0.004, 0.0, 0.0]))
    mtq_rw_qpw = make_qp_controller(MTQ_w_RW_QPW, qp_sat, 5.0e-5, 1.0e-3, 0.0, np.array([0.004, 0.0, 0.0]))

    pointing_goal = ECI_Goal(np.array([1.0, 0.0, 0.0]))
    static_goal = RWStaticGoal(q_err=np.array([0.08, -0.03, 0.02]), w_ref_eci=np.array([0.0, 0.0, 1.0e-3]))
    lp_pointing_goal = LPStaticGoal(q_err=np.array([0.05, -0.02, 0.01]))

    return {
        "sat_mtq": sat_mtq,
        "sat_mixed": sat_mixed,
        "qp_sat": qp_sat,
        "os0": os0,
        "os1": os1,
        "x_mtq": x_mtq,
        "x_mixed": x_mixed,
        "qp_x": qp_x,
        "sens_mtq1": sens_mtq1,
        "sens_mixed": sens_mixed,
        "qp_sens": qp_sens,
        "bdot": bdot,
        "lovera": lovera,
        "wisniewski": wisniewski,
        "mtq_rw": mtq_rw,
        "mtq_rw_lp": mtq_rw_lp,
        "mtq_rw_qp": mtq_rw_qp,
        "mtq_rw_qpc": mtq_rw_qpc,
        "mtq_rw_qpg": mtq_rw_qpg,
        "mtq_rw_qpw": mtq_rw_qpw,
        "pointing_goal": pointing_goal,
        "static_goal": static_goal,
        "lp_pointing_goal": lp_pointing_goal,
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
    return [
        Benchmark(
            name="bdot_find_u_steady_state",
            func=lambda: _REFERENCE["bdot"].find_u(
                _REFERENCE["x_mtq"],
                _REFERENCE["sens_mtq1"],
                _REFERENCE["sat_mtq"],
                _REFERENCE["os1"],
                No_Goal(),
            ),
            max_regression=1.50,
        ),
        Benchmark(
            name="mtq_lovera_find_u",
            func=lambda: _REFERENCE["lovera"].find_u(
                _REFERENCE["x_mixed"],
                _REFERENCE["sens_mixed"],
                _REFERENCE["sat_mixed"],
                _REFERENCE["os0"],
                _REFERENCE["pointing_goal"],
            ),
            max_regression=1.50,
        ),
        Benchmark(
            name="mtq_wisniewski_find_u",
            func=lambda: _REFERENCE["wisniewski"].find_u(
                _REFERENCE["x_mixed"],
                _REFERENCE["sens_mixed"],
                _REFERENCE["sat_mixed"],
                _REFERENCE["os0"],
                _REFERENCE["pointing_goal"],
            ),
            max_regression=1.50,
        ),
        Benchmark(
            name="mtq_w_rw_find_u",
            func=lambda: _REFERENCE["mtq_rw"].find_u(
                _REFERENCE["x_mixed"],
                _REFERENCE["sens_mixed"],
                _REFERENCE["sat_mixed"],
                _REFERENCE["os0"],
                _REFERENCE["static_goal"],
            ),
            max_regression=1.50,
        ),
        Benchmark(
            name="mtq_w_rw_lp_pointing_find_u",
            func=lambda: _REFERENCE["mtq_rw_lp"].find_u(
                _REFERENCE["x_mixed"],
                _REFERENCE["sens_mixed"],
                _REFERENCE["sat_mixed"],
                _REFERENCE["os0"],
                _REFERENCE["lp_pointing_goal"],
            ),
            max_regression=1.50,
        ),
        Benchmark(
            name="mtq_w_rw_lp_desat_find_u",
            func=lambda: _REFERENCE["mtq_rw_lp"].find_u(
                _REFERENCE["x_mixed"],
                _REFERENCE["sens_mixed"],
                _REFERENCE["sat_mixed"],
                _REFERENCE["os0"],
                No_Goal(),
            ),
            max_regression=1.50,
        ),
        Benchmark(
            name="mtq_w_rw_qp_find_u",
            func=lambda: _REFERENCE["mtq_rw_qp"].find_u(
                _REFERENCE["qp_x"],
                _REFERENCE["qp_sens"],
                _REFERENCE["qp_sat"],
                _REFERENCE["os0"],
                _REFERENCE["lp_pointing_goal"],
            ),
            max_regression=1.50,
            min_seconds=0.040,
            max_loops=512,
        ),
        Benchmark(
            name="mtq_w_rw_qpc_find_u",
            func=lambda: _REFERENCE["mtq_rw_qpc"].find_u(
                _REFERENCE["qp_x"],
                _REFERENCE["qp_sens"],
                _REFERENCE["qp_sat"],
                _REFERENCE["os0"],
                _REFERENCE["lp_pointing_goal"],
            ),
            max_regression=1.50,
            min_seconds=0.040,
            max_loops=512,
        ),
        Benchmark(
            name="mtq_w_rw_qpg_find_u",
            func=lambda: _REFERENCE["mtq_rw_qpg"].find_u(
                _REFERENCE["qp_x"],
                _REFERENCE["qp_sens"],
                _REFERENCE["qp_sat"],
                _REFERENCE["os0"],
                _REFERENCE["lp_pointing_goal"],
            ),
            max_regression=1.50,
            min_seconds=0.040,
            max_loops=512,
        ),
        Benchmark(
            name="mtq_w_rw_qpw_find_u",
            func=lambda: _REFERENCE["mtq_rw_qpw"].find_u(
                _REFERENCE["qp_x"],
                _REFERENCE["qp_sens"],
                _REFERENCE["qp_sat"],
                _REFERENCE["os0"],
                _REFERENCE["lp_pointing_goal"],
            ),
            max_regression=1.50,
            min_seconds=0.040,
            max_loops=512,
        ),
    ]


def run_benchmarks(samples: int) -> dict:
    benchmarks = _make_benchmarks()
    _warmup(benchmarks)
    calibration_seconds, calibration_loops = _median_time(_calibration_work, 0.040, 64, samples)
    results = {
        "schema_version": 1,
        "description": "Controllers fast benchmark results normalized by calibration_seconds.",
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
    print("\nControllers benchmarks")
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
    parser = argparse.ArgumentParser(description="Benchmark ADCS controller paths.")
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
