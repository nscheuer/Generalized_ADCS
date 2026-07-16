"""Fast estimator performance checks.

This module intentionally does not depend on pytest-benchmark. It can be run
directly in CI and compares each benchmark against a checked-in baseline after
normalizing by a small CPU/Numpy calibration workload:

    python testing/benchmarks/benchmark_estimators.py --compare

To refresh the baseline after an intentional performance change:

    python testing/benchmarks/benchmark_estimators.py --update-baseline
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

from ADCS.estimators.orbit_estimators import Orbit_EKF, Orbit_GPS
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite, Satellite
from ADCS.satellite_hardware.sensors import GPS
from testing.test_estimators.srukf.helpers import make_srukf
from testing.test_estimators.ukf.helpers import make_baseline_sensors, make_estimate_guess, make_mtqs, make_orbital_state, make_rws, make_satellites, make_state, make_ukf


BASELINE_PATH = REPO_ROOT / "testing" / "benchmarks" / "baselines" / "estimators.json"
RESULT_PATH = REPO_ROOT / "testing" / "artifacts" / "benchmarks" / "estimators_current.json"

DEFAULT_MIN_SECONDS = 0.020
DEFAULT_MAX_LOOPS = 512


@dataclass(frozen=True)
class Benchmark:
    name: str
    func: Callable[[], object]
    max_regression: float
    min_seconds: float = DEFAULT_MIN_SECONDS
    max_loops: int = DEFAULT_MAX_LOOPS


def _make_orbit_estimator_setup() -> dict[str, object]:
    ephem = Ephemeris()
    true_state = Orbital_State(
        ephem=ephem,
        J2000=0.22,
        R=np.array([7000.0, 1200.0, -800.0]),
        V=np.array([1.1, 7.4, 2.0]),
    )
    gps = GPS(noise=Noise(noise=np.zeros(6), std_noise=np.full(6, 1.0e-3)))
    real_sat = Satellite(sensors=[gps])
    est_sat = EstimatedSatellite.from_satellite(real_sat)
    template = Orbital_State(ephem=ephem, J2000=0.22, R=true_state.R.copy(), V=true_state.V.copy())
    measurement_full = gps.clean_reading(x=None, os=true_state)
    measurement_pos = np.asarray(true_state.ECEF, dtype=float)
    os_guess = Orbital_State(
        ephem=ephem,
        J2000=0.22,
        R=true_state.R + np.array([5.0, -3.0, 2.0]),
        V=true_state.V + np.array([0.01, -0.02, 0.03]),
    )
    p_hat = np.diag([25.0, 25.0, 25.0, 1.0e-2, 1.0e-2, 1.0e-2])
    q_hat = np.diag([1.0e-2, 1.0e-2, 1.0e-2, 1.0e-5, 1.0e-5, 1.0e-5])
    return {
        "true_state": true_state,
        "gps": gps,
        "real_sat": real_sat,
        "est_sat": est_sat,
        "template": template,
        "measurement_full": measurement_full,
        "measurement_pos": measurement_pos,
        "os_guess": os_guess,
        "p_hat": p_hat,
        "q_hat": q_hat,
    }


def _reference_case() -> dict[str, object]:
    real_sat, est_sat = make_satellites(
        sensors=make_baseline_sensors(),
        estimated_sensors=make_baseline_sensors(),
    )
    os0 = make_orbital_state()
    x_true = make_state(
        w=np.array([2.0e-3, -1.0e-3, 1.2e-3]),
        q=np.array([0.98, 0.08, -0.05, 0.16]),
    )
    sensors = real_sat.noiseless_sensor_readings(x_true, os0)
    u = np.zeros(len(real_sat.actuators))

    real_sat_rw, est_sat_rw = make_satellites(
        sensors=make_baseline_sensors(),
        estimated_sensors=make_baseline_sensors(),
        actuators=make_mtqs() + make_rws(h=0.8),
        estimated_actuators=make_mtqs() + make_rws(h=0.0),
        disturbances=[],
        estimated_disturbances=[],
    )
    x_true_rw = make_state(
        w=np.array([1.0e-3, -1.5e-3, 0.8e-3]),
        q=np.array([0.99, 0.04, -0.03, 0.11]),
        h=np.array([0.8, 0.8, 0.8]),
    )
    sensors_rw = real_sat_rw.noiseless_sensor_readings(x_true_rw, os0)
    u_rw = np.zeros(len(real_sat_rw.actuators))

    orbit_setup = _make_orbit_estimator_setup()

    return {
        "real_sat": real_sat,
        "est_sat": est_sat,
        "os0": os0,
        "x_true": x_true,
        "sensors": sensors,
        "u": u,
        "real_sat_rw": real_sat_rw,
        "est_sat_rw": est_sat_rw,
        "x_true_rw": x_true_rw,
        "sensors_rw": sensors_rw,
        "u_rw": u_rw,
        "orbit": orbit_setup,
    }


_REFERENCE = _reference_case()


def _suppress_runtime_warnings(func: Callable[[], object]) -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func()


def _uakf_update_baseline() -> np.ndarray:
    ukf = make_ukf(_REFERENCE["est_sat"], dt=5.0, cross_term=False)
    return ukf.update(u=_REFERENCE["u"], sensors=np.array(_REFERENCE["sensors"], copy=True), os=_REFERENCE["os0"])


def _uakf_update_reaction_wheels() -> np.ndarray:
    x_hat = make_estimate_guess(_REFERENCE["est_sat_rw"], with_rw=True)
    x_hat[7:10] = np.zeros(3)
    ukf = make_ukf(_REFERENCE["est_sat_rw"], x_hat=x_hat, dt=5.0, cross_term=False)
    return ukf.update(u=_REFERENCE["u_rw"], sensors=np.array(_REFERENCE["sensors_rw"], copy=True), os=_REFERENCE["os0"])


def _srukf_update_baseline() -> np.ndarray:
    srukf = make_srukf(_REFERENCE["est_sat"], dt=5.0, cross_term=False)
    return srukf.update(u=_REFERENCE["u"], sensors=np.array(_REFERENCE["sensors"], copy=True), os=_REFERENCE["os0"])


def _srukf_update_reaction_wheels() -> np.ndarray:
    x_hat = make_estimate_guess(_REFERENCE["est_sat_rw"], with_rw=True)
    x_hat[7:10] = np.zeros(3)
    srukf = make_srukf(_REFERENCE["est_sat_rw"], x_hat=x_hat, dt=5.0, cross_term=False)
    return srukf.update(u=_REFERENCE["u_rw"], sensors=np.array(_REFERENCE["sensors_rw"], copy=True), os=_REFERENCE["os0"])


def _orbit_gps_update_full() -> object:
    setup = _REFERENCE["orbit"]
    return _suppress_runtime_warnings(
        lambda: Orbit_GPS(
            est_sat=setup["est_sat"],
            J2000=setup["true_state"].J2000,
            os_template=setup["template"],
        ).update(
            GPS_measurements=[setup["measurement_full"]],
            J2000=setup["true_state"].J2000,
        )
    )


def _orbit_gps_update_position_only() -> object:
    setup = _REFERENCE["orbit"]
    return _suppress_runtime_warnings(
        lambda: Orbit_GPS(
            est_sat=setup["est_sat"],
            J2000=setup["true_state"].J2000,
            os_template=setup["template"],
        ).update(
            GPS_measurements=[setup["measurement_pos"]],
            J2000=setup["true_state"].J2000,
        )
    )


def _orbit_ekf_predict_only() -> object:
    setup = _REFERENCE["orbit"]
    return _suppress_runtime_warnings(
        lambda: Orbit_EKF(
            est_sat=setup["est_sat"],
            J2000=setup["os_guess"].J2000,
            os_hat=setup["os_guess"].copy(),
            P_hat=np.array(setup["p_hat"], copy=True),
            Q_hat=np.array(setup["q_hat"], copy=True),
            dt=1.0,
        ).update(
            GPS_measurements=[],
            J2000=setup["true_state"].J2000 + TimeConstants.sec2cent,
        )
    )


def _orbit_ekf_update_full() -> object:
    setup = _REFERENCE["orbit"]
    return _suppress_runtime_warnings(
        lambda: Orbit_EKF(
            est_sat=setup["est_sat"],
            J2000=setup["os_guess"].J2000,
            os_hat=setup["os_guess"].copy(),
            P_hat=np.array(setup["p_hat"], copy=True),
            Q_hat=np.array(setup["q_hat"], copy=True),
            dt=1.0,
        ).update(
            GPS_measurements=[setup["measurement_full"]],
            J2000=setup["true_state"].J2000 + TimeConstants.sec2cent,
        )
    )


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
            name="uakf_update_baseline",
            func=_uakf_update_baseline,
            max_regression=1.50,
            min_seconds=0.025,
            max_loops=64,
        ),
        Benchmark(
            name="uakf_update_reaction_wheels",
            func=_uakf_update_reaction_wheels,
            max_regression=1.50,
            min_seconds=0.025,
            max_loops=64,
        ),
        Benchmark(
            name="srukf_update_baseline",
            func=_srukf_update_baseline,
            max_regression=1.50,
            min_seconds=0.025,
            max_loops=64,
        ),
        Benchmark(
            name="srukf_update_reaction_wheels",
            func=_srukf_update_reaction_wheels,
            max_regression=1.50,
            min_seconds=0.025,
            max_loops=64,
        ),
        Benchmark(
            name="orbit_gps_update_full",
            func=_orbit_gps_update_full,
            max_regression=1.50,
        ),
        Benchmark(
            name="orbit_gps_update_position_only",
            func=_orbit_gps_update_position_only,
            max_regression=1.50,
        ),
        Benchmark(
            name="orbit_ekf_predict_only",
            func=_orbit_ekf_predict_only,
            max_regression=1.50,
            min_seconds=0.025,
            max_loops=256,
        ),
        Benchmark(
            name="orbit_ekf_update_full",
            func=_orbit_ekf_update_full,
            max_regression=1.50,
            min_seconds=0.025,
            max_loops=256,
        ),
    ]


def run_benchmarks(samples: int) -> dict:
    benchmarks = _make_benchmarks()
    _warmup(benchmarks)
    calibration_seconds, calibration_loops = _median_time(_calibration_work, 0.040, 64, samples)
    results = {
        "schema_version": 1,
        "description": "Estimators fast benchmark results normalized by calibration_seconds.",
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
    print("\nEstimators benchmarks")
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
    parser = argparse.ArgumentParser(description="Benchmark ADCS estimator paths.")
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
