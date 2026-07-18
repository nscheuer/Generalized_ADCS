"""Microbenchmarks for the structured spacecraft state model."""

from __future__ import annotations

import argparse
import json
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
from ADCS.state import EstimatedState, State
from testing.test_estimators.ukf.helpers import make_baseline_sensors, make_satellites, make_ukf


def _time(func: Callable[[], object], samples: int) -> float:
    timings = []
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
                break
            loops *= 2
    return statistics.median(timings)


def run_benchmarks(samples: int) -> dict:
    state = State(w=[0.01, -0.02, 0.03], q=[1.0, 0.0, 0.0, 0.0], h=[0.1, 0.2, 0.3])
    estimated = EstimatedState(
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

    cases = {
        "construct": lambda: State(w=state.w, q=state.q, h=state.h),
        "field_access": lambda: (state.w, state.q, state.h),
        "copy": state.copy,
        "physical_conversion": state.as_array,
        "augmented_conversion": estimated.as_estimator_array,
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
    return {
        "schema_version": 1,
        "description": "Structured State and EstimatedState microbenchmarks.",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "benchmarks": {name: {"seconds": _time(func, samples)} for name, func in cases.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("--samples must be at least 3")
    result = run_benchmarks(args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
