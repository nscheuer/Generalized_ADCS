"""Benchmark the selectable magnetic field models against each other.

Times every model in ADCS.environment.MAGNETIC_MODELS through the same
``field_gc`` entry point the package uses, on:

* a batch of samples sharing one date (one simulation step's worth of points),
* a batch with one date per sample (a whole trajectory, as Orbit computes it),
* a single scalar call (Orbital_State.get_b_eci, the simulation's inner loop),
* constructing a propagated Orbit end to end.

Each case is warmed up once first so numba's JIT compile is not counted.

    python debug/debug_orbit/benchmark_igrf.py [--n 5000] [--repeats 30]
"""

import argparse
import os
import statistics
import sys
import time
from datetime import datetime, timedelta

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from ADCS.environment import MAGNETIC_MODELS, field_gc
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants


def _time(fn, repeats):
    fn()  # warm-up: JIT compile, coefficient cache, imports
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times)


def _fmt(seconds):
    if seconds >= 1.0:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds * 1e6:.1f} us"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=5000, help="samples per batch")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--orbit-steps", type=int, default=2000, help="10 s steps in the Orbit case")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    n = args.n
    r = 6371.2 + rng.uniform(300.0, 800.0, n)
    th = rng.uniform(0.5, 179.5, n)
    ph = rng.uniform(-180.0, 180.0, n)
    d0 = datetime(2024, 3, 1, 12)
    dates = [d0 + timedelta(seconds=10.0 * i) for i in range(n)]

    ephem = Ephemeris()
    R0, V0 = np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.0])
    end = 0.22 + args.orbit_steps * 10.0 * TimeConstants.sec2cent

    rows = []
    for model in MAGNETIC_MODELS:
        try:
            field_gc(model, 6771.2, 45.0, 10.0, d0)
        except ImportError as exc:
            print(f"skipping {model}: {exc}")
            continue
        one = _time(lambda: field_gc(model, r, th, ph, d0), args.repeats)
        per = _time(lambda: field_gc(model, r, th, ph, dates), args.repeats)
        scal = _time(lambda: field_gc(model, 6771.2, 45.0, 10.0, d0), max(args.repeats, 50))
        os0 = Orbital_State(ephem, 0.22, R0, V0, magnetic_model=model)
        orb = _time(lambda: Orbit(os0, end_time=end, dt=10.0, verbose=False), 3)
        rows.append((model, one, per, scal, orb))

    head = (
        "Model",
        f"Batch {n}, one date (median / best)",
        f"Batch {n}, per-sample dates (median / best)",
        "Per sample",
        "Scalar call (median)",
        f"Orbit, {args.orbit_steps} steps (median)",
    )
    table = [head] + [
        (
            m,
            f"{_fmt(one[0])} / {_fmt(one[1])}",
            f"{_fmt(per[0])} / {_fmt(per[1])}",
            _fmt(per[0] / n),
            _fmt(scal[0]),
            _fmt(orb[0]),
        )
        for m, one, per, scal, orb in rows
    ]
    widths = [max(len(row[i]) for row in table) for i in range(len(head))]
    for k, row in enumerate(table):
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))
        if k == 0:
            print("-+-".join("-" * w for w in widths))

    if len(rows) == 2:
        (_, a1, a2, a3, a4), (_, b1, b2, b3, b4) = rows
        print(
            f"\n{rows[0][0]} speedup over {rows[1][0]}: "
            f"one date {b1[0] / a1[0]:.0f}x, per-sample dates {b2[0] / a2[0]:.0f}x, "
            f"scalar {b3[0] / a3[0]:.0f}x, Orbit {b4[0] / a4[0]:.1f}x"
        )


if __name__ == "__main__":
    main()
