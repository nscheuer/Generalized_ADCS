"""Benchmark SGP4Orbit over a ten-revolution LEO mission.

Runs a direct script-level benchmark without pytest-benchmark:

    .venv/bin/python debug/debug_orbit/benchmark_sgp4_orbit.py
    .venv/bin/python debug/debug_orbit/benchmark_sgp4_orbit.py --dt 120 --repeats 5
"""

import argparse
from pathlib import Path
import statistics
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.sgp4_orbit import SGP4Orbit
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants
from ADCS.satellite_hardware.disturbances import Drag_Disturbance
from ADCS.satellite_hardware.disturbances.helpers.geometry_config import GeometryConfig, GeometryFace
from ADCS.satellite_hardware.satellite.satellite import Satellite


def _median_seconds(fn, repeats):
    samples = []
    last_result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
        if result is None:
            raise RuntimeError("benchmark operation returned None")
        last_result = result
    return statistics.median(samples), last_result


def _benchmark_satellite():
    faces = [
        GeometryFace(area=0.1, centroid=np.zeros(3), normal=normal, CD=2.2)
        for normal in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])
    ]
    drag = Drag_Disturbance(GeometryConfig(faces))
    return Satellite(mass=5.0, disturbances=[drag])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--altitude-km", type=float, default=400.0)
    parser.add_argument("--dt", type=float, default=300.0, help="state-query spacing in seconds")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    radius = EarthConstants.R_e + args.altitude_km
    speed = np.sqrt(EarthConstants.mu_e / radius)
    os0 = Orbital_State.from_position(J2000=0.22, R=[radius, 0.0, 0.0], V=[0.0, speed, 0.0])
    period = 2.0 * np.pi * np.sqrt(radius**3 / EarthConstants.mu_e)
    duration = 10.0 * period
    end = os0.J2000 + duration * TimeConstants.sec2cent
    satellite = _benchmark_satellite()

    construction, orbit = _median_seconds(
        lambda: SGP4Orbit(os0, end_time=end, dt=args.dt, satellite=satellite, verbose=True), args.repeats
    )
    query_times = orbit.times

    def query_mission():
        return [orbit.get_os(float(t)) for t in query_times]

    query_time, _ = _median_seconds(query_mission, args.repeats)
    print("SGP4Orbit ten-orbit LEO benchmark")
    print(f"Altitude: {args.altitude_km:.1f} km; period: {period:.1f} s; mission: {duration / 3600:.2f} h")
    print(f"Queries: {len(query_times)} at {args.dt:g} s spacing; geometry-derived B*: {orbit.bstar:.6g} Earth radii⁻¹")
    print(f"Full mission precomputation: {construction:.6f} s median")
    print(f"Cached mission state lookups: {query_time:.3f} s median ({query_time / len(query_times) * 1e3:.3f} ms/state)")
    print(f"Query throughput: {len(query_times) / query_time:.1f} states/s")


if __name__ == "__main__":
    main()
