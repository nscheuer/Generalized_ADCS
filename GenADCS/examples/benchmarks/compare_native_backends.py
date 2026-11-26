"""Compare full Rust/C++ simulate calls using the benchmark suite workloads.

Run after release builds: venv/bin/python -m GenADCS.examples.benchmarks.compare_native_backends
Object construction and warmup are excluded; integration and result recording are timed.
"""
import argparse
import gc
from importlib import import_module
import json
from statistics import median
from time import perf_counter

import numpy as np


def workload(backend, case, steps):
    def module(name):
        return import_module(f"GenADCS.backends.{backend}.{name}")
    states, hardware, env = module("state"), module("hardware"), module("environment")
    config, propagation, solvers = module("config"), module("propagation"), module("solver")
    geometry = hardware.SatelliteGeometry(mass=12.0, inertia=np.diag((0.13, 0.10, 0.05)), panels=[])
    satellite = hardware.Satellite(geometry=geometry)
    environment = env.Environment(planetary_body=env.Earth(), atmosphere=env.NRLMSISE00(), ephemeris=env.DE421())
    forces = [module("dynamics").ZonalForceModel(geometry, environment, degrees=(2,))] if case.startswith("coupled") else []
    dynamics = config.DynamicsConfig(forces=forces, torques=[])
    state = states.State(
        attitude=states.AttitudeState(orientation=(1., 0., 0., 0.), rates=(.01, -.015, .02)),
        orbit=states.OrbitalState.from_cartesian(position=(6878137., 0., 0.), velocity=(0., 7612.6, 0.), julian_date=2461000.5),
    )
    dt = .25
    if case.startswith("coupled"):
        cls = propagation.CoupledRK4Propagator if case == "coupled_rk4" else propagation.CoupledDOP853Propagator
        solver = solvers.CoupledSolver(cls(dt=dt), tf=steps*dt, progress=False)
    else:
        attitude = propagation.RK4AttitudePropagator(dt=dt) if case == "separate_rk4" else propagation.ReferenceAttitudePropagator(dt=dt, mode="sun-pointing")
        solver = solvers.Solver(orbital=solvers.OrbitalSolver(propagation.KeplerianOrbitalPropagator(dt=dt)), attitude=solvers.AttitudeSolver(attitude), tf=steps*dt, progress=False)
    simulate = module("simulate").simulate
    def run():
        result = simulate(satellite, environment, dynamics, state, solver, backend={"rust": "rust", "cpp": "C"}[backend])
        times = result.satellites["sat_000"].times_s
        assert len(times) == steps+1 and times[-1] == steps*dt
        return result
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backends", nargs="+", default=["cpp", "rust"], choices=["cpp", "rust"])
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")
    timings = {}
    for case in ("coupled_rk4", "coupled_dop853", "separate_rk4", "separate_reference"):
        timings[case] = {}
        for backend in args.backends:
            workload(backend, case, 32)()
            run = workload(backend, case, args.steps)
            samples = []
            for _ in range(args.repeats):
                gc.collect()
                start = perf_counter()
                result = run()
                samples.append(perf_counter()-start)
                del result
            timings[case][backend] = {"median_seconds": median(samples), "samples_seconds": samples}
            print(json.dumps({"case": case, "backend": backend, **timings[case][backend]}), flush=True)
        if "rust" in timings[case] and "cpp" in timings[case]:
            print(json.dumps({"case": case, "rust_over_cpp": timings[case]["rust"]["median_seconds"] / timings[case]["cpp"]["median_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
