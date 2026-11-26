"""Benchmark separate Keplerian orbit and RK4 attitude propagation.

Run from the repository root with
``venv/bin/python GenADCS/examples/benchmarks/keplerian_solver_rk4.py``.
The default run uses 4,000,000 intervals and the same 0.25 s timestep as
``coupled_solver_rk4.py``. The Keplerian orbit uses only central gravity;
the coupled benchmark's J2 force is therefore absent from this run.
"""

from importlib import import_module
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Select the backend for this example: "rust", "cpp", or "python".
backend = "cpp"

if backend not in ("rust", "cpp", "python"):
    raise ValueError("backend must be rust, cpp, or python")

_state = import_module(f"GenADCS.backends.{backend}.state")
AttitudeState = _state.AttitudeState
OrbitalState = _state.OrbitalState
State = _state.State
_hardware = import_module(f"GenADCS.backends.{backend}.hardware")
Satellite = _hardware.Satellite
SatelliteGeometry = _hardware.SatelliteGeometry
_environment = import_module(f"GenADCS.backends.{backend}.environment")
DE421 = _environment.DE421
Earth = _environment.Earth
Environment = _environment.Environment
NRLMSISE00 = _environment.NRLMSISE00
_config = import_module(f"GenADCS.backends.{backend}.config")
DynamicsConfig = _config.DynamicsConfig
_propagation = import_module(f"GenADCS.backends.{backend}.propagation")
KeplerianOrbitalPropagator = _propagation.KeplerianOrbitalPropagator
RK4AttitudePropagator = _propagation.RK4AttitudePropagator
_solver = import_module(f"GenADCS.backends.{backend}.solver")
AttitudeSolver = _solver.AttitudeSolver
OrbitalSolver = _solver.OrbitalSolver
Solver = _solver.Solver
_simulate = import_module(f"GenADCS.backends.{backend}.simulate")
simulate = _simulate.simulate
ResultsFormat = _simulate.ResultsFormat


def main(steps: int = 4_000_000):
    dt = 0.25
    geometry = SatelliteGeometry(
        mass=12.0, inertia=np.diag((0.13, 0.10, 0.05)), panels=[]
    )
    satellite = Satellite(geometry=geometry)
    environment = Environment(
        planetary_body=Earth(), atmosphere=NRLMSISE00(), ephemeris=DE421()
    )
    dynamics = DynamicsConfig(forces=[], torques=[])
    state = State(
        attitude=AttitudeState(
            orientation=(1.0, 0.0, 0.0, 0.0), rates=(0.01, -0.015, 0.02)
        ),
        orbit=OrbitalState.from_cartesian(
            position=(6_878_137.0, 0.0, 0.0),
            velocity=(0.0, 7_612.6, 0.0),
            julian_date=2_461_000.5,
        ),
    )
    solver = Solver(
        orbital=OrbitalSolver(KeplerianOrbitalPropagator(dt=dt)),
        attitude=AttitudeSolver(RK4AttitudePropagator(dt=dt)),
        tf=steps * dt,
        progress=True,
    )
    start = perf_counter()
    result = simulate(
        satellite, environment, dynamics, state, solver,
        backend={"python": "Python", "cpp": "C", "rust": "rust"}[backend],
        resultsformat=ResultsFormat(N=500),
    )
    elapsed = perf_counter() - start
    print(f"Backend: {backend}; integrator: Keplerian + RK4; elapsed: {elapsed:.2f} s")
    print(
        f"Recorded samples: {len(result.satellites['sat_000'].times_s):,}; "
        f"final time: {steps * dt:,.0f} s"
    )
    return result


if __name__ == "__main__":
    main()
