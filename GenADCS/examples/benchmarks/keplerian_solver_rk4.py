"""Benchmark separate Keplerian orbit and RK4 attitude propagation.

Run from the repository root with
``venv/bin/python GenADCS/examples/benchmarks/keplerian_solver_rk4.py``.
The default run uses the same 4,000,000 intervals and 0.25 s timestep as
``coupled_solver_rk4.py``. The Keplerian orbit uses only central gravity;
the coupled benchmark's J2 force is therefore absent from this run.
"""

from importlib import import_module
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import DE421, NRLMSISE00, Earth, Environment
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.propagation import (
    KeplerianOrbitalPropagator,
    RK4AttitudePropagator,
)
from GenADCS.backends.python.simulate import simulate
from GenADCS.backends.python.solver import AttitudeSolver, OrbitalSolver, Solver
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State

# Select the backend for this example: "cpp" or "python".
backed = "cpp"

if backed not in ("cpp", "python"):
    raise ValueError("backed must be cpp or python")

_state = import_module(f"GenADCS.backends.{backed}.state")
AttitudeState = _state.AttitudeState
OrbitalState = _state.OrbitalState
State = _state.State
_hardware = import_module(f"GenADCS.backends.{backed}.hardware")
Satellite = _hardware.Satellite
SatelliteGeometry = _hardware.SatelliteGeometry
_environment = import_module(f"GenADCS.backends.{backed}.environment")
DE421 = _environment.DE421
Earth = _environment.Earth
Environment = _environment.Environment
NRLMSISE00 = _environment.NRLMSISE00
_config = import_module(f"GenADCS.backends.{backed}.config")
DynamicsConfig = _config.DynamicsConfig
_propagation = import_module(f"GenADCS.backends.{backed}.propagation")
KeplerianOrbitalPropagator = _propagation.KeplerianOrbitalPropagator
RK4AttitudePropagator = _propagation.RK4AttitudePropagator
_solver = import_module(f"GenADCS.backends.{backed}.solver")
AttitudeSolver = _solver.AttitudeSolver
OrbitalSolver = _solver.OrbitalSolver
Solver = _solver.Solver
_simulate = import_module(f"GenADCS.backends.{backed}.simulate")
simulate = _simulate.simulate


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
        backend={"python": "Python", "cpp": "C"}[backed],
    )
    elapsed = perf_counter() - start
    print(f"Backend: {backed}; integrator: Keplerian + RK4; elapsed: {elapsed:.2f} s")
    print(
        f"Recorded samples: {len(result.satellites['sat_000'].times_s):,}; "
        f"final time: {steps * dt:,.0f} s"
    )
    return result


if __name__ == "__main__":
    main()
