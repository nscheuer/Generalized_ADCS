"""Benchmark 4,000,000 coupled dt intervals with rust DOP853.

Run from the repository root with ``venv/bin/python GenADCS/examples/benchmarks/rust_dop853.py``.
DOP853 may take additional adaptive substeps inside each dt interval.
"""

# Select the backend for this example: "rust", "cpp", or "python".
backed = "rust"

if backed not in ("rust", "cpp", "python"):
    raise ValueError("backed must be rust, cpp, or python")

from importlib import import_module

from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
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
_dynamics = import_module(f"GenADCS.backends.{backed}.dynamics")
ZonalForceModel = _dynamics.ZonalForceModel
_propagation = import_module(f"GenADCS.backends.{backed}.propagation")
CoupledDOP853Propagator = _propagation.CoupledDOP853Propagator
CoupledRK4Propagator = _propagation.CoupledRK4Propagator
_solver = import_module(f"GenADCS.backends.{backed}.solver")
CoupledSolver = _solver.CoupledSolver
_simulate = import_module(f"GenADCS.backends.{backed}.simulate")
ResultsFormat = _simulate.ResultsFormat
SimulationResult = _simulate.SimulationResult
simulate = _simulate.simulate


def main():
    steps = 4_000_000
    dt = 0.25
    geometry = SatelliteGeometry(mass=12.0, inertia=np.diag((0.13, 0.10, 0.05)), panels=[])
    satellite = Satellite(geometry=geometry)
    environment = Environment(
        planetary_body=Earth(), atmosphere=NRLMSISE00(), ephemeris=DE421()
    )
    dynamics = DynamicsConfig(
        forces=[ZonalForceModel(geometry, environment, degrees=(2,))], torques=[]
    )
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
    solver = CoupledSolver(CoupledDOP853Propagator(dt=dt), tf=steps * dt, progress=True)
    start = perf_counter()
    result = simulate(satellite, environment, dynamics, state, solver, backend={"python": "Python", "cpp": "C", "rust": "Rust"}[backed])
    elapsed = perf_counter() - start
    print(f"Backend: {backed}; integrator: DOP853; elapsed: {elapsed:.2f} s")
    print(f"Recorded samples: {len(result.satellites["sat_000"].times_s):,}; final time: {steps * dt:,.0f} s")
    return result


if __name__ == "__main__":
    main()
