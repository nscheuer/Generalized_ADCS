"""Benchmark 4,000,000 coupled dt intervals with rust RK4.

Run from the repository root with ``venv/bin/python GenADCS/examples/benchmarks/rust_rk4.py``.
DOP853 may take additional adaptive substeps inside each dt interval.
"""

from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from GenADCS.backends.rust.state import AttitudeState, OrbitalState, State
from GenADCS.backends.rust.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.rust.environment import DE421, Earth, Environment, NRLMSISE00
from GenADCS.backends.rust.config import DynamicsConfig
from GenADCS.backends.rust.dynamics import ZonalForceModel
from GenADCS.backends.rust.propagation import CoupledDOP853Propagator, CoupledRK4Propagator
from GenADCS.backends.rust.solver import CoupledSolver
from GenADCS.backends.rust.simulate import ResultsFormat, SimulationResult, simulate


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
    solver = CoupledSolver(CoupledRK4Propagator(dt=dt), tf=steps * dt, progress=True)
    start = perf_counter()
    result = simulate(satellite, environment, dynamics, state, solver, backend="Rust")
    elapsed = perf_counter() - start
    print(f"Backend: rust; integrator: RK4; elapsed: {elapsed:.2f} s")
    print(f"Recorded samples: {len(result.times_s):,}; final time: {steps * dt:,.0f} s")
    return result


if __name__ == "__main__":
    main()
