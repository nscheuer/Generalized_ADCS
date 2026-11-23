"""Save selected coupled results to HDF5 and load them back.

Run from the repository root with ``venv/bin/python GenADCS/examples/input_output/rust_simulationresult.py``.
"""

from pathlib import Path
import sys

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
    dt = 0.25
    steps = 100
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
    resultsformat = ResultsFormat(
        satellite_attitude=True, satellite_rate=False, satellite_wheel_momenta=False, N=4
    )
    result = simulate(
        satellite, environment, dynamics, state, solver,
        backend="Rust", resultsformat=resultsformat,
    )
    output = Path(__file__).with_suffix(".h5")
    result.save(output)
    loaded = SimulationResult.load(output)
    sat_000 = loaded.satellites["sat_000"]
    print(f"Saved {len(sat_000.times_s)} quaternion samples to {output}")
    print(f"Omitted rate: {sat_000.satellite_rate is None}; interval N: {loaded.N}")
    return loaded


if __name__ == "__main__":
    main()
