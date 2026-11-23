"""Run a 40,000-step DOP853 simulation and save the quaternion plot.

Run from the repository root:
    venv/bin/python GenADCS/examples/plotting/attitude/inertial_angular_rate.py
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.dynamics import ZonalForceModel
from GenADCS.backends.python.environment import Earth, Environment
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.propagation import CoupledDOP853Propagator
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.python.solver import CoupledSolver
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State
from GenADCS.backends.shared.plotting import InertialAngularRatePlot, plot


def main():
    dt = 0.25
    steps = 40_000
    geometry = SatelliteGeometry(mass=12.0, inertia=np.diag((0.13, 0.10, 0.05)))
    satellite = Satellite(geometry=geometry)
    environment = Environment(planetary_body=Earth())
    dynamics = DynamicsConfig(
        forces=[ZonalForceModel(geometry, environment, degrees=(2,))], torques=[]
    )
    state = State(
        attitude=AttitudeState(orientation=(1.0, 0.0, 0.0, 0.0),
                               rates=(0.01, -0.015, 0.02)),
        orbit=OrbitalState.from_cartesian(
            position=(6_878_137.0, 0.0, 0.0),
            velocity=(0.0, 7_612.6, 0.0), julian_date=2_461_000.5,
        ),
    )
    solver = CoupledSolver(CoupledDOP853Propagator(dt=dt), tf=steps * dt)
    result = simulate(
        satellite, environment, dynamics, state, solver,
        resultsformat=ResultsFormat(satellite_attitude=True, satellite_rate=True, satellite_wheel_momenta=False),
    )
    fig, _ = plot(
        result,
        InertialAngularRatePlot(linewidth=1),
        figure_kwargs={"figsize": (11, 5)}, show=True,
    )
    print(f"Displayed plot with {len(result.satellites['sat_000'].times_s)} samples")
    plt.show()
    return result


if __name__ == "__main__":
    main()
