"""Attach a gyroscope to a Python ADCS subsystem and measure body rate.

Run from the repository root with
``venv/bin/python GenADCS/examples/plotting/sensors/gyroscope.py``.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import Gyroscope
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State
from GenADCS.backends.python.environment import Earth, Environment
from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.propagation import CoupledRK4Propagator
from GenADCS.backends.python.solver import CoupledSolver
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.shared.plotting import GyroscopePlot, plot


def main():
    geometry = SatelliteGeometry(
        mass=12.0,
        inertia=np.diag((0.13, 0.10, 0.05)),
        panels=[],
    )
    gyroscope = Gyroscope(
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        misalignment=np.array([1.0, 0.0, 0.0, 0.0]),
        saturation_value=10.0,
        noise=np.diag((1e-8, 1e-8, 1e-8)),
        bias=np.array([1e-4, -2e-4, 1.5e-4]),
    )
    adcs = ADCSSubsystem(sensors=[gyroscope])
    adcs.N = 1
    satellite = Satellite(geometry=geometry, adcs=adcs)
    state = State(
        attitude=AttitudeState((1.0, 0.0, 0.0, 0.0), (0.01, -0.015, 0.02)),
        orbit=OrbitalState.from_cartesian((6_878_137.0, 0.0, 0.0),
                                          (0.0, 7_612.6, 0.0), 2_461_000.5),
    )
    result = simulate(
        satellite, Environment(planetary_body=Earth()), DynamicsConfig(), state,
        CoupledSolver(CoupledRK4Propagator(0.25), tf=20.0, progress=True),
        resultsformat=ResultsFormat(gyroscope_clean=True, gyroscope_reading=True),
    )
    plot(result, GyroscopePlot(), show=False)
    import matplotlib.pyplot as plt
    plt.show()
    return result


if __name__ == "__main__":
    main()
