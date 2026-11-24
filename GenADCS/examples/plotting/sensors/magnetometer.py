"""Simulate and plot clean and measured magnetometer readings.

Run from the repository root with
``venv/bin/python GenADCS/examples/plotting/sensors/magnetometer.py``.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import Magnetometer
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State
from GenADCS.backends.python.environment import Earth, Environment, IGRF14
from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.propagation import CoupledRK4Propagator
from GenADCS.backends.python.solver import CoupledSolver
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.shared.plotting import MagnetometerPlot, plot


def main():
    geometry = SatelliteGeometry(
        mass=12.0,
        inertia=np.diag((0.13, 0.10, 0.05)),
        panels=[],
    )
    magnetometer = Magnetometer(
        position=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        misalignment=np.array([1.0, 0.0, 0.0, 0.0]),
        saturation_value=1e-3,
        noise=np.diag((1e-12, 1e-12, 1e-12)),
        bias=np.array([1e-7, -2e-7, 1.5e-7]),
    )
    satellite = Satellite(geometry=geometry, adcs=ADCSSubsystem(sensors=[magnetometer]))
    satellite.adcs.N = 1
    state = State(
        attitude=AttitudeState((1.0, 0.0, 0.0, 0.0), (0.01, -0.015, 0.02)),
        orbit=OrbitalState.from_cartesian(
            (6_878_137.0, 0.0, 0.0), (0.0, 7_612.6, 0.0), 2_461_000.5,
        ),
    )
    environment = Environment(planetary_body=Earth(), magnetic_field=IGRF14())
    result = simulate(
        satellite, environment, DynamicsConfig(), state,
        CoupledSolver(CoupledRK4Propagator(0.25), tf=20.0, progress=True),
        resultsformat=ResultsFormat(magnetometer_clean=True, magnetometer_reading=True),
    )
    plot(result, MagnetometerPlot(), show=False)
    import matplotlib.pyplot as plt
    plt.show()
    return result


if __name__ == "__main__":
    main()
