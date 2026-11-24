"""Simulate and plot clean and measured magnetometer readings.

Run from the repository root with
``venv/bin/python GenADCS/examples/plotting/sensors/magnetometer.py``.
"""

# Select the backend for this example: "rust", "cpp", or "python".
backed = "python"

if backed not in ("rust", "cpp", "python"):
    raise ValueError("backed must be rust, cpp, or python")

from importlib import import_module

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

_hardware = import_module(f"GenADCS.backends.{backed}.hardware")
Satellite = _hardware.Satellite
SatelliteGeometry = _hardware.SatelliteGeometry
_hardware_ADCS_subsystem = import_module(f"GenADCS.backends.{backed}.hardware.ADCS_subsystem")
ADCSSubsystem = _hardware_ADCS_subsystem.ADCSSubsystem
_hardware_ADCS_subsystem_sensors = import_module(f"GenADCS.backends.{backed}.hardware.ADCS_subsystem.sensors")
Magnetometer = _hardware_ADCS_subsystem_sensors.Magnetometer
_state = import_module(f"GenADCS.backends.{backed}.state")
AttitudeState = _state.AttitudeState
OrbitalState = _state.OrbitalState
State = _state.State
_environment = import_module(f"GenADCS.backends.{backed}.environment")
Earth = _environment.Earth
Environment = _environment.Environment
IGRF14 = import_module(f"GenADCS.backends.{backed}.environment.magnetic_field_model").IGRF14
_config = import_module(f"GenADCS.backends.{backed}.config")
DynamicsConfig = _config.DynamicsConfig
_propagation = import_module(f"GenADCS.backends.{backed}.propagation")
CoupledRK4Propagator = _propagation.CoupledRK4Propagator
_solver = import_module(f"GenADCS.backends.{backed}.solver")
CoupledSolver = _solver.CoupledSolver
_simulate = import_module(f"GenADCS.backends.{backed}.simulate")
ResultsFormat = _simulate.ResultsFormat
simulate = _simulate.simulate
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
