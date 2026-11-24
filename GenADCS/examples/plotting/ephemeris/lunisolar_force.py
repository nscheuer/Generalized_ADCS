"""Run a 40,000-step DOP853 simulation and save the quaternion plot.

Run from the repository root:
    venv/bin/python GenADCS/examples/plotting/attitude/lunisolar_force.py
"""

# Select the backend for this example: "rust", "cpp", or "python".
backed = "python"

if backed not in ("rust", "cpp", "python"):
    raise ValueError("backed must be rust, cpp, or python")

from importlib import import_module

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

_config = import_module(f"GenADCS.backends.{backed}.config")
DynamicsConfig = _config.DynamicsConfig
_dynamics = import_module(f"GenADCS.backends.{backed}.dynamics")
ZonalForceModel = _dynamics.ZonalForceModel
ProjectedAreaSRPForceModel = _dynamics.ProjectedAreaSRPForceModel
ProjectedAreaSRPTorqueModel = _dynamics.ProjectedAreaSRPTorqueModel
LunisolarForceModel = _dynamics.LunisolarForceModel
_environment = import_module(f"GenADCS.backends.{backed}.environment")
Earth = _environment.Earth
Environment = _environment.Environment
DE421 = _environment.DE421
_hardware = import_module(f"GenADCS.backends.{backed}.hardware")
Satellite = _hardware.Satellite
SatelliteGeometry = _hardware.SatelliteGeometry
SatellitePanel = _hardware.SatellitePanel
_propagation = import_module(f"GenADCS.backends.{backed}.propagation")
CoupledDOP853Propagator = _propagation.CoupledDOP853Propagator
_simulate = import_module(f"GenADCS.backends.{backed}.simulate")
ResultsFormat = _simulate.ResultsFormat
simulate = _simulate.simulate
_solver = import_module(f"GenADCS.backends.{backed}.solver")
CoupledSolver = _solver.CoupledSolver
_state = import_module(f"GenADCS.backends.{backed}.state")
AttitudeState = _state.AttitudeState
OrbitalState = _state.OrbitalState
State = _state.State
from GenADCS.backends.shared.plotting import LunisolarForcePlot, plot


def main():
    dt = 0.25
    steps = 1_000
    geometry = SatelliteGeometry(mass=12.0, inertia=np.diag((0.13, 0.10, 0.05)), panels=[SatellitePanel(position=(0.0, 0.5, 0.0), orientation=(0.0, 0.0, 0.0), dimensions=(1.0, 1.0))])
    satellite = Satellite(geometry=geometry)
    environment = Environment(planetary_body=Earth(), ephemeris=DE421())
    dynamics = DynamicsConfig(forces=[ZonalForceModel(geometry, environment, degrees=(2,)), LunisolarForceModel(geometry, environment.ephemeris)], torques=[])
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
        resultsformat=ResultsFormat(satellite_attitude=False, satellite_rate=False, satellite_wheel_momenta=False, lunisolar_force=True),
    )
    fig, _ = plot(
        result,
        LunisolarForcePlot(),
        figure_kwargs={"figsize": (11, 5)}, show=True,
    )
    print(f"Displayed plot with {len(result.satellites['sat_000'].times_s)} samples")
    plt.show()
    return result


if __name__ == "__main__":
    main()
