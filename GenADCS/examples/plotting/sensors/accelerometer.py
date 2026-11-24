"""Plot accelerometer specific force while the spacecraft rotates in orbit."""

# Select the backend for this example: "rust", "cpp", or "python".
backed = "cpp"

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
Accelerometer = _hardware_ADCS_subsystem_sensors.Accelerometer
_state = import_module(f"GenADCS.backends.{backed}.state")
AttitudeState = _state.AttitudeState
OrbitalState = _state.OrbitalState
State = _state.State
_environment = import_module(f"GenADCS.backends.{backed}.environment")
Earth = _environment.Earth
Environment = _environment.Environment
_config = import_module(f"GenADCS.backends.{backed}.config")
DynamicsConfig = _config.DynamicsConfig
_dynamics = import_module(f"GenADCS.backends.{backed}.dynamics")
ZonalForceModel = _dynamics.ZonalForceModel
_propagation = import_module(f"GenADCS.backends.{backed}.propagation")
CoupledRK4Propagator = _propagation.CoupledRK4Propagator
_solver = import_module(f"GenADCS.backends.{backed}.solver")
CoupledSolver = _solver.CoupledSolver
_simulate = import_module(f"GenADCS.backends.{backed}.simulate")
ResultsFormat = _simulate.ResultsFormat
simulate = _simulate.simulate
from GenADCS.backends.shared.plotting import AccelerometerPlot, plot

def main():
    geometry = SatelliteGeometry(12., np.diag((.13, .10, .05)), panels=[])
    accelerometer = Accelerometer(np.zeros(3), np.array([1., 0., 0., 0.]),
        np.array([1., 0., 0., 0.]), 20., np.eye(3)*1e-6,
        np.array([.01, -.015, .02]))
    satellite = Satellite(geometry, ADCSSubsystem(sensors=[accelerometer]))
    env = Environment(planetary_body=Earth())
    dynamics = DynamicsConfig(forces=[ZonalForceModel(geometry, env, degrees=(2,))], torques=[])
    state = State(AttitudeState((1., 0., 0., 0.), (np.deg2rad(4.), np.deg2rad(3.), np.deg2rad(2.))),
        OrbitalState.from_cartesian((6_878_137., 0., 0.), (0., 7_612.6, 0.), 2_461_000.5))
    result = simulate(satellite, env, dynamics, state,
        CoupledSolver(CoupledRK4Propagator(.25), tf=60., progress=True),
        resultsformat=ResultsFormat(accelerometer_clean=True, accelerometer_reading=True))
    plot(result, AccelerometerPlot(), show=False)
    import matplotlib.pyplot as plt
    plt.show()
    return result

if __name__ == '__main__':
    main()
