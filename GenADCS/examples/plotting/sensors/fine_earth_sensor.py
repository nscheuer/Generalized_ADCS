"""Show fine Earth sensor alpha/beta angles during a spacecraft turn."""

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
FineEarthSensor = _hardware_ADCS_subsystem_sensors.FineEarthSensor
_state = import_module(f"GenADCS.backends.{backed}.state")
AttitudeState = _state.AttitudeState
OrbitalState = _state.OrbitalState
State = _state.State
_environment = import_module(f"GenADCS.backends.{backed}.environment")
Earth = _environment.Earth
Environment = _environment.Environment
_config = import_module(f"GenADCS.backends.{backed}.config")
DynamicsConfig = _config.DynamicsConfig
_propagation = import_module(f"GenADCS.backends.{backed}.propagation")
CoupledRK4Propagator = _propagation.CoupledRK4Propagator
_solver = import_module(f"GenADCS.backends.{backed}.solver")
CoupledSolver = _solver.CoupledSolver
_simulate = import_module(f"GenADCS.backends.{backed}.simulate")
ResultsFormat = _simulate.ResultsFormat
simulate = _simulate.simulate
from GenADCS.backends.shared.plotting import FineEarthSensorPlot, plot
_helpers_math_rotations_quaternion_conversions = import_module(f"GenADCS.backends.{backed}.helpers.math.rotations.quaternion.conversions")
_axis_angle_to_quaternion = _helpers_math_rotations_quaternion_conversions._axis_angle_to_quaternion

def main():
    identity = np.array([1., 0., 0., 0.])
    sensor = FineEarthSensor(np.zeros(3), identity, identity,
        np.diag((np.deg2rad(.05)**2, np.deg2rad(.05)**2)), np.zeros(2),
        np.deg2rad(np.array([50., 50.])))
    satellite = Satellite(SatelliteGeometry(12., np.eye(3)*.1, panels=[]),
                          ADCSSubsystem(sensors=[sensor]))
    position = np.array([6_878_137., 0., 0.])
    # Earth is along inertial -X. Map that direction onto sensor +Z.
    q0 = _axis_angle_to_quaternion(np.array([0., 1., 0.]), np.pi / 2)
    state = State(AttitudeState(q0, (0., np.deg2rad(5.), np.deg2rad(2.))),
        OrbitalState.from_cartesian(position, (0., 7_612.6, 0.), 2_461_000.5))
    result = simulate(satellite, Environment(planetary_body=Earth()), DynamicsConfig(), state,
        CoupledSolver(CoupledRK4Propagator(.1), tf=20., progress=True),
        resultsformat=ResultsFormat(fine_earth_sensor_clean=True, fine_earth_sensor_reading=True))
    plot(result, FineEarthSensorPlot(), show=False)
    import matplotlib.pyplot as plt
    plt.show()
    return result

if __name__ == '__main__':
    main()
