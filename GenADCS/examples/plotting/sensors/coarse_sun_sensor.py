"""Plot a coarse Sun sensor as the spacecraft rotates away from the Sun."""

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
CoarseSunSensor = _hardware_ADCS_subsystem_sensors.CoarseSunSensor
_state = import_module(f"GenADCS.backends.{backed}.state")
AttitudeState = _state.AttitudeState
OrbitalState = _state.OrbitalState
State = _state.State
_environment = import_module(f"GenADCS.backends.{backed}.environment")
Earth = _environment.Earth
Environment = _environment.Environment
DE421 = _environment.DE421
_config = import_module(f"GenADCS.backends.{backed}.config")
DynamicsConfig = _config.DynamicsConfig
_propagation = import_module(f"GenADCS.backends.{backed}.propagation")
CoupledRK4Propagator = _propagation.CoupledRK4Propagator
_solver = import_module(f"GenADCS.backends.{backed}.solver")
CoupledSolver = _solver.CoupledSolver
_simulate = import_module(f"GenADCS.backends.{backed}.simulate")
ResultsFormat = _simulate.ResultsFormat
simulate = _simulate.simulate
from GenADCS.backends.shared.plotting import CoarseSunSensorPlot, plot


def main():
    identity = np.array([1., 0., 0., 0.])
    sensor = CoarseSunSensor(np.zeros(3), identity, identity, np.array([0., 0., 1.]),
                             1.0, 1.0, 1e-8, 0.0, np.deg2rad(45.))
    satellite = Satellite(SatelliteGeometry(12., np.eye(3)*.1, panels=[]), ADCSSubsystem(sensors=[sensor]))
    _helpers_math_rotations_quaternion_conversions = import_module(f"GenADCS.backends.{backed}.helpers.math.rotations.quaternion.conversions")
    _axis_angle_to_quaternion = _helpers_math_rotations_quaternion_conversions._axis_angle_to_quaternion
    _helpers_math_time_conversions = import_module(f"GenADCS.backends.{backed}.helpers.math.time.conversions")
    utc_jd_to_tdb_jd = _helpers_math_time_conversions.utc_jd_to_tdb_jd
    sun_position = 1000. * DE421().position(0, utc_jd_to_tdb_jd(2461000.5))
    sun_direction = sun_position / np.linalg.norm(sun_position)
    position = 6_878_137. * sun_direction  # Day side of the orbit.
    tangent = np.cross(sun_direction, np.array([0., 0., 1.]))
    tangent /= np.linalg.norm(tangent)
    velocity = 7_612.6 * tangent
    # Map the initial Sun direction onto the sensor's +Z normal.
    axis = np.cross(sun_direction, np.array([0., 0., 1.]))
    angle = np.arccos(np.clip(sun_direction[2], -1., 1.))
    q0 = _axis_angle_to_quaternion(axis, angle)
    state = State(AttitudeState(q0, (0., np.deg2rad(18.), 0.)),
                  OrbitalState.from_cartesian(position, velocity, 2461000.5))
    env = Environment(planetary_body=Earth(), ephemeris=DE421())
    result = simulate(satellite, env, DynamicsConfig(), state,
                      CoupledSolver(CoupledRK4Propagator(.25), tf=20., progress=True),
                      resultsformat=ResultsFormat(coarse_sun_sensor_clean=True,
                                                  coarse_sun_sensor_reading=True))
    plot(result, CoarseSunSensorPlot(), show=False)
    import matplotlib.pyplot as plt
    plt.show()
    return result

if __name__ == '__main__':
    main()
