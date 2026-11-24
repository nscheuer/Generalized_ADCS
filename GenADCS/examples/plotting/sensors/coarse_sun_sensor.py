"""Plot a coarse Sun sensor as the spacecraft rotates away from the Sun."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import CoarseSunSensor
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State
from GenADCS.backends.python.environment import Earth, Environment, DE421
from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.propagation import CoupledRK4Propagator
from GenADCS.backends.python.solver import CoupledSolver
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.shared.plotting import CoarseSunSensorPlot, plot


def main():
    identity = np.array([1., 0., 0., 0.])
    sensor = CoarseSunSensor(np.zeros(3), identity, identity, np.array([0., 0., 1.]),
                             1.0, 1.0, 1e-8, 0.0, np.deg2rad(45.))
    satellite = Satellite(SatelliteGeometry(12., np.eye(3)*.1, panels=[]), ADCSSubsystem(sensors=[sensor]))
    from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import _axis_angle_to_quaternion
    from GenADCS.backends.python.helpers.math.time.conversions import utc_jd_to_tdb_jd
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
