"""Show fine Sun sensor alpha/beta angles during a spacecraft turn."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import FineSunSensor
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State
from GenADCS.backends.python.environment import Earth, Environment, DE421
from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.propagation import CoupledRK4Propagator
from GenADCS.backends.python.solver import CoupledSolver
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.shared.plotting import FineSunSensorPlot, plot
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import _axis_angle_to_quaternion
from GenADCS.backends.python.helpers.math.time.conversions import utc_jd_to_tdb_jd

def main():
    identity = np.array([1., 0., 0., 0.])
    sensor = FineSunSensor(np.zeros(3), identity, identity,
        np.diag((np.deg2rad(.05)**2, np.deg2rad(.05)**2)), np.zeros(2),
        np.deg2rad(np.array([50., 50.])))
    satellite = Satellite(SatelliteGeometry(12., np.eye(3)*.1, panels=[]), ADCSSubsystem(sensors=[sensor]))
    epoch = 2461000.5
    sun = 1000. * DE421().position(0, utc_jd_to_tdb_jd(epoch))
    direction = sun / np.linalg.norm(sun)
    position = 6_878_137. * direction
    tangent = np.cross(direction, [0., 0., 1.]); tangent /= np.linalg.norm(tangent)
    qaxis = np.cross(direction, [0., 0., 1.])
    q0 = _axis_angle_to_quaternion(qaxis, np.arccos(np.clip(direction[2], -1., 1.)))
    state = State(AttitudeState(q0, (np.deg2rad(8.), np.deg2rad(4.), 0.)),
        OrbitalState.from_cartesian(position, 7612.6*tangent, epoch))
    env = Environment(planetary_body=Earth(), ephemeris=DE421())
    result = simulate(satellite, env, DynamicsConfig(), state,
        CoupledSolver(CoupledRK4Propagator(.1), tf=20., progress=True),
        resultsformat=ResultsFormat(fine_sun_sensor_clean=True, fine_sun_sensor_reading=True))
    plot(result, FineSunSensorPlot(), show=False)
    import matplotlib.pyplot as plt
    plt.show()
    return result

if __name__ == '__main__':
    main()
