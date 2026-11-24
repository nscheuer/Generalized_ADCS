"""Show fine Earth sensor alpha/beta angles during a spacecraft turn."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import FineEarthSensor
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State
from GenADCS.backends.python.environment import Earth, Environment
from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.propagation import CoupledRK4Propagator
from GenADCS.backends.python.solver import CoupledSolver
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.shared.plotting import FineEarthSensorPlot, plot
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import _axis_angle_to_quaternion

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
