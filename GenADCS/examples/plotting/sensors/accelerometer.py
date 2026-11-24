"""Plot accelerometer specific force while the spacecraft rotates in orbit."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import Accelerometer
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State
from GenADCS.backends.python.environment import Earth, Environment
from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.dynamics import ZonalForceModel
from GenADCS.backends.python.propagation import CoupledRK4Propagator
from GenADCS.backends.python.solver import CoupledSolver
from GenADCS.backends.python.simulate import ResultsFormat, simulate
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
        CoupledSolver(CoupledRK4Propagator(.25), tf=60.),
        resultsformat=ResultsFormat(accelerometer_clean=True, accelerometer_reading=True))
    plot(result, AccelerometerPlot(), show=False)
    import matplotlib.pyplot as plt
    plt.show()
    return result

if __name__ == '__main__':
    main()
