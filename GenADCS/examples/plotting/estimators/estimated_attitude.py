"""Simulate with AttitudeMEKF and plot estimated attitude."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Earth, Environment
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem, AttitudeMEKF
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import Gyroscope, AutonomousStarTracker
from GenADCS.backends.python.propagation import CoupledRK4Propagator
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.python.solver import CoupledSolver
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State
from GenADCS.backends.shared.plotting import EstimatedAttitudePlot, plot


def main():
    identity = np.array([1., 0., 0., 0.])
    gyro = Gyroscope(np.zeros(3), identity, identity, 10., np.eye(3)*1e-8, np.zeros(3))
    tracker = AutonomousStarTracker(np.zeros(3), identity, identity, 1.,
                                    np.eye(3) * 1e-6, np.zeros(3))
    rate = np.array([.01, -.015, .02])
    initial_attitude = np.array([np.cos(.1), np.sin(.1), 0., 0.])
    estimator = AttitudeMEKF(initial_attitude, rate + np.array([1e-3, -1e-3, 2e-3]),
                             np.eye(6)*1e-2, 1e-8)
    satellite = Satellite(SatelliteGeometry(12., np.diag([.13, .1, .05]), []),
                          ADCSSubsystem(sensors=[gyro, tracker], estimator=estimator))
    state = State(AttitudeState(identity, rate),
                  OrbitalState.from_cartesian((6878137., 0., 0.), (0., 7612.6, 0.), 2461000.5))
    result = simulate(satellite, Environment(planetary_body=Earth()), DynamicsConfig(), state,
        CoupledSolver(CoupledRK4Propagator(.25), tf=120., progress=True),
        resultsformat=ResultsFormat(satellite_attitude=True, estimated_attitude=True,
                                    estimated_rate=True))
    plot(result, EstimatedAttitudePlot(), show=True)
    import matplotlib.pyplot as plt
    plt.show()
    return result

if __name__ == "__main__":
    main()
