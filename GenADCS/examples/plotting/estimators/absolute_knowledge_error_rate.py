"""Plot attitude and rate knowledge error in individual and magnitude modes."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Earth, Environment
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem, AttitudeMEKF
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import Gyroscope
from GenADCS.backends.python.propagation import CoupledRK4Propagator
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.python.solver import CoupledSolver
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State
from GenADCS.backends.shared.plotting import AbsoluteKnowledgeErrorRatePlot, plot


def main():
    identity = np.array([1., 0., 0., 0.])
    rate = np.array([.01, -.015, .02])
    gyro = Gyroscope(np.zeros(3), identity, identity, 10., np.eye(3)*1e-8, np.zeros(3))
    initial = AttitudeMEKF(identity, rate + np.array([1e-3, -1e-3, 2e-3]),
                           np.eye(6)*1e-3, 1e-8)
    satellite = Satellite(SatelliteGeometry(12., np.diag([.13, .1, .05]), []),
                          ADCSSubsystem(sensors=[gyro], estimator=initial))
    state = State(AttitudeState(identity, rate),
                  OrbitalState.from_cartesian((6878137., 0., 0.), (0., 7612.6, 0.), 2461000.5))
    result = simulate(satellite, Environment(planetary_body=Earth()), DynamicsConfig(), state,
        CoupledSolver(CoupledRK4Propagator(.25), tf=120., progress=True),
        resultsformat=ResultsFormat(satellite_attitude=True, satellite_rate=True,
                                    estimated_attitude=True, estimated_rate=True))
    plot(result, AbsoluteKnowledgeErrorRatePlot(option="individual"), AbsoluteKnowledgeErrorRatePlot(option="magnitude"), layout=(2, 1),
         title="Rate knowledge error", show=True)
    import matplotlib.pyplot as plt
    plt.show()
    return result

if __name__ == "__main__":
    main()
