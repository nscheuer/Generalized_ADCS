"""Plot GNSS position and velocity measurements in two panels."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import GNSS
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State
from GenADCS.backends.python.environment import Earth, Environment
from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.propagation import CoupledRK4Propagator
from GenADCS.backends.python.solver import CoupledSolver
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.shared.plotting import GNSSPositionPlot, GNSSVelocityPlot, plot

def main():
    sigma = np.array([8., 8., 8., .02, .02, .02])
    gnss = GNSS(np.diag(sigma**2), np.array([2., -3., 1., .005, -.004, .003]))
    satellite = Satellite(SatelliteGeometry(12., np.diag((.13, .10, .05)), panels=[]),
                          ADCSSubsystem(sensors=[gnss]))
    state = State(AttitudeState((1., 0., 0., 0.), (0., 0., 0.)),
        OrbitalState.from_cartesian((6_878_137., 0., 0.), (0., 7_612.6, 0.), 2_461_000.5))
    result = simulate(satellite, Environment(planetary_body=Earth()), DynamicsConfig(), state,
        CoupledSolver(CoupledRK4Propagator(.25), tf=60., progress=True),
        resultsformat=ResultsFormat(gnss_clean=True, gnss_reading=True))
    plot(result, GNSSPositionPlot(), GNSSVelocityPlot(), layout=(2, 1),
         title="GNSS Position and Velocity", show=False)
    import matplotlib.pyplot as plt
    plt.show()
    return result

if __name__ == '__main__':
    main()
