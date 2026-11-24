"""Plot star tracker quaternion readings as Earth enters its exclusion cone.

Run with ``venv/bin/python GenADCS/examples/plotting/sensors/star_tracker.py``.
"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import AutonomousStarTracker
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State
from GenADCS.backends.python.environment import Earth, Environment
from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.propagation import CoupledRK4Propagator
from GenADCS.backends.python.solver import CoupledSolver
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.shared.plotting import StarTrackerPlot, plot


def main():
    identity = np.array([1., 0., 0., 0.])
    tracker = AutonomousStarTracker(np.zeros(3), identity, identity, 1.,
                                    np.eye(3) * 1e-6, np.zeros(3),
                                    fov_half_angle=np.deg2rad(25.))
    geometry = SatelliteGeometry(12., np.eye(3) * .1, panels=[])
    satellite = Satellite(geometry, ADCSSubsystem(sensors=[tracker]))
    state = State(
        AttitudeState(identity, (0., -np.pi / 40., 0.)),
        OrbitalState.from_cartesian((6_878_137., 0., 0.), (0., 7_612.6, 0.), 2_461_000.5),
    )
    result = simulate(satellite, Environment(planetary_body=Earth()), DynamicsConfig(), state,
                      CoupledSolver(CoupledRK4Propagator(.25), tf=20., progress=True),
                      resultsformat=ResultsFormat(autonomous_star_tracker_clean=True,
                                                  autonomous_star_tracker_reading=True))
    plot(result, StarTrackerPlot(), show=False)
    import matplotlib.pyplot as plt
    plt.show()
    return result


if __name__ == '__main__':
    main()
