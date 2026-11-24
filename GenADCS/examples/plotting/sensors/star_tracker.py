"""Plot star tracker quaternion readings as Earth enters its exclusion cone.

Run with ``venv/bin/python GenADCS/examples/plotting/sensors/star_tracker.py``.
"""

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
AutonomousStarTracker = _hardware_ADCS_subsystem_sensors.AutonomousStarTracker
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
