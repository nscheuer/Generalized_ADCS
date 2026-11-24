"""Plot GNSS position and velocity measurements in two panels."""

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
GNSS = _hardware_ADCS_subsystem_sensors.GNSS
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
