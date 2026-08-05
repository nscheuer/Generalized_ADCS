from . import controller
from . import remote
from . import orbits
from . import satellite_factory
from .CONOPS import goals
from .CONOPS.goallist import GoalList
from .estimators.attitude_estimators import Attitude_Estimator, SRUAKF, UAKF
from .estimators.orbit_estimators import Orbit_EKF, Orbit_Estimator, Orbit_GPS
from .helpers import plot as plots
from .helpers.plot import plot
from .helpers.simresults import RunResults, SimulationResults
from .mc import MCConfig, simulate_mc
from .orbits.ephemeris import Ephemeris
from .orbits.orbital_state import Orbital_State
from .satellite_hardware import disturbances
from .satellite_hardware.actuators import Actuator, MTQ, RW
from .satellite_hardware.errors import Bias, Noise
from .satellite_hardware.satellite import EstimatedSatellite, Satellite
from .satellite_hardware.sensors import GPS, Gyro, MTM, StarTracker, SunPair, SunSensor, StarTrackerQuaternion, EarthHorizonSensor
from .simulate import simulate
from .simulate_remote import simulate_remote

__all__ = [
    "Satellite",
    "EstimatedSatellite",
    "simulate",
    "simulate_remote",
    "SimulationResults",
    "RunResults",
    "disturbances",
    "Actuator",
    "RW",
    "MTQ",
    "MTM",
    "Gyro",
    "SunSensor",
    "SunPair",
    "StarTracker",
    "StarTrackerQuaternion",
    "GPS",
    "EarthHorizonSensor",
    "Bias",
    "Noise",
    "controller",
    "Attitude_Estimator",
    "UAKF",
    "SRUAKF",
    "Orbit_Estimator",
    "Orbit_EKF",
    "Orbit_GPS",
    "goals",
    "GoalList",
    "plot",
    "plots",
    "MCConfig",
    "simulate_mc",
    "Orbital_State",
    "Ephemeris",
    "orbits",
    "satellite_factory",
]

# Package version, read from the installed distribution metadata so it cannot
# drift from pyproject.toml. The fallback covers a source checkout that was
# never installed (e.g. run straight from a git clone via PYTHONPATH).
from importlib.metadata import PackageNotFoundError as _PkgNotFound, version as _version

try:
    __version__ = _version("Generalized_ADCS")
except _PkgNotFound:  # pragma: no cover - source tree, not installed
    __version__ = "0.0.0.dev0"

del _version, _PkgNotFound
