import os as _os

# ADCS.pipeline's source lives with the paper it belongs to, at
# papers/Generalized_ACS/pipeline/, while still being imported as
# ADCS.pipeline. An installed wheel has it physically at ADCS/pipeline/ (see
# the package-dir mapping in pyproject.toml), but in a source checkout the
# directory is not under ADCS/, so extend this package's submodule search path
# to cover it. Without this, `import ADCS.pipeline` fails for anyone running
# from a clone -- which is how the test suite and every papers/ script run.
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir))
_pipeline_parent = _os.path.join(_repo_root, "papers", "Generalized_ACS")
if _os.path.isdir(_os.path.join(_pipeline_parent, "pipeline")):
    __path__.append(_pipeline_parent)
del _os, _repo_root, _pipeline_parent

from . import controller
from . import remote
from . import orbits
from . import satellite_factory
from .CONOPS import goals
from .CONOPS.goallist import GoalList
from .estimators.attitude_estimators import (
    AttitudeEstimator,
    Attitude_Estimator,
    EKF,
    MEKF,
    SRUAKF,
    UAKF,
)
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
from .covariance import Covariance
from .state import EstimatorState, State

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
    "AttitudeEstimator",
    "EKF",
    "MEKF",
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
    "State",
    "EstimatorState",
    "Covariance",
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
