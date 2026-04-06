import importlib
from typing import Any

__all__ = [
    "Satellite",
    "EstimatedSatellite",
    "simulate",
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
    "GPS",
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
]

_SYMBOL_TO_MODULE = {
    "Satellite": ".satellite_hardware.satellite",
    "EstimatedSatellite": ".satellite_hardware.satellite",
    "Actuator": ".satellite_hardware.actuators",
    "RW": ".satellite_hardware.actuators",
    "MTQ": ".satellite_hardware.actuators",
    "MTM": ".satellite_hardware.sensors",
    "Gyro": ".satellite_hardware.sensors",
    "SunSensor": ".satellite_hardware.sensors",
    "SunPair": ".satellite_hardware.sensors",
    "StarTracker": ".satellite_hardware.sensors",
    "GPS": ".satellite_hardware.sensors",
    "Bias": ".satellite_hardware.errors",
    "Noise": ".satellite_hardware.errors",
    "disturbances": ".satellite_hardware.disturbances",
    "controller": ".controller",
    "Attitude_Estimator": ".estimators.attitude_estimators",
    "UAKF": ".estimators.attitude_estimators",
    "SRUAKF": ".estimators.attitude_estimators",
    "Orbit_Estimator": ".estimators.orbit_estimators",
    "Orbit_EKF": ".estimators.orbit_estimators",
    "Orbit_GPS": ".estimators.orbit_estimators",
    "Orbital_State": ".orbits.orbital_state",
    "Ephemeris": ".orbits.ephemeris",
    "orbits": ".orbits",
    "goals": ".CONOPS",
    "GoalList": ".CONOPS.goallist",
    "simulate": ".simulate",
    "SimulationResults": ".helpers.simresults",
    "RunResults": ".helpers.simresults",
    "plot": ".helpers.plot",
    "plots": ".helpers.plot",
    "MCConfig": ".mc",
    "simulate_mc": ".mc",
}


def __getattr__(name: str) -> Any:
    if name not in _SYMBOL_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(_SYMBOL_TO_MODULE[name], __name__)
    if name == "plots":
        value = module
    elif name == "goals":
        value = module.goals
    elif name in {"disturbances", "controller", "orbits"}:
        value = module
    else:
        value = getattr(module, name)

    globals()[name] = value
    return value
