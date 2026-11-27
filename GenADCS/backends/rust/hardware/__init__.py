"""Rust-backed hardware components."""

_CONTROLLER_NAMES = {
    "Controller",
    "NoController",
    "BDot",
    "BCross",
    "AttitudeController",
    "Feedforward",
    "Passthrough",
    "Feedback",
    "MRPFeedbackLinearized",
    "Allocator",
    "L2Allocator",
    "LInfinityAllocator",
    "Desaturation",
    "NoDesaturation",
    "HoganSchaubManagement",
}

__all__ = [
    "Satellite",
    "SatelliteGeometry",
    "SatellitePanel",
    "ADCSSubsystem",
    "Magnetorquer",
    "ReactionWheel",
    *_CONTROLLER_NAMES,
]


def __getattr__(name):
    if name == "Satellite":
        from .satellite import Satellite
        return Satellite
    if name in ("SatelliteGeometry", "SatellitePanel"):
        from . import satellite_geometry
        return getattr(satellite_geometry, name)
    if name in ("ADCSSubsystem", "Magnetorquer", "ReactionWheel") or name in _CONTROLLER_NAMES:
        from . import ADCS_subsystem
        return getattr(ADCS_subsystem, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
