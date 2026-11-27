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

_OBC_NAMES = {
    "OBCSubsystem",
    "OBCStateMachine",
    "Mode",
    "Transition",
    "Trigger",
    "Telecommand",
    "InFOV",
    "Completion",
    "After",
    "FDIRTriggered",
    "FDIR",
    "FDIRMonitor",
    "AngularRateMonitor",
}

__all__ = [
    "Satellite",
    "SatelliteGeometry",
    "SatellitePanel",
    "ADCSMode",
    "ADCSStateMachine",
    "ADCSSubsystem",
    "Magnetorquer",
    "ReactionWheel",
    *_CONTROLLER_NAMES,
    *_OBC_NAMES,
]


def __getattr__(name):
    if name == "Satellite":
        from .satellite import Satellite
        return Satellite
    if name in ("SatelliteGeometry", "SatellitePanel"):
        from . import satellite_geometry
        return getattr(satellite_geometry, name)
    if name in ("ADCSMode", "ADCSStateMachine", "ADCSSubsystem", "Magnetorquer", "ReactionWheel") or name in _CONTROLLER_NAMES:
        from . import ADCS_subsystem
        return getattr(ADCS_subsystem, name)
    if name in _OBC_NAMES:
        from . import OBC_subsystem
        if name == "OBCStateMachine":
            return getattr(OBC_subsystem, "StateMachine")
        return getattr(OBC_subsystem, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
