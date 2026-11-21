"""Rust-backed hardware components."""

__all__ = ["Satellite", "SatelliteGeometry", "SatellitePanel", "ADCSSubsystem", "Magnetorquer", "ReactionWheel"]


def __getattr__(name):
    if name == "Satellite":
        from .satellite import Satellite
        return Satellite
    if name in ("SatelliteGeometry", "SatellitePanel"):
        from . import satellite_geometry
        return getattr(satellite_geometry, name)
    if name in ("ADCSSubsystem", "Magnetorquer", "ReactionWheel"):
        from . import ADCS_subsystem
        return getattr(ADCS_subsystem, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
