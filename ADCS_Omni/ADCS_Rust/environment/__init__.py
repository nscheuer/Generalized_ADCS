"""Rust-backed environment models, loaded from their feature modules on demand."""

__all__ = ["Earth", "Environment", "PlanetaryBody", "DE421", "NRLMSISE00"]


def __getattr__(name):
    if name == "Environment":
        from .environment import Environment
        return Environment
    if name in ("Earth", "PlanetaryBody"):
        from . import planetary_body
        return getattr(planetary_body, name)
    if name == "DE421":
        from .ephemeris import DE421
        return DE421
    if name == "NRLMSISE00":
        from .atmosphere import NRLMSISE00
        return NRLMSISE00
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
