"""Rust-backed environment models, loaded from their feature modules on demand."""

__all__ = ["Earth", "Environment", "PlanetaryBody"]


def __getattr__(name):
    if name == "Environment":
        from .environment import Environment
        return Environment
    if name in ("Earth", "PlanetaryBody"):
        from . import planetary_body
        return getattr(planetary_body, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
