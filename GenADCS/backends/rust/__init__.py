"""Rust-backed ADCS Omni components."""

__all__ = ["SimulationResult", "simulate"]


def __getattr__(name):
    if name in __all__:
        from importlib import import_module
        simulation = import_module(".simulate", __name__)
        return getattr(simulation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

