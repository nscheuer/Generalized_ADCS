"""Backward-compatible alias for :mod:`GenADCS.backends.rust.propagation.uncoupled`."""

from .uncoupled import SeparatePropagator, SeparateSolver, UncoupledSolver, simulate_uncoupled

__all__ = [
    "SeparatePropagator",
    "SeparateSolver",
    "UncoupledSolver",
    "simulate_uncoupled",
]
