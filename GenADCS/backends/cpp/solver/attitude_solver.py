"""Python wrapper for the native attitude-solver metadata."""

from ._solver import _AttitudeSolver


class AttitudeSolver:
    __slots__ = ("propagator", "_native")

    def __init__(self, propagator):
        self.propagator = propagator
        self._native = _AttitudeSolver(propagator.dt)

    @property
    def dt(self):
        return self._native.dt
