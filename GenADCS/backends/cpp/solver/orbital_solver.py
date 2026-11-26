"""Native solver wrapper for an orbital propagator."""
from GenADCS.backends.cpp.propagation import _separate as _separate_native
from ._solver import OrbitalSolver as _NativeOrbitalSolver
from GenADCS.backends.python.propagation.coupled_propagators._state_layout import pack_state
from GenADCS.backends.python.state import OrbitalState, State
import numpy as np

class OrbitalSolver:
    def __init__(self, propagator):
        self.propagator = propagator
    @property
    def propagator(self):
        return self._propagator
    @propagator.setter
    def propagator(self, value):
        native_propagator = getattr(value, "_native", value)
        self._native = _NativeOrbitalSolver(native_propagator)
        self._propagator = value
    @property
    def dt(self):
        return self._native.dt
    def step(self, state, environment, dynamics):
        mu = float(environment.planetary_body.gravitational_parameter)
        values = self._native.step(pack_state(state, state.internal.wheel_momentum.size), mu)
        orbit = OrbitalState.from_cartesian(values[7:10], values[10:13],
            state.orbit.julian_date + self.dt / 86400.0, state.orbit.ephemeris)
        return State(state.attitude, orbit, state.internal)
    def integrate(self, state, environment, dynamics, tf, record_every=1):
        if isinstance(record_every, bool) or not isinstance(record_every, int) or record_every < 1:
            raise ValueError("record_every must be a positive integer")
        mu = float(environment.planetary_body.gravitational_parameter)
        times, values = self._native.integrate(
            pack_state(state, state.internal.wheel_momentum.size), mu, float(tf), record_every)
        return np.asarray(times, dtype=float), np.asarray(values, dtype=float)
