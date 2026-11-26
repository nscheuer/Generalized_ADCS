"""C++ universal-variable two-body orbital propagation."""
import math
import numpy as np
from GenADCS.backends.cpp.propagation._separate import KeplerianOrbitPropagator as _NativePropagator
from GenADCS.backends.python.propagation.coupled_propagators._state_layout import pack_state
from GenADCS.backends.python.state import OrbitalState, State

class KeplerianOrbitalPropagator:
    def __init__(self, dt):
        self.dt=float(dt)
        if not math.isfinite(self.dt) or self.dt<=0:
            raise ValueError("dt must be finite and positive seconds")
        self._native=_NativePropagator(self.dt)
    def integrate(self,state,environment,dynamics,tf,record_every=1):
        if isinstance(record_every,bool) or not isinstance(record_every,int) or record_every<1:
            raise ValueError("record_every must be a positive integer")
        mu=float(environment.planetary_body.gravitational_parameter)
        times,values=self._native.integrate(pack_state(state,state.internal.wheel_momentum.size),mu,float(tf),record_every)
        return np.asarray(times,dtype=float),np.asarray(values,dtype=float)
    def step(self,state,environment,dynamics):
        mu=float(environment.planetary_body.gravitational_parameter)
        y=self._native.step(pack_state(state,state.internal.wheel_momentum.size),mu)
        orbit=OrbitalState.from_cartesian(y[7:10],y[10:13],state.orbit.julian_date+self.dt/86400.0,state.orbit.ephemeris)
        return State(state.attitude,orbit,state.internal)
