"""Adaptive native coupled DOP853 propagator."""
import numpy as np
from ._coupled_dynamics import _prepare
from ._state_layout import pack_state,unpack_state
from ._coupled import integrate_dop853


class CoupledDOP853Propagator:
    def __init__(self,dt,rtol=1e-9,atol=1e-12,max_step=np.inf,min_step=1e-6):
        self.dt=float(dt);self.rtol=float(rtol);self.max_step=float(max_step);self.min_step=float(min_step)
        absolute=np.asarray(atol,dtype=np.float64)
        self._scalar_atol=absolute.ndim==0
        if self._scalar_atol:absolute=np.full(13,float(absolute),dtype=np.float64)
        if absolute.ndim!=1 or absolute.size<13:raise ValueError("atol must be a scalar or vector with at least 13 elements")
        self.atol=np.ascontiguousarray(absolute)
        if not np.isfinite(self.dt) or self.dt<=0:raise ValueError("dt must be finite and positive seconds")
        if not np.isfinite(self.rtol) or self.rtol<=0:raise ValueError("rtol must be finite and positive")
        if not np.all(np.isfinite(self.atol)) or np.any(self.atol<=0):raise ValueError("atol must contain finite positive values")
        if self.max_step<=0 or np.isnan(self.max_step):raise ValueError("max_step must be positive")
        if not np.isfinite(self.min_step) or self.min_step<=0 or self.min_step>self.max_step:raise ValueError("min_step must be finite, positive, and no greater than max_step")

    def integrate(self,state,satellite,environment,dynamics,tf,progress=False,progress_chunk_steps=500,record_every=0):
        runtime=_prepare(satellite,environment,dynamics)
        y=pack_state(state,runtime.wheel_commands.size)
        atol=np.array([self.atol[0]]) if self._scalar_atol else self.atol
        return integrate_dop853(y,state.orbit.julian_date,self.dt,float(tf),runtime,self.rtol,atol,self.max_step,self.min_step,progress,progress_chunk_steps,record_every)

    def step(self,state,satellite,environment,dynamics):
        _,vectors=self.integrate(state,satellite,environment,dynamics,self.dt)
        return unpack_state(vectors[-1],state,self.dt)
