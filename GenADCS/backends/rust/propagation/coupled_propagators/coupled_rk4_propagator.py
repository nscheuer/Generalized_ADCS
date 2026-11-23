"""Fixed-step native coupled RK4 propagator."""
import numpy as np
from ._coupled_dynamics import _prepare
from ._state_layout import pack_state,unpack_state
from ._coupled import integrate_rk4,rk4_step,project


class CoupledRK4Propagator:
    def __init__(self,dt):
        self.dt=float(dt)
        if not np.isfinite(self.dt) or self.dt<=0:
            raise ValueError("dt must be finite and positive seconds")
        self._runtime=None
        self._configuration=None

    def reset_runtime(self):
        self._runtime=None
        self._configuration=None

    def integrate(self,state,satellite,environment,dynamics,tf,progress=False,progress_chunk_steps=500,record_every=1):
        runtime=_prepare(satellite,environment,dynamics)
        y=pack_state(state,runtime.wheel_commands.size)
        return integrate_rk4(y,state.orbit.julian_date,self.dt,float(tf),runtime,progress,progress_chunk_steps,record_every)

    def step(self,state,satellite,environment,dynamics):
        config=(id(satellite),id(environment),id(dynamics))
        if self._configuration!=config:
            self._runtime=_prepare(satellite,environment,dynamics)
            self._configuration=config
        y=pack_state(state,self._runtime.wheel_commands.size)
        y=project(y,self._runtime)
        result=rk4_step(y,state.orbit.julian_date,self.dt,self._runtime)
        return unpack_state(result,state,self.dt)
