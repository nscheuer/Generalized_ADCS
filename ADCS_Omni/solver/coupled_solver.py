from dataclasses import dataclass

from ADCS_Omni.propagation import CoupledPropagator


@dataclass(slots=True)
class CoupledSolver:
    propagator: CoupledPropagator

    @property
    def dt(self):
        return self.propagator.dt

    def step(self, state, satellite, environment, dynamics):
        """Advance the coupled orbit and attitude by one fixed RK4 step."""
        return self.propagator.step(state, satellite, environment, dynamics)
