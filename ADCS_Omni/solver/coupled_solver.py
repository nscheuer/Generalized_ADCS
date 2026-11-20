from dataclasses import dataclass

from ADCS_Omni.propagation import CoupledPropagator


@dataclass(slots=True)
class CoupledSolver:
    propagator: CoupledPropagator

    @property
    def dt(self):
        return self.propagator.dt
