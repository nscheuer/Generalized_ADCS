from dataclasses import dataclass

from ADCS_Omni.propagation import AdaptiveRK45AttitudePropagator, RK4AttitudePropagator


@dataclass(slots=True)
class AttitudeSolver:
    propagator: RK4AttitudePropagator | AdaptiveRK45AttitudePropagator

    @property
    def dt(self):
        return self.propagator.dt
