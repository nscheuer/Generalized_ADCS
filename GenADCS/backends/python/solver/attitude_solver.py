from dataclasses import dataclass

from GenADCS.backends.python.propagation import (
    AdaptiveRK45AttitudePropagator,
    RK4AttitudePropagator,
    ReferenceAttitudePropagator,
)


@dataclass(slots=True)
class AttitudeSolver:
    propagator: RK4AttitudePropagator | AdaptiveRK45AttitudePropagator | ReferenceAttitudePropagator

    @property
    def dt(self) -> float:
        return self.propagator.dt
