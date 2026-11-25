from dataclasses import dataclass

from GenADCS.backends.python.propagation import (
    KeplerianOrbitalPropagator,
    SGP4OrbitalPropagator,
)


@dataclass(slots=True)
class OrbitalSolver:
    propagator: KeplerianOrbitalPropagator | SGP4OrbitalPropagator

    @property
    def dt(self) -> float:
        return self.propagator.dt
