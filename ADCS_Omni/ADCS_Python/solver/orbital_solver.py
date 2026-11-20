from dataclasses import dataclass

from ADCS_Omni.ADCS_Python.propagation import KeplerianOrbitalPropagator, SGP4OrbitalPropagator


@dataclass(slots=True)
class OrbitalSolver:
    propagator: KeplerianOrbitalPropagator | SGP4OrbitalPropagator

    @property
    def dt(self):
        return self.propagator.dt
