from dataclasses import dataclass, field

from ADCS_Omni.ADCS_Python.config.dynamics_config import DynamicsConfig

from .attitude_solver import AttitudeSolver
from .coupled_solver import CoupledSolver
from .orbital_solver import OrbitalSolver


@dataclass(slots=True)
class Solver:
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    orbital: OrbitalSolver | None = None
    attitude: AttitudeSolver | None = None
    coupled: CoupledSolver | None = None

    def __post_init__(self):
        if (self.coupled is None and (self.orbital is None or self.attitude is None)) or (self.coupled is not None and (self.orbital is not None or self.attitude is not None)):
            raise ValueError("provide a coupled solver or both orbital and attitude solvers")
