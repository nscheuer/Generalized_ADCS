import math
from dataclasses import dataclass, field

from GenADCS.backends.python.config.dynamics_config import DynamicsConfig

from .attitude_solver import AttitudeSolver
from .coupled_solver import CoupledSolver
from .orbital_solver import OrbitalSolver


@dataclass(slots=True)
class Solver:
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    orbital: OrbitalSolver | None = None
    attitude: AttitudeSolver | None = None
    coupled: CoupledSolver | None = None
    tf: float | None = None
    progress: bool = False
    progress_chunk_steps: int = 500

    def __post_init__(self) -> None:
        if (
            self.coupled is None and (self.orbital is None or self.attitude is None)
        ) or (
            self.coupled is not None
            and (self.orbital is not None or self.attitude is not None)
        ):
            raise ValueError(
                "provide a coupled solver or both orbital and attitude solvers"
            )
        if self.coupled is None:
            if self.tf is None or not math.isfinite(float(self.tf)) or self.tf < 0:
                raise ValueError("separate solvers require a finite nonnegative tf")
            self.tf = float(self.tf)
            if (
                self.orbital is not None
                and self.attitude is not None
                and self.orbital.dt != self.attitude.dt
            ):
                raise ValueError("separate propagators require matching dt")
            if (
                isinstance(self.progress_chunk_steps, bool)
                or not isinstance(self.progress_chunk_steps, int)
                or self.progress_chunk_steps < 1
            ):
                raise ValueError("progress_chunk_steps must be a positive integer")
