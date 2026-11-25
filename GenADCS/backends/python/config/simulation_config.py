from dataclasses import dataclass, field

import numpy as np

from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.hardware import Satellite
from GenADCS.backends.python.propagation import Propagator
from GenADCS.backends.python.solver import Solver
from GenADCS.backends.python.state import State


@dataclass(slots=True)
class SimulationConfig:
    satellite: Satellite
    environment: Environment
    state: State
    propagator: Propagator
    solver: Solver
    backend: str = "Python"
    state_record: np.ndarray = field(init=False, repr=False)
    julian_date: float = field(init=False)

    def __post_init__(self) -> None:
        if self.backend not in ("Python", "C", "Rust", "JAX"):
            raise ValueError("backend must be Python, C, Rust, or JAX")
        self.state_record = self.state.to_record(
            len(self.satellite.adcs.reaction_wheels)
        )
        self.julian_date = self.state.orbit.julian_date
