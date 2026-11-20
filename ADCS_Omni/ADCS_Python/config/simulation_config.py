from dataclasses import dataclass, field

import numpy as np

from ADCS_Omni.ADCS_Python.environment import Environment
from ADCS_Omni.ADCS_Python.hardware import Satellite
from ADCS_Omni.ADCS_Python.propagation import Propagator
from ADCS_Omni.ADCS_Python.solver import Solver
from ADCS_Omni.ADCS_Python.state import State


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

    def __post_init__(self):
        if self.backend not in ("Python", "C", "Rust", "JAX"):
            raise ValueError("backend must be Python, C, Rust, or JAX")
        self.state_record = self.state.to_record(len(self.satellite.adcs.reaction_wheels))
        self.julian_date = self.state.orbit.julian_date
