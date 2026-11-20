from dataclasses import dataclass

from ADCS_Omni.environment import Environment
from ADCS_Omni.hardware import Satellite
from ADCS_Omni.propagation import Propagator
from ADCS_Omni.solver import Solver
from ADCS_Omni.state import State


@dataclass(slots=True)
class SimulationConfig:
    satellite: Satellite
    environment: Environment
    state: State
    propagator: Propagator
    solver: Solver
    backend: str = "Python"

    def __post_init__(self):
        if self.backend not in ("Python", "C", "Rust", "JAX"):
            raise ValueError("backend must be Python, C, Rust, or JAX")
