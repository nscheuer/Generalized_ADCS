import math
from dataclasses import dataclass

from GenADCS.backends.python.config.dynamics_config import DynamicsConfig
from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.hardware import Satellite
from GenADCS.backends.python.propagation import (
    CoupledDOP853Propagator,
    CoupledRK4Propagator,
)
from GenADCS.backends.python.state import State


@dataclass(slots=True)
class CoupledSolver:
    propagator: CoupledRK4Propagator | CoupledDOP853Propagator
    tf: float
    progress: bool = False
    progress_chunk_steps: int = 500

    def __post_init__(self) -> None:
        self.tf = float(self.tf)
        if not math.isfinite(self.tf) or self.tf < 0.0:
            raise ValueError("tf must be finite and nonnegative seconds")
        if (
            isinstance(self.progress_chunk_steps, bool)
            or self.progress_chunk_steps <= 0
        ):
            raise ValueError("progress_chunk_steps must be a positive integer")

    @property
    def dt(self) -> float:
        return self.propagator.dt

    def step(
        self,
        state: State,
        satellite: Satellite,
        environment: Environment,
        dynamics: DynamicsConfig,
    ) -> State:
        """Advance the coupled orbit and attitude by one propagator step."""
        return self.propagator.step(state, satellite, environment, dynamics)
