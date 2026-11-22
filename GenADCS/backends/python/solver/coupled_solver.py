from dataclasses import dataclass

import math

from GenADCS.backends.python.propagation import CoupledDOP853Propagator, CoupledRK4Propagator


@dataclass(slots=True)
class CoupledSolver:
    propagator: CoupledRK4Propagator | CoupledDOP853Propagator
    tf: float
    progress: bool = False
    progress_chunk_steps: int = 500

    def __post_init__(self):
        self.tf = float(self.tf)
        if not math.isfinite(self.tf) or self.tf < 0.0:
            raise ValueError("tf must be finite and nonnegative seconds")
        if isinstance(self.progress_chunk_steps, bool) or not isinstance(self.progress_chunk_steps, int) or self.progress_chunk_steps <= 0:
            raise ValueError("progress_chunk_steps must be a positive integer")

    @property
    def dt(self):
        return self.propagator.dt

    def step(self, state, satellite, environment, dynamics):
        """Advance the coupled orbit and attitude by one propagator step."""
        return self.propagator.step(state, satellite, environment, dynamics)
