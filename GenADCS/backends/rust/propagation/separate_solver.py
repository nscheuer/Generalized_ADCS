"""Native Keplerian orbit and RK4/reference attitude propagation."""
import math

from .coupled_propagators._coupled import integrate_separate
from .coupled_propagators._coupled_dynamics import _prepare
from .coupled_propagators._state_layout import pack_state



class SeparateSolver:
    """Run the full split orbit/attitude integration loop in Rust."""

    def __init__(self, dt: float, reference_mode: int = -1) -> None:
        self.dt = float(dt)
        self.reference_mode = int(reference_mode)
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive seconds")
        if self.reference_mode not in (-1, 0, 1, 2):
            raise ValueError("reference_mode must be -1, 0, 1, or 2")

    def integrate(
        self,
        state,
        satellite,
        environment,
        dynamics,
        tf,
        progress=False,
        progress_chunk_steps=500,
        record_every=1,
    ):
        tf = float(tf)
        if not math.isfinite(tf) or tf < 0.0:
            raise ValueError("tf must be finite and nonnegative seconds")
        if isinstance(record_every, bool) or not isinstance(record_every, int) or record_every < 1:
            raise ValueError("record_every must be a positive integer")
        if (
            isinstance(progress_chunk_steps, bool)
            or not isinstance(progress_chunk_steps, int)
            or progress_chunk_steps < 1
        ):
            raise ValueError("progress_chunk_steps must be a positive integer")
        runtime = _prepare(satellite, environment, dynamics)
        y = pack_state(state, 0)
        return integrate_separate(
            y,
            state.orbit.julian_date,
            self.dt,
            tf,
            runtime,
            self.reference_mode,
            progress,
            progress_chunk_steps,
            record_every,
        )



SeparatePropagator = SeparateSolver
