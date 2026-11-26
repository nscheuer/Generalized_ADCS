"""Native uncoupled Keplerian orbit and RK4/reference attitude propagation."""
import math

from .coupled_propagators._coupled import integrate_uncoupled
from .coupled_propagators._coupled_dynamics import _prepare
from .coupled_propagators._state_layout import pack_state


def simulate_uncoupled(
    state,
    satellite,
    environment,
    dynamics,
    dt: float,
    tf: float,
    record_every: int = 1,
    progress: bool = False,
    progress_chunk_steps: int = 500,
    reference_mode: int = -1,
):
    """Run the full uncoupled orbit/attitude integration loop in Rust."""
    dt = float(dt)
    tf = float(tf)
    reference_mode = int(reference_mode)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive seconds")
    if not math.isfinite(tf) or tf < 0.0:
        raise ValueError("tf must be finite and nonnegative seconds")
    if reference_mode not in (-1, 0, 1, 2):
        raise ValueError("reference_mode must be -1, 0, 1, or 2")
    if isinstance(record_every, bool) or not isinstance(record_every, int) or record_every < 1:
        raise ValueError("record_every must be a positive integer")
    if (
        isinstance(progress_chunk_steps, bool)
        or not isinstance(progress_chunk_steps, int)
        or progress_chunk_steps < 1
    ):
        raise ValueError("progress_chunk_steps must be a positive integer")
    runtime = _prepare(satellite, environment, dynamics)
    y = pack_state(state, runtime.wheel_commands.size)
    return integrate_uncoupled(
        y,
        state.orbit.julian_date,
        dt,
        tf,
        runtime,
        reference_mode,
        progress,
        progress_chunk_steps,
        record_every,
    )


class UncoupledSolver:
    """Run the full uncoupled orbit/attitude integration loop in Rust."""

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
        return simulate_uncoupled(
            state,
            satellite,
            environment,
            dynamics,
            self.dt,
            tf,
            record_every=record_every,
            progress=progress,
            progress_chunk_steps=progress_chunk_steps,
            reference_mode=self.reference_mode,
        )


SeparateSolver = UncoupledSolver
SeparatePropagator = UncoupledSolver
