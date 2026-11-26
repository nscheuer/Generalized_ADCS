"""Configuration marker for Rust RK4 attitude propagation."""

import math


class RK4AttitudePropagator:
    """Select RK4 attitude evolution for a Rust split solver."""

    def __init__(self, dt: float) -> None:
        self.dt = float(dt)
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive seconds")
        self.mode = -1
