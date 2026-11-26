"""Configuration marker for Rust reference attitude propagation."""

import math

_REFERENCE_MODES = {
    "nadir": 0,
    "nadir-pointing": 0,
    "along-track": 1,
    "sun": 2,
    "sun-pointing": 2,
}


class ReferenceAttitudePropagator:
    """Select a prescribed reference attitude for a Rust split solver."""

    def __init__(self, dt: float, mode: str) -> None:
        self.dt = float(dt)
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive seconds")
        try:
            self.mode = _REFERENCE_MODES[mode]
        except KeyError as exc:
            raise ValueError(f"unknown reference attitude mode: {mode!r}") from exc
