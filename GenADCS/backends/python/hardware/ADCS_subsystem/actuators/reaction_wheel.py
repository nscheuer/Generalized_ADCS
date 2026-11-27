from dataclasses import dataclass
import math


@dataclass(slots=True)
class ReactionWheel:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float]
    rotational_inertia: float
    max_torque: float
    max_momentum: float
    misalignment: tuple[float, float, float] = (0.0, 0.0, 0.0)
    noise: float = 0.0
    bias: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.noise) or self.noise < 0.0:
            raise ValueError("noise must be finite and nonnegative")
        if not math.isfinite(self.bias):
            raise ValueError("bias must be finite")
