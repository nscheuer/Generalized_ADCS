from dataclasses import dataclass, field

import numpy as np

from GenADCS.backends.python.bias import Bias
from GenADCS.backends.python.noise import Noise

from .effects import apply_effects


@dataclass(slots=True)
class ReactionWheel:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float]
    rotational_inertia: float
    max_torque: float
    max_momentum: float
    misalignment: tuple[float, float, float] = (0.0, 0.0, 0.0)
    effects: list[Noise | Bias] = field(default_factory=list)

    @property
    def noises(self) -> list[Noise]:
        return [effect for effect in self.effects if isinstance(effect, Noise)]

    @property
    def biases(self) -> list[Bias]:
        return [effect for effect in self.effects if isinstance(effect, Bias)]

    def apply_errors(
        self,
        value: np.ndarray,
        dt: float,
        rng: object | None = None,
    ) -> np.ndarray:
        return apply_effects(value, self.effects, dt, rng)
