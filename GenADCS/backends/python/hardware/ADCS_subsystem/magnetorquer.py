from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.random import Generator

from GenADCS.backends.python.bias.bias import Bias
from GenADCS.backends.python.noise.noise import Noise

from .effects import apply_effects


@dataclass(slots=True)
class Magnetorquer:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float]
    max_dipole: float
    misalignment: tuple[float, float, float] = (0.0, 0.0, 0.0)
    effects: list[Noise | Bias] = field(
        default_factory=lambda: cast(list[Noise | Bias], [])
    )

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
        rng: Generator | None = None,
    ) -> np.ndarray:
        return apply_effects(value, self.effects, dt, rng)
