from dataclasses import dataclass, field

from GenADCS.backends.python.bias import Bias
from GenADCS.backends.python.noise import Noise

from .effects import apply_effects


@dataclass(slots=True)
class Magnetorquer:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float]
    max_dipole: float
    misalignment: tuple[float, float, float] = (0.0, 0.0, 0.0)
    effects: list[Noise | Bias] = field(default_factory=list)

    @property
    def noises(self):
        return [effect for effect in self.effects if isinstance(effect, Noise)]

    @property
    def biases(self):
        return [effect for effect in self.effects if isinstance(effect, Bias)]

    def apply_errors(self, value, dt, rng=None):
        return apply_effects(value, self.effects, dt, rng)
