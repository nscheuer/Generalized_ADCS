from collections.abc import Iterable

import numpy as np

from GenADCS.backends.python.bias import Bias
from GenADCS.backends.python.noise import Noise


def apply_effects(
    value: np.ndarray,
    effects: Iterable[Noise | Bias],
    dt: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    for effect in effects:
        value = effect.apply(value, dt, rng)
    return value
