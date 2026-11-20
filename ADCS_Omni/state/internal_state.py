from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class InternalState:
    wheel_momentum: object = field(default_factory=list)

    def __post_init__(self):
        momentum = np.asarray(self.wheel_momentum, dtype=np.float64)
        if momentum.ndim != 1 or not np.all(np.isfinite(momentum)):
            raise ValueError("wheel_momentum must be a finite 1D array")
        self.wheel_momentum = np.ascontiguousarray(momentum)
