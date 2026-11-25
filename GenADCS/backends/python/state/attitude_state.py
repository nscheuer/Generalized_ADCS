from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from GenADCS.backends.python.helpers.math.rotations.quaternion.quaternion import (
    Quaternion,
)


@dataclass(slots=True)
class AttitudeState:
    orientation: Quaternion | np.ndarray
    rates: np.ndarray

    def __post_init__(self) -> None:
        rates = np.asarray(self.rates, dtype=np.float64)
        if rates.shape != (3,) or not np.all(np.isfinite(rates)):
            raise ValueError("rates must be a finite 3-vector")
        self.rates = np.ascontiguousarray(rates)

    @property
    def quaternion_array(self) -> np.ndarray:
        if isinstance(self.orientation, Quaternion) or hasattr(self.orientation, "hamilton_components"):
            components = cast(Any, self.orientation).hamilton_components
            quaternion: np.ndarray = np.asarray(components, dtype=np.float64)
        else:
            quaternion = np.asarray(self.orientation, dtype=np.float64)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("orientation must resolve to a finite Hamilton quaternion")
        magnitude = np.linalg.norm(quaternion)
        if magnitude == 0.0:
            raise ValueError("orientation quaternion must be nonzero")
        return cast(np.ndarray, np.ascontiguousarray(quaternion / magnitude))
