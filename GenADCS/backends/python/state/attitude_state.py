from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AttitudeState:
    orientation: object
    rates: object

    def __post_init__(self) -> None:
        rates = np.asarray(self.rates, dtype=np.float64)
        if rates.shape != (3,) or not np.all(np.isfinite(rates)):
            raise ValueError("rates must be a finite 3-vector")
        self.rates = np.ascontiguousarray(rates)

    @property
    def quaternion_array(self) -> np.ndarray:
        if hasattr(self.orientation, "hamilton_components"):
            quaternion = np.asarray(
                self.orientation.hamilton_components, dtype=np.float64
            )
        else:
            quaternion = np.asarray(self.orientation, dtype=np.float64)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("orientation must resolve to a finite Hamilton quaternion")
        magnitude = np.linalg.norm(quaternion)
        if magnitude == 0.0:
            raise ValueError("orientation quaternion must be nonzero")
        return np.ascontiguousarray(quaternion / magnitude)
