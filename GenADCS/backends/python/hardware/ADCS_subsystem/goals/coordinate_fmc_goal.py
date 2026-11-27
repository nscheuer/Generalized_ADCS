"""Coordinate-targeted ground-speed reduction (FMC) goal."""

import math
from typing import Any

import numpy as np

from .coordinate_goal import CoordinateGoal, _coordinate_fmc_rate
from .goal import validate_reference_state


class CoordinateFMCGoal(CoordinateGoal):
    """Tilt toward a geodetic coordinate with a selected FMC reduction level.

    ``fmc_level=n`` sets the camera tilt from nadir to ``acos(1 / n)``. This
    gives a transverse ground speed of ``v / n`` when the coordinate-defined
    tilt azimuth follows the ground track. The coordinate sets tilt azimuth;
    FMC level sets tilt magnitude.

    Parameters are latitude and longitude in degrees, altitude in kilometres,
    and an integer FMC level from 1 through 10.
    """

    kernel = staticmethod(_coordinate_fmc_rate)

    def __init__(
        self,
        lat: float,
        lon: float,
        alt: float = 0.0,
        fmc_level: int = 1,
    ) -> None:
        super().__init__(lat, lon, alt)
        if isinstance(fmc_level, bool) or not isinstance(
            fmc_level, (int, np.integer)
        ):
            raise ValueError("fmc_level must be an integer from 1 through 10")
        level = int(fmc_level)
        if not 1 <= level <= 10:
            raise ValueError("fmc_level must be an integer from 1 through 10")
        self.fmc_level = level

    def reference(
        self, state: np.ndarray, epoch_jd: float, runtime: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        state = validate_reference_state(state)
        epoch = float(epoch_jd)
        if not math.isfinite(epoch):
            raise ValueError("epoch_jd must be finite")
        return self.kernel(
            state, epoch, self.target_ecef_m, runtime.mu, self.fmc_level
        )


coordinatefmcgoal = CoordinateFMCGoal
