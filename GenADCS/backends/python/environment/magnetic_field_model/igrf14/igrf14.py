"""IGRF-14 internal field, 1900–2030, with Numba evaluation."""

from pathlib import Path

import numpy as np
from numba import njit

from ....helpers.math.time.conversions import calendar_to_jd
from ..magnetic_field_model import (
    MagneticFieldModel,
    magnetic_field_inertial_tesla,
    read_shc,
)

_KERNEL_DIR = (
    Path(__file__).resolve().parents[5]
    / "kernels/environment/magnetic_field_model/igrf14"
)


@njit(cache=True, nogil=True)
def igrf14_field_inertial_tesla(
    position_inertial_m: np.ndarray,
    epoch_jd: float,
    epochs: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """Compiled scalar GCRF field [T]; coefficients are loaded by IGRF14."""
    if epoch_jd < epochs[0] or epoch_jd > epochs[-1]:
        raise ValueError("IGRF-14 coefficients cover 1900-01-01 to 2030-01-01")
    k = np.searchsorted(epochs, epoch_jd, side="right") - 1
    if k >= len(epochs) - 1:
        k = len(epochs) - 2
    weight = (epoch_jd - epochs[k]) / (epochs[k + 1] - epochs[k])
    return magnetic_field_inertial_tesla(
        position_inertial_m,
        epoch_jd,
        g[k] + weight * (g[k + 1] - g[k]),
        h[k] + weight * (h[k + 1] - h[k]),
        13,
    )


class IGRF14(MagneticFieldModel):
    """IGRF-14 using bundled coefficients and paired UTC Julian date input."""

    def __init__(self) -> None:
        years, self._g, self._h = read_shc(str(_KERNEL_DIR / "IGRF14.shc"), 1, 13)
        self._epochs = np.array([calendar_to_jd(int(y), 1, 1) for y in years])

    def field_inertial_tesla(
        self, position_inertial_m: np.ndarray, epoch_jd: float
    ) -> np.ndarray:
        return igrf14_field_inertial_tesla(
            np.asarray(position_inertial_m, dtype=np.float64),
            float(epoch_jd),
            self._epochs,
            self._g,
            self._h,
        )
