"""CHAOS-8.7 core and static lithospheric internal field."""

from pathlib import Path

import numpy as np
from numba import njit

from ..magnetic_field_model import (
    MagneticFieldModel,
    magnetic_field_inertial_tesla,
    read_shc,
)

_KERNEL_DIR = (
    Path(__file__).resolve().parents[5]
    / "kernels/environment/magnetic_field_model/chaos8"
)


@njit(cache=True, nogil=True)
def chaos8_field_inertial_tesla(
    position_inertial_m, epoch_jd, times, g, h, static_g, static_h, include_static
):
    """Compiled scalar GCRF field [T] with six-point core interpolation."""
    year = 2000.0 + (epoch_jd - 2451544.5) / 365.25
    if year < times[0] or year > times[-1]:
        raise ValueError("Epoch outside CHAOS-8.7 core coefficient range")
    start = (np.searchsorted(times[::5], year, side="right") - 1) * 5
    start = max(0, min(start, len(times) - 6))
    weights = np.empty(6)
    for j in range(6):
        weight = 1.0
        for k in range(6):
            if j != k:
                weight *= (year - times[start + k]) / (
                    times[start + j] - times[start + k]
                )
        weights[j] = weight
    nmax = 185 if include_static else 20
    gb = np.zeros((nmax + 1, nmax + 1))
    hb = np.zeros_like(gb)
    if include_static:
        gb += static_g
        hb += static_h
    for j in range(6):
        for n in range(1, 21):
            for m in range(n + 1):
                gb[n, m] += weights[j] * g[start + j, n, m]
                hb[n, m] += weights[j] * h[start + j, n, m]
    return magnetic_field_inertial_tesla(position_inertial_m, epoch_jd, gb, hb, nmax)


class CHAOS8(MagneticFieldModel):
    """CHAOS-8.7 internal field; optional degree 21–185 static field."""

    def __init__(self, include_static=True):
        self._times, self._g, self._h = read_shc(
            str(_KERNEL_DIR / "CHAOS-8.7_core.shc"), 1, 20
        )
        self.include_static = bool(include_static)
        if self.include_static:
            _, gs, hs = read_shc(str(_KERNEL_DIR / "CHAOS-8.7_static.shc"), 21, 185)
            self._static_g, self._static_h = gs[0], hs[0]
        else:
            self._static_g = np.zeros((1, 1))
            self._static_h = np.zeros((1, 1))

    def field_inertial_tesla(self, position_inertial_m, epoch_jd):
        return chaos8_field_inertial_tesla(
            np.asarray(position_inertial_m, dtype=np.float64),
            float(epoch_jd),
            self._times,
            self._g,
            self._h,
            self._static_g,
            self._static_h,
            self.include_static,
        )
