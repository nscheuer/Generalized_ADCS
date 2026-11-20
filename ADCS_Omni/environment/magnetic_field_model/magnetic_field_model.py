"""Native internal geomagnetic field kernels. Input is GCRF metres and UTC JD."""
from functools import lru_cache

import numpy as np
from numba import njit

from ...helpers.math.frames.conversions import gcrf_to_ecef_matrix
from ...helpers.math.time.conversions import utc_jd_to_tt_jd
from ._spherical_harmonics import _eval_one, _schmidt_table



@lru_cache(maxsize=4)
def read_shc(path, nmin, nmax):
    """Read an SHC file and fold Schmidt factors into its coefficients."""
    with open(path, encoding="ascii") as source:
        lines = (line for line in source if line.strip() and not line.lstrip().startswith("#"))
        low, high, count, _, _ = map(int, next(lines).split()[:5])
        if (low, high) != (nmin, nmax):
            raise ValueError(f"Unexpected harmonic degrees in {path}")
        times = np.fromstring(next(lines), sep=" ")
        if len(times) != count or np.any(np.diff(times) <= 0):
            raise ValueError(f"Invalid SHC epochs in {path}")
        g = np.zeros((count, nmax + 1, nmax + 1))
        h = np.zeros_like(g)
        seen = set()
        for line in lines:
            values = np.fromstring(line, sep=" ")
            if len(values) != count + 2:
                raise ValueError(f"Malformed SHC row in {path}")
            n, signed_m = int(values[0]), int(values[1])
            if n < nmin or n > nmax or abs(signed_m) > n or (n, signed_m) in seen:
                raise ValueError(f"Invalid SHC coefficient in {path}")
            seen.add((n, signed_m))
            if signed_m >= 0:
                g[:, n, signed_m] = values[2:]
            else:
                h[:, n, -signed_m] = values[2:]
        if len(seen) != (nmax + 1)**2 - nmin**2:
            raise ValueError(f"Incomplete SHC file: {path}")
    scale = _schmidt_table(nmax)
    return times, np.ascontiguousarray(g * scale), np.ascontiguousarray(h * scale)


@njit(cache=True, nogil=True)
def magnetic_field_inertial_tesla(position_inertial_m, epoch_jd, g, h, nmax):
    """Evaluate Schmidt normalized internal Gauss coefficients in GCRF [T]."""
    jd_tt = utc_jd_to_tt_jd(epoch_jd) if epoch_jd >= 2441317.5 else epoch_jd
    rotation = gcrf_to_ecef_matrix(epoch_jd, jd_tt)
    fixed = rotation @ position_inertial_m
    radius = np.sqrt(np.dot(fixed, fixed))
    if radius <= 0.0:
        raise ValueError("position must have positive radius")
    theta = np.arccos(max(-1.0, min(1.0, fixed[2] / radius)))
    phi = np.arctan2(fixed[1], fixed[0])
    pc = np.zeros(nmax + 2); p1 = np.zeros(nmax + 2); p2 = np.zeros(nmax + 2)
    dc = np.zeros(nmax + 2); d1 = np.zeros(nmax + 2); d2 = np.zeros(nmax + 2)
    cosmp = np.zeros(nmax + 1); sinmp = np.zeros(nmax + 1)
    br, bt, bp = _eval_one(radius / 1000.0, theta * 180.0 / np.pi,
                            phi * 180.0 / np.pi, g, h, nmax,
                            pc, p1, p2, dc, d1, d2, cosmp, sinmp)
    st, ct, cp, sp = np.sin(theta), np.cos(theta), np.cos(phi), np.sin(phi)
    fixed_field = np.array((br * st * cp + bt * ct * cp - bp * sp,
                            br * st * sp + bt * ct * sp + bp * cp,
                            br * ct - bt * st)) * 1e-9
    return rotation.T @ fixed_field


class MagneticFieldModel:
    """Interface for GCRF magnetic vectors [T] at UTC Julian dates."""
    def field_inertial_tesla(self, position_inertial_m, epoch_jd):
        raise NotImplementedError("Choose a magnetic field model")

    def field(self, position_inertial_m, epoch_jd):
        return self.field_inertial_tesla(position_inertial_m, epoch_jd)
