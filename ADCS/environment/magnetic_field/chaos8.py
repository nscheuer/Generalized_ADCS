"""CHAOS-8.7 internal field from the DTU core and lithospheric SHC files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from numba import njit

from ADCS.environment.igrf import _eval_one, _schmidt_table

from ._coefficients import KERNEL_DIR, coordinate_dates, ensure_kernel, evaluate_static, shaped_field
from .magnetic_field import MagneticField


CHAOS8_VERSION = "8.7"
_URL_ROOT = "https://zenodo.org/records/22959079/files/"
CHAOS8_CORE_URL = _URL_ROOT + "CHAOS-8.7_core.shc?download=1"
CHAOS8_STATIC_URL = _URL_ROOT + "CHAOS-8.7_static.shc?download=1"


@lru_cache(maxsize=4)
def _read_shc(path: str, expected_min: int, expected_max: int):
    """Load a CHAOS SHC snapshot table, folding Schmidt factors into g/h."""
    with open(path, encoding="ascii") as stream:
        data = (line for line in stream if line.strip() and not line.lstrip().startswith("#"))
        nmin, nmax, ntimes, order, step = map(int, next(data).split()[:5])
        if (nmin, nmax) != (expected_min, expected_max):
            raise ValueError(f"Unexpected CHAOS degree range in {path}: {nmin}-{nmax}")
        if (nmin == 1 and (order, step) != (6, 5)) or (nmin == 21 and ntimes != 1):
            raise ValueError(f"Unsupported CHAOS snapshot layout in {path}")
        times = np.fromstring(next(data), sep=" ")
        if times.size != ntimes or not np.all(np.diff(times) > 0):
            raise ValueError(f"Invalid CHAOS time axis in {path}")
        g = np.zeros((ntimes, nmax + 1, nmax + 1))
        h = np.zeros_like(g)
        count = 0
        seen = set()
        for line in data:
            values = np.fromstring(line, sep=" ")
            if values.size != ntimes + 2:
                raise ValueError(f"Malformed CHAOS coefficient row in {path}")
            n, signed_m = int(values[0]), int(values[1])
            if n < nmin or n > nmax or abs(signed_m) > n or (n, signed_m) in seen:
                raise ValueError(f"Invalid CHAOS degree/order in {path}: {(n, signed_m)}")
            seen.add((n, signed_m))
            if signed_m >= 0:
                g[:, n, signed_m] = values[2:]
            else:
                h[:, n, -signed_m] = values[2:]
            count += 1
        if count != (nmax + 1) ** 2 - nmin ** 2:
            raise ValueError(f"Incomplete CHAOS coefficient file: {path}")
    s = _schmidt_table(nmax)
    return times, g * s[None, :, :], h * s[None, :, :]


def _chaos_years(dates):
    # ChaosMagPy maps SHC decimal years to MJD2000 using 365.25 days/year.
    days = (dates - np.datetime64("2000-01-01T00:00:00", "ns")) / np.timedelta64(1, "D")
    return 2000.0 + np.asarray(days, dtype=float) / 365.25


def _segment(times, years):
    # Each 6 snapshots define one degree-5 polynomial; adjacent segments
    # share the boundary snapshot (SHC N_STEP = 5).
    boundaries = times[::5]
    return np.clip(np.searchsorted(boundaries, years, side="right") - 1, 0, len(boundaries) - 2) * 5


@njit(cache=True, fastmath=True, nogil=True)
def _weights(times, start, year):
    weights = np.empty(6)
    for j in range(6):
        value = 1.0
        for k in range(6):
            if k != j:
                value *= (year - times[start + k]) / (times[start + j] - times[start + k])
        weights[j] = value
    return weights


@njit(cache=True, fastmath=True, nogil=True)
def _core_varying(r, theta, phi, times, years, starts, g, h, br, bt, bp):
    """Interpolate each core snapshot and evaluate its degree-20 field."""
    nmax = 20
    pc = np.zeros(nmax + 2); p1 = np.zeros(nmax + 2); p2 = np.zeros(nmax + 2)
    dc = np.zeros(nmax + 2); d1 = np.zeros(nmax + 2); d2 = np.zeros(nmax + 2)
    cosmp = np.zeros(nmax + 1); sinmp = np.zeros(nmax + 1)
    gb = np.zeros((nmax + 1, nmax + 1)); hb = np.zeros_like(gb)
    for i in range(r.size):
        start = starts[i]
        w = _weights(times, start, years[i])
        for n in range(1, nmax + 1):
            for m in range(n + 1):
                gv = 0.0; hv = 0.0
                for j in range(6):
                    gv += w[j] * g[start + j, n, m]
                    hv += w[j] * h[start + j, n, m]
                gb[n, m] = gv; hb[n, m] = hv
        br[i], bt[i], bp[i] = _eval_one(
            r[i], theta[i], phi[i], gb, hb, nmax,
            pc, p1, p2, dc, d1, d2, cosmp, sinmp,
        )


class CHAOS8(MagneticField):
    """CHAOS-8.7 core plus static lithospheric internal magnetic field.

    The two official SHC kernels are downloaded on first use. This evaluates
    CHAOS's *internal* field; its external and induced field terms require
    additional activity indices and are outside this model's interface.
    """

    name = "chaos8"

    def __init__(
        self,
        core_path: str | Path | None = None,
        static_path: str | Path | None = None,
        *,
        include_static: bool = True,
    ):
        self.core_path = Path(core_path) if core_path is not None else KERNEL_DIR / "CHAOS-8.7_core.shc"
        self.static_path = Path(static_path) if static_path is not None else KERNEL_DIR / "CHAOS-8.7_static.shc"
        self.include_static = bool(include_static)

    def field_gc(self, r, theta, phi, date):
        shape, r, theta, phi, dates = coordinate_dates(r, theta, phi, date)
        core_path = ensure_kernel(
            self.core_path, CHAOS8_CORE_URL, "a90a67e45d2a4613629d6c37fc9a0418"
        )
        times, g, h = _read_shc(str(core_path), 1, 20)
        years = _chaos_years(dates)
        if np.any((years < times[0]) | (years > times[-1])):
            raise ValueError(
                f"CHAOS-{CHAOS8_VERSION} core coefficients cover {times[0]:.3f} to {times[-1]:.3f}"
            )
        if dates.size == 1 or np.all(years == years[0]):
            start = int(_segment(times, years[:1])[0])
            weights = _weights(times, start, float(years[0]))
            gb = np.tensordot(weights, g[start:start + 6], axes=(0, 0))
            hb = np.tensordot(weights, h[start:start + 6], axes=(0, 0))
            components = evaluate_static(r, theta, phi, gb, hb)
        else:
            years = np.ascontiguousarray(years)
            if years.size != r.size:
                raise ValueError(f"date has {years.size} entries; expected 1 or {r.size}")
            starts = np.ascontiguousarray(_segment(times, years), dtype=np.int64)
            components = tuple(np.empty(r.size) for _ in range(3))
            _core_varying(r, theta, phi, times, years, starts, g, h, *components)

        if self.include_static:
            static_path = ensure_kernel(
                self.static_path, CHAOS8_STATIC_URL, "5730a1b3a155a698eae5e2f0b60f2433"
            )
            _, gs, hs = _read_shc(str(static_path), 21, 185)
            static_field = evaluate_static(r, theta, phi, gs[0], hs[0])
            components = tuple(core + lith for core, lith in zip(components, static_field))
        return shaped_field(shape, components)
