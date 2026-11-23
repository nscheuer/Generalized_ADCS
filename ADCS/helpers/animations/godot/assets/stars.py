r"""
Star table for the viewer from the Yale Bright Star Catalogue (BSC5).

Output columns (little-endian float32, one row per star), see ``COLUMNS``:
unit direction (ICRS/J2000, epoch 2000.0), V magnitude, B-V, linear sRGB
colour normalised to a maximum of 1, illuminance at the observer [lux], and
proper motion [mas/yr].
"""

__all__ = ["COLUMNS", "parse_bsc5", "bv_to_temperature", "temperature_to_linear_srgb",
           "magnitude_to_lux", "star_table"]

import gzip
from pathlib import Path
from typing import Dict, List

import numpy as np

COLUMNS = ("x", "y", "z", "vmag", "b_v", "r", "g", "b", "illuminance_lux", "pm_ra_mas", "pm_dec_mas")

#: Illuminance of a V = 0 star [lux]. Literature values span 2.0-2.6e-6; this
#: one reproduces the Sun (V = -26.74) at ~1.3e5 lux.
LUX_AT_V0 = 2.54e-6


def parse_bsc5(path: Path) -> Dict[str, np.ndarray]:
    r"""
    Parse the CDS V/50 ``catalog`` (plain or gzip), fixed-width per its ReadMe.

    Entries without a J2000 position (deleted or non-stellar objects) are
    skipped; a missing B-V becomes NaN.

    :return: ``{"hr", "ra_rad", "dec_rad", "vmag", "b_v", "pm_ra_mas", "pm_dec_mas"}``.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    rows: Dict[str, List[float]] = {k: [] for k in ("hr", "ra_rad", "dec_rad", "vmag", "b_v", "pm_ra_mas", "pm_dec_mas")}
    with opener(path, "rt", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n").ljust(197)
            if not line[75:77].strip() or not line[102:107].strip():
                continue

            def num(a: int, b: int) -> float:  # 1-based inclusive byte range, as in the ReadMe
                s = line[a - 1:b].strip()
                return float(s) if s else np.nan

            ra_h = num(76, 77) + num(78, 79) / 60.0 + num(80, 83) / 3600.0
            dec = num(85, 86) + num(87, 88) / 60.0 + num(89, 90) / 3600.0
            if line[83] == "-":
                dec = -dec
            rows["hr"].append(num(1, 4))
            rows["ra_rad"].append(np.radians(15.0 * ra_h))
            rows["dec_rad"].append(np.radians(dec))
            rows["vmag"].append(num(103, 107))
            rows["b_v"].append(num(110, 114))
            rows["pm_ra_mas"].append(1000.0 * num(149, 154))
            rows["pm_dec_mas"].append(1000.0 * num(155, 160))
    return {k: np.array(v, dtype=float) for k, v in rows.items()}


def bv_to_temperature(b_v: np.ndarray) -> np.ndarray:
    r"""Effective temperature [K] from B-V (Ballesteros 2012, EPL 97, 34008)."""
    b_v = np.asarray(b_v, dtype=float)
    return 4600.0 * (1.0 / (0.92 * b_v + 1.7) + 1.0 / (0.92 * b_v + 0.62))


def _cie_xyz_bar(lam_nm: np.ndarray) -> np.ndarray:
    # CIE 1931 2-degree colour matching functions, multi-lobe Gaussian fit
    # (Wyman, Sloan & Shirley 2013, JCGT 2(2)).
    def g(x, mu, s1, s2):
        s = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / s) ** 2)

    x = 1.056 * g(lam_nm, 599.8, 37.9, 31.0) + 0.362 * g(lam_nm, 442.0, 16.0, 26.7) - 0.065 * g(lam_nm, 501.1, 20.4, 26.2)
    y = 0.821 * g(lam_nm, 568.8, 46.9, 40.5) + 0.286 * g(lam_nm, 530.9, 16.3, 31.1)
    z = 1.217 * g(lam_nm, 437.0, 11.8, 36.0) + 0.681 * g(lam_nm, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


_XYZ_TO_LINEAR_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])


def temperature_to_linear_srgb(temp_k: np.ndarray) -> np.ndarray:
    r"""
    Chromaticity of a blackbody as linear sRGB, scaled so the largest channel is 1.

    Planck spectrum integrated against the CIE 1931 colour matching functions.
    Real stellar spectra deviate from blackbodies; this is the usual
    approximation for rendering.
    """
    temp_k = np.atleast_1d(np.asarray(temp_k, dtype=float))
    lam = np.arange(380.0, 781.0, 5.0)
    lam_m = lam * 1e-9
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    planck = 1.0 / (lam_m[None, :] ** 5 * np.expm1(h * c / (lam_m[None, :] * k * temp_k[:, None])))
    xyz = planck @ _cie_xyz_bar(lam)
    rgb = np.clip(xyz @ _XYZ_TO_LINEAR_SRGB.T, 0.0, None)
    return rgb / rgb.max(axis=1, keepdims=True)


def magnitude_to_lux(vmag: np.ndarray) -> np.ndarray:
    r"""Illuminance at the observer [lux] of a star of visual magnitude V."""
    return LUX_AT_V0 * 10.0 ** (-0.4 * np.asarray(vmag, dtype=float))


def star_table(cat: Dict[str, np.ndarray]) -> np.ndarray:
    r"""Rows of :data:`COLUMNS` as float32, brightest first."""
    ra, dec = cat["ra_rad"], cat["dec_rad"]
    xyz = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    b_v = cat["b_v"]
    # Stars without B-V get a solar-like colour.
    rgb = temperature_to_linear_srgb(bv_to_temperature(np.where(np.isfinite(b_v), b_v, 0.65)))
    table = np.column_stack([xyz, cat["vmag"], b_v, rgb, magnitude_to_lux(cat["vmag"]),
                             np.nan_to_num(cat["pm_ra_mas"]), np.nan_to_num(cat["pm_dec_mas"])])
    return table[np.argsort(cat["vmag"])].astype("<f4")
